#!/usr/bin/env python3
"""
step1.py - Binary (.dbd/.ecd) to L0 NetCDF conversion.

Reads raw Slocum glider binary files and creates a single L0 timeseries NetCDF.
"""
import os
import sys
import glob
import time
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BINARY_DIR, CACHE_DIR, DEPLOY_YAML, OUTPUT_DIR,
    GPS_LAT_MIN, GPS_LAT_MAX, GPS_LON_MIN, GPS_LON_MAX,
    TEMP_MIN, TEMP_MAX, COND_MIN, PRES_MIN,
    PROFILE_FILT_SECS, PROFILE_MIN_SECS, GLIDER_ID,
)

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

try:
    import dbdreader
    HAS_DBDREADER = True
except ImportError:
    HAS_DBDREADER = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_all_params(mdb):
    pn = mdb.parameterNames
    if isinstance(pn, dict):
        all_params = set()
        for key in pn:
            all_params.update(pn[key])
        return all_params
    return set(pn)


def load_config(yaml_path):
    if not os.path.exists(yaml_path):
        print(f"  WARNING: deployment.yml not found at {yaml_path} "
              f"— using default metadata")
        return {
            "metadata": {
                "deployment_name": f"glider_{GLIDER_ID}",
                "glider_serial": GLIDER_ID,
                "glider_model": "Slocum",
                "institution": "INCOIS",
                "source": "Glider observations",
                "comment": "",
                "sea_name": "",
                "platform_type": "Slocum Glider",
            },
            "netcdf_variables": {},
        }
    if not HAS_YAML:
        print(f"  WARNING: PyYAML not installed — cannot read deployment.yml, "
              f"using default metadata")
        return {
            "metadata": {
                "deployment_name": f"glider_{GLIDER_ID}",
                "glider_serial": GLIDER_ID,
                "glider_model": "Slocum",
                "institution": "INCOIS",
                "source": "Glider observations",
                "comment": "",
                "sea_name": "",
                "platform_type": "Slocum Glider",
            },
            "netcdf_variables": {},
        }
    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"  WARNING: deployment.yml unreadable ({e}) "
              f"— using default metadata")
        return {
            "metadata": {
                "deployment_name": f"glider_{GLIDER_ID}",
                "glider_serial": GLIDER_ID,
                "glider_model": "Slocum",
                "institution": "INCOIS",
                "source": "Glider observations",
                "comment": "",
                "sea_name": "",
                "platform_type": "Slocum Glider",
            },
            "netcdf_variables": {},
        }


def _sanitize_cache_dir(cache_dir):
    """Remove .cac files containing non-ASCII bytes (dbdreader can't read them)."""
    if not os.path.isdir(cache_dir):
        return
    removed = 0
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".cac"):
            continue
        fp = os.path.join(cache_dir, fn)
        try:
            with open(fp, "rb") as fh:
                fh.read().decode("ascii")
        except (UnicodeDecodeError, OSError):
            try:
                os.remove(fp)
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"  Removed {removed} corrupted .cac file(s) from cache dir")


def _open_multidbd(filenames, cache_dir):
    """
    Open a dbdreader.MultiDBD robustly.
    Suppresses dbdreader's internal "File X could not be loaded" prints.
    Falls back to file-by-file on any open error (cache missing, non-ASCII,
    corrupt files).
    """
    import contextlib

    os.makedirs(cache_dir, exist_ok=True)
    _sanitize_cache_dir(cache_dir)

    if not filenames:
        raise RuntimeError("No files provided")

    devnull = open(os.devnull, "w")

    # Fast path: try all at once, suppress noisy prints
    try:
        with contextlib.redirect_stdout(devnull):
            result = dbdreader.MultiDBD(filenames=filenames, cacheDir=cache_dir)
        devnull.close()
        return result
    except UnicodeDecodeError:
        pass   # fall through to file-by-file
    except Exception as e:
        err_str = str(e).lower()
        if any(k in err_str for k in ("cache", "could not be found",
                                      "could not be loaded", "dbderror",
                                      "no loadable")):
            pass   # fall through to file-by-file
        else:
            devnull.close()
            raise
    finally:
        try:
            devnull.close()
        except Exception:
            pass

    # Slow path: one file at a time, suppress prints, categorize errors
    print(f"  Batch open failed — trying {len(filenames)} files one-by-one...")
    good_files = []
    counts = {"unicode": 0, "cache": 0, "corrupt": 0}

    devnull2 = open(os.devnull, "w")
    for fpath in filenames:
        try:
            with contextlib.redirect_stdout(devnull2):
                tmp = dbdreader.MultiDBD(filenames=[fpath], cacheDir=cache_dir)
            tmp.close()
            good_files.append(fpath)
        except UnicodeDecodeError:
            counts["unicode"] += 1
        except Exception as e:
            err_str = str(e).lower()
            if "cache" in err_str or "could not be found" in err_str:
                counts["cache"] += 1
            elif "could not be loaded" in err_str or "truncat" in err_str:
                counts["corrupt"] += 1
            else:
                good_files.append(fpath)   # unknown — include anyway
    devnull2.close()

    total_bad = sum(counts.values())
    if total_bad:
        parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
        print(f"  Skipped {total_bad} file(s): {', '.join(parts)}")
    if not good_files:
        raise RuntimeError("No loadable files found")

    print(f"  Loaded {len(good_files)}/{len(filenames)} files successfully")
    return dbdreader.MultiDBD(filenames=good_files, cacheDir=cache_dir)


def _filter_files_by_cache(pattern, cache_dir):
    """
    Return files matching pattern. On first run (empty cache), returns all files
    so dbdreader can build the cache. The ASCII safety check is handled by
    _filter_ascii_files inside _open_multidbd.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        return files

    # If cache is populated, only return files whose .cac exists
    cac_files = glob.glob(os.path.join(cache_dir, "*.cac"))
    if not cac_files:
        return files   # first run — return all, let _open_multidbd filter

    good_files = []
    skipped = 0
    for fpath in files:
        try:
            with open(fpath, "rb") as fh:
                header = fh.read(300).decode("ascii", errors="replace")
            key = None
            for line in header.splitlines():
                if "sensor_list_crc:" in line.lower():
                    key = line.split(":")[-1].strip().lower()
                    break
            if key is None:
                good_files.append(fpath)
                continue
            cac_path = os.path.join(cache_dir, key + ".cac")
            if os.path.exists(cac_path):
                good_files.append(fpath)
            else:
                skipped += 1
        except Exception:
            good_files.append(fpath)

    if skipped > 0:
        print(f"  WARNING: skipped {skipped} files with missing cache entries")
    return good_files


def read_flight(data_dir, cache_dir):
    # Accept both .dcd (compressed) and .dbd (full-res) flight files
    dcd_files = glob.glob(os.path.join(data_dir, "*.[dD][cC][dD]"))
    dbd_files = glob.glob(os.path.join(data_dir, "*.[dD][bB][dD]"))
    all_flight = dcd_files + dbd_files
    print(f"  Reading {len(all_flight)} flight files "
          f"({len(dcd_files)} .dcd, {len(dbd_files)} .dbd)...")

    if not all_flight:
        print("  WARNING: no flight files found")
        return {}

    # Filter to files whose cache entry exists — avoids crashing on mixed folders
    dcd_ok = _filter_files_by_cache(
        os.path.join(data_dir, "*.[dD][cC][dD]"), cache_dir)
    dbd_ok = _filter_files_by_cache(
        os.path.join(data_dir, "*.[dD][bB][dD]"), cache_dir)
    usable = dcd_ok + dbd_ok
    if len(usable) < len(all_flight):
        print(f"  Using {len(usable)}/{len(all_flight)} flight files (rest missing cache)")
    if not usable:
        print("  ERROR: no usable flight files after cache check")
        return {}

    try:
        mdb = _open_multidbd(usable, cache_dir)
    except Exception as e:
        print(f"  ERROR: could not open any flight files: {e}")
        return {}

    available = get_all_params(mdb)
    print(f"  {len(available)} parameters available")

    want = ["m_lat", "m_lon", "m_heading", "m_pitch", "m_roll",
            "m_depth", "m_pressure", "c_wpt_lat", "c_wpt_lon"]
    want = [v for v in want if v in available]

    data = {}
    for var in want:
        try:
            use_decimal = var in ("m_lat", "m_lon", "c_wpt_lat", "c_wpt_lon")
            t, v = mdb.get(var, decimalLatLon=use_decimal)
            good = np.isfinite(v) & np.isfinite(t)
            data[var] = (t[good], v[good])
            print(f"  {var}: {np.sum(good):,} points")
        except Exception as ex:
            print(f"  {var}: {ex}")

    mdb.close()
    return data


def read_science(data_dir, cache_dir):
    # Accept both .ecd (compressed) and .ebd (full-res) science files
    ecd_files = glob.glob(os.path.join(data_dir, "*.[eE][cC][dD]"))
    ebd_files = glob.glob(os.path.join(data_dir, "*.[eE][bB][dD]"))
    all_sci = ecd_files + ebd_files
    print(f"  Reading {len(all_sci)} science files "
          f"({len(ecd_files)} .ecd, {len(ebd_files)} .ebd)...")

    if not all_sci:
        print("  WARNING: no science files found")
        return {}

    ecd_ok = _filter_files_by_cache(
        os.path.join(data_dir, "*.[eE][cC][dD]"), cache_dir)
    ebd_ok = _filter_files_by_cache(
        os.path.join(data_dir, "*.[eE][bB][dD]"), cache_dir)
    usable = ecd_ok + ebd_ok
    if len(usable) < len(all_sci):
        print(f"  Using {len(usable)}/{len(all_sci)} science files (rest missing cache)")
    if not usable:
        print("  ERROR: no usable science files after cache check")
        return {}

    try:
        mec = _open_multidbd(usable, cache_dir)
    except Exception as e:
        print(f"  ERROR: could not open any science files: {e}")
        return {}

    available = get_all_params(mec)
    print(f"  {len(available)} parameters available")

    want = ["sci_water_temp", "sci_water_cond", "sci_water_pressure",
            "sci_oxy4_oxygen", "sci_oxy4_saturation",
            "sci_flbbcd_chlor_units", "sci_flbbcd_cdom_units",
            "sci_flbbcd_bb_units"]
    want = [v for v in want if v in available]

    data = {}
    for var in want:
        try:
            t, v = mec.get(var)
            good = np.isfinite(v) & np.isfinite(t)
            data[var] = (t[good], v[good])
            print(f"  {var}: {np.sum(good):,} points")
        except Exception as ex:
            print(f"  {var}: {ex}")

    mec.close()
    return data


def filter_gps(flight):
    print("  Filtering GPS data...")
    for var in ["m_lat", "m_lon", "c_wpt_lat", "c_wpt_lon"]:
        if var not in flight:
            continue
        t, v = flight[var]
        n_before = len(v)
        good = np.ones(len(v), dtype=bool)
        # Basic sanity: reject null-island and physically impossible values
        good &= np.abs(v) > 0.01
        good &= np.abs(v) < (90 if "lat" in var else 180)

        # GPS bounding box: ONLY apply if bounds were actually detected
        # (not global defaults) AND we had enough fixes to trust the box.
        # With < 50 fixes from .mlg headers, the box is unreliable for
        # long deployments — skip it and let the hemisphere filter handle
        # gross errors later.
        bounds_are_global = (GPS_LAT_MIN <= -89 and GPS_LAT_MAX >= 89)
        if not bounds_are_global:
            # Use a very generous approach: only reject points that are
            # clearly outside the deployment region (>10° beyond detected box)
            margin = 10.0  # degrees — generous enough for any real transit
            if var == "m_lat":
                good &= (v >= GPS_LAT_MIN - margin) & (v <= GPS_LAT_MAX + margin)
            elif var == "m_lon":
                good &= (v >= GPS_LON_MIN - margin) & (v <= GPS_LON_MAX + margin)
            elif var == "c_wpt_lat":
                good &= (v >= GPS_LAT_MIN - margin) & (v <= GPS_LAT_MAX + margin)
            elif var == "c_wpt_lon":
                good &= (v >= GPS_LON_MIN - margin) & (v <= GPS_LON_MAX + margin)

        flight[var] = (t[good], v[good])
        n_removed = n_before - len(v[good])
        if n_removed > 0:
            print(f"  {var}: removed {n_removed}/{n_before} bad points")

    return flight


def filter_science(science):
    print("  Pre-filtering science data...")
    if "sci_water_temp" in science:
        t, v = science["sci_water_temp"]
        good = (v >= TEMP_MIN) & (v <= TEMP_MAX)
        science["sci_water_temp"] = (t[good], v[good])
        print(f"  temperature: removed {len(v) - np.sum(good)} out-of-range")

    if "sci_water_cond" in science:
        t, v = science["sci_water_cond"]
        good = (v >= COND_MIN) & (v < 10.0)
        science["sci_water_cond"] = (t[good], v[good])
        print(f"  conductivity: removed {len(v) - np.sum(good)} out-of-range")

    if "sci_water_pressure" in science:
        t, v = science["sci_water_pressure"]
        good = (v >= PRES_MIN / 10.0) & (v < 200.0)
        science["sci_water_pressure"] = (t[good], v[good])
        print(f"  pressure: removed {len(v) - np.sum(good)} out-of-range")

    return science


def sync_data(flight, science):
    """
    Sync all variables onto a common time axis.

    The master time axis is the UNION of all sensor timestamps (science +
    flight), not just sci_water_temp. This ensures GPS fixes acquired at the
    surface (when CTD is off) are preserved in the output.
    """
    print("  Syncing all data onto unified time axis...")

    # Build master time from ALL available timestamps
    all_times = []
    for var, (t, v) in science.items():
        if len(t) > 0:
            all_times.append(t)
    for var, (t, v) in flight.items():
        if len(t) > 0:
            all_times.append(t)

    if not all_times:
        raise ValueError("No sensor data available to build time axis")

    master_t = np.unique(np.concatenate(all_times))
    master_t = np.sort(master_t)
    # Remove duplicates (timestamps within 0.01s of each other)
    unique_mask = np.concatenate([[True], np.diff(master_t) > 0.01])
    master_t = master_t[unique_mask]

    # Remove obviously invalid timestamps (before year 2000 or after 2035)
    t_min_valid = 946684800.0    # 2000-01-01 UTC
    t_max_valid = 2051222400.0   # 2035-01-01 UTC
    time_valid = (master_t >= t_min_valid) & (master_t <= t_max_valid)

    # If deployment window is known, use it (with ±7 day margin) for tighter filtering
    # This prevents cross-deployment contamination from binary files.
    # NOTE: must import the module (not the variable) to get the runtime-updated value.
    try:
        import config as _cfg
        deploy_start = getattr(_cfg, 'DEPLOY_TIME_START', None)
        deploy_end = getattr(_cfg, 'DEPLOY_TIME_END', None)
        margin_s = 7 * 86400  # 7 days margin
        applied_window = False
        if deploy_start is not None:
            dt_start = np.datetime64(deploy_start).astype("datetime64[s]").astype(float)
            time_valid &= (master_t >= dt_start - margin_s)
            applied_window = True
        if deploy_end is not None:
            dt_end = np.datetime64(deploy_end).astype("datetime64[s]").astype(float)
            time_valid &= (master_t <= dt_end + margin_s)
            applied_window = True
        if applied_window:
            print(f"  Using deployment-window filter: {deploy_start} to {deploy_end} (±7d)")
        else:
            print(f"  No deployment window set — using broad 2000-2035 filter only")
    except Exception as e:
        print(f"  WARNING: deployment-window filter failed ({type(e).__name__}: {e}) "
              f"— using broad 2000-2035 filter only")

    n_bad_t = int(np.sum(~time_valid))
    if n_bad_t > 0:
        print(f"  WARNING: filtered {n_bad_t} timestamps outside deployment window")
        master_t = master_t[time_valid]

    n = len(master_t)

    print(f"  Master time: {n:,} points (union of all sensors)")
    print(f"  Duration: {(master_t[-1] - master_t[0]) / 86400:.1f} days")

    synced = {"time": master_t}

    for var, (t, v) in science.items():
        if len(t) < 2:
            continue
        order = np.argsort(t)
        t, v = t[order], v[order]
        umask = np.concatenate([[True], np.diff(t) > 0])
        t, v = t[umask], v[umask]
        if len(t) < 2:
            continue
        f = interp1d(t, v, bounds_error=False, fill_value=np.nan)
        synced[var] = f(master_t)
        print(f"  {var}: {np.sum(np.isfinite(synced[var])):,}/{n:,} valid")

    for var, (t, v) in flight.items():
        if len(t) < 2:
            continue
        order = np.argsort(t)
        t, v = t[order], v[order]
        umask = np.concatenate([[True], np.diff(t) > 0])
        t, v = t[umask], v[umask]
        if len(t) < 2:
            continue
        f = interp1d(t, v, bounds_error=False, fill_value=np.nan)
        synced[var] = f(master_t)
        print(f"  {var}: {np.sum(np.isfinite(synced[var])):,}/{n:,} valid")

    return synced


def derive_variables(synced):
    print("  Computing derived variables...")

    if "sci_water_pressure" in synced:
        synced["pressure_dbar"] = synced["sci_water_pressure"] * 10.0

    if "pressure_dbar" in synced:
        lat_mean = float(np.nanmean(synced.get("m_lat", np.array([12.0]))))
        if HAS_GSW:
            synced["depth"] = -gsw.z_from_p(synced["pressure_dbar"], lat_mean)
        else:
            synced["depth"] = synced["pressure_dbar"] * 1.019716

    if HAS_GSW and all(k in synced for k in ["sci_water_cond", "sci_water_temp", "pressure_dbar"]):
        C = synced["sci_water_cond"] * 10
        T = synced["sci_water_temp"]
        P = synced["pressure_dbar"]
        try:
            SP = gsw.SP_from_C(C, T, P)
            synced["salinity"] = SP
            lon_mean = float(np.nanmean(synced.get("m_lon", np.array([70.0]))))
            lat_mean = float(np.nanmean(synced.get("m_lat", np.array([12.0]))))
            SA = gsw.SA_from_SP(SP, P, lon_mean, lat_mean)
            CT = gsw.CT_from_t(SA, T, P)
            synced["potential_temperature"] = CT
            synced["density"] = gsw.rho(SA, CT, P)
            synced["potential_density"] = gsw.sigma0(SA, CT) + 1000
            print("  salinity, potential_temperature, density: computed")
        except Exception as ex:
            print(f"  Salinity calc failed: {ex}")

    return synced


def detect_profiles(synced):
    print("  Detecting profiles...")
    if "pressure_dbar" not in synced:
        print("  SKIP: no pressure")
        return synced

    t = synced["time"]
    p = synced["pressure_dbar"].copy()
    valid = np.isfinite(p)
    if np.sum(valid) < 100:
        print("  SKIP: too few pressure points")
        return synced

    p_filled = p.copy()
    p_filled[~valid] = np.interp(t[~valid], t[valid], p[valid])

    dt = float(np.median(np.diff(t)))
    n_smooth = max(1, int(PROFILE_FILT_SECS / max(dt, 0.1)))
    n_smooth = min(n_smooth, len(p_filled) // 2)
    if n_smooth > 1:
        kernel = np.ones(n_smooth) / n_smooth
        p_smooth = np.convolve(p_filled, kernel, mode="same")
    else:
        p_smooth = p_filled

    dp = np.gradient(p_smooth, t)
    direction = np.sign(dp)

    idx = np.zeros(len(t), dtype=int)
    cur = 0
    seg_start = 0
    for i in range(1, len(direction)):
        if direction[i] != direction[i - 1] and direction[i] != 0:
            if (t[i] - t[seg_start]) >= PROFILE_MIN_SECS:
                cur += 1
                seg_start = i
        idx[i] = cur

    synced["profile_index"] = idx.astype(float)
    synced["profile_direction"] = direction
    print(f"  {len(np.unique(idx))} profiles detected")
    return synced


def write_netcdf(synced, config, output_path):
    print("  Writing L0 NetCDF...")
    meta = config.get("metadata", {})
    ncvar = config.get("netcdf_variables", {})

    # Filter out invalid timestamps before conversion.
    # Valid Slocum timestamps are between 2000 and 2030 (epoch seconds).
    t_raw = synced["time"]
    t_min = 946684800.0    # 2000-01-01 UTC
    t_max = 1893456000.0   # 2030-01-01 UTC
    valid_time = (t_raw >= t_min) & (t_raw <= t_max) & np.isfinite(t_raw)

    if not np.all(valid_time):
        n_bad = int(np.sum(~valid_time))
        print(f"  WARNING: removing {n_bad} invalid timestamps "
              f"(out of {len(t_raw)} total)")
        # Filter all arrays in synced to keep only valid timestamps
        for key in synced:
            if isinstance(synced[key], np.ndarray) and len(synced[key]) == len(t_raw):
                synced[key] = synced[key][valid_time]
        t_raw = synced["time"]

    time_dt = np.array([np.datetime64(int(t * 1e9), "ns") for t in t_raw])
    ds = xr.Dataset(coords={"time": time_dt})

    vmap = [
        ("m_lat", "latitude"),
        ("m_lon", "longitude"),
        ("m_heading", "heading"),
        ("m_pitch", "pitch"),
        ("m_roll", "roll"),
        ("c_wpt_lat", "waypoint_latitude"),
        ("c_wpt_lon", "waypoint_longitude"),
        ("sci_water_temp", "temperature"),
        ("sci_water_cond", "conductivity"),
        ("pressure_dbar", "pressure"),
        ("sci_oxy4_oxygen", "oxygen_concentration"),
        ("sci_flbbcd_chlor_units", "chlorophyll"),
        ("sci_flbbcd_cdom_units", "cdom"),
        ("sci_flbbcd_bb_units", "backscatter_700"),
        ("depth", "depth"),
        ("salinity", "salinity"),
        ("potential_temperature", "potential_temperature"),
        ("density", "density"),
        ("potential_density", "potential_density"),
        ("profile_index", "profile_index"),
        ("profile_direction", "profile_direction"),
    ]

    for internal, nc_name in vmap:
        if internal not in synced:
            continue
        attrs = {}
        if nc_name in ncvar:
            for k, v in ncvar[nc_name].items():
                if k not in ("source", "coordinates", "conversion"):
                    attrs[k] = v
        ds[nc_name] = xr.DataArray(synced[internal].astype(np.float64), dims=["time"], attrs=attrs)

    # Check optics calibration: Slocum's sci_flbbcd_*_units should already be
    # in physical units (mg/m³, ppb, m⁻¹) from firmware calibration. But if
    # values look like raw counts (chlorophyll max > 50 mg/m³), flag them.
    _optics_cal_warning = False
    for nc_name, max_plausible in [("chlorophyll", 50.0),
                                    ("cdom", 200.0),
                                    ("backscatter_700", 0.05),
                                    ("oxygen_concentration", 500.0)]:
        if nc_name in ds:
            vmax = float(np.nanmax(ds[nc_name].values))
            if vmax > max_plausible:
                _optics_cal_warning = True
                ds[nc_name].attrs["comment"] = (
                    f"WARNING: max value ({vmax:.1f}) exceeds plausible "
                    f"physical range. Values may be uncalibrated raw counts "
                    f"or corrupted. Verify sensor calibration coefficients.")
                print(f"  WARNING: {nc_name} max={vmax:.1f} exceeds "
                      f"plausible range ({max_plausible}) — may be uncalibrated")
            else:
                ds[nc_name].attrs["comment"] = (
                    "Firmware-calibrated values from glider sensor. "
                    "No additional post-processing calibration applied.")

    if "latitude" in ds and "longitude" in ds:
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        dlat = np.diff(np.nan_to_num(lat, nan=0))
        dlon = np.diff(np.nan_to_num(lon, nan=0))
        cos_lat = np.cos(np.radians(float(np.nanmean(lat))))
        dd = np.sqrt((dlat * 111320) ** 2 + (dlon * 111320 * cos_lat) ** 2)
        dist = np.concatenate([[0], np.cumsum(dd)])
        ds["distance_over_ground"] = xr.DataArray(dist, dims=["time"],
            attrs={"long_name": "distance over ground", "units": "m"})

    ds.attrs = {
        "Conventions": "CF-1.8",
        "title": f"Glider {meta.get('glider_serial', GLIDER_ID)} L0 Timeseries",
        "institution": meta.get("institution", "INCOIS"),
        "source": meta.get("source", "Glider observations"),
        "comment": meta.get("comment", ""),
        "deployment_name": meta.get("deployment_name", ""),
        "glider_serial": meta.get("glider_serial", ""),
        "glider_model": meta.get("glider_model", ""),
        "platform_type": meta.get("platform_type", "Slocum Glider"),
        "sea_name": meta.get("sea_name", ""),
        "processing_level": "L0 - raw decoded data, no QC applied",
        "date_created": datetime.utcnow().isoformat() + "Z",
    }

    ds.to_netcdf(output_path)
    print(f"  L0 saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    print(f"  Observations: {len(time_dt):,}")
    ds.close()
    return output_path


def run_step1():
    print("=" * 60)
    print("  STEP 1: Binary -> L0 NetCDF")
    print("=" * 60)
    t0 = time.time()

    if not HAS_DBDREADER:
        print("ERROR: dbdreader not installed. pip install dbdreader")
        sys.exit(1)

    if not os.path.exists(BINARY_DIR):
        print(f"ERROR: Binary directory not found: {BINARY_DIR}")
        sys.exit(1)

    config = load_config(DEPLOY_YAML)
    flight = read_flight(BINARY_DIR, CACHE_DIR)
    flight = filter_gps(flight)
    science = read_science(BINARY_DIR, CACHE_DIR)
    science = filter_science(science)
    synced = sync_data(flight, science)
    synced = derive_variables(synced)
    synced = detect_profiles(synced)

    output_path = os.path.join(OUTPUT_DIR, f"incois_glider_{GLIDER_ID}_L0.nc")
    write_netcdf(synced, config, output_path)

    print(f"\n  STEP 1 COMPLETE in {time.time() - t0:.1f}s")
    return output_path


if __name__ == "__main__":
    run_step1()
