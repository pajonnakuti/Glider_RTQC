#!/usr/bin/env python3
"""
step23.py - L0 to L1 processing with GliderTools QC + ARGO flags.

Combines pre-cleaning, optics correction, physics QC, oxygen lag correction,
and complete ARGO RTQC flagging.
"""
import os
import sys
import time
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import OUTPUT_DIR, GLIDER_ID, MAX_DEPTH_DBAR, OXYGEN_TAU

PHYS_TEMP_MIN, PHYS_TEMP_MAX = -2.5, 40.0
PHYS_SAL_MIN, PHYS_SAL_MAX = 2.0, 41.0
PHYS_OXY_MIN, PHYS_OXY_MAX = -5.0, 600.0

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================
# PRE-CLEANING: Remove factory/lab tests before QC
# ============================================================

def pre_clean(ds):
    """Remove pre-deployment factory tests, zero GPS, and cross-hemisphere data."""
    # Read from config at call time so detect_deployment() values are used
    clean_factory  = config.CLEAN_FACTORY_TESTS
    fact_lat_min   = config.FACTORY_LAT_MIN
    fact_lat_max   = config.FACTORY_LAT_MAX
    fact_lon_min   = config.FACTORY_LON_MIN
    fact_lon_max   = config.FACTORY_LON_MAX
    clean_zero     = config.CLEAN_ZERO_GPS
    clean_hemi     = config.CLEAN_HEMISPHERE
    clean_year     = config.CLEAN_MODE_YEAR

    n_orig = len(ds.time)

    if "latitude" in ds and "longitude" in ds:
        if clean_factory:
            mask = ~((ds.latitude > fact_lat_min) & (ds.latitude < fact_lat_max)
                     & (ds.longitude > fact_lon_min) & (ds.longitude < fact_lon_max))
            ds = ds.where(mask, drop=True)

        if clean_zero:
            ds = ds.where(~((ds.latitude == 0) & (ds.longitude == 0)), drop=True)

    if "waypoint_latitude" in ds and "waypoint_longitude" in ds:
        if clean_factory:
            wp_mask = ~((ds["waypoint_latitude"] > fact_lat_min)
                        & (ds["waypoint_latitude"] < fact_lat_max)
                        & (ds["waypoint_longitude"] > fact_lon_min)
                        & (ds["waypoint_longitude"] < fact_lon_max))
            ds["waypoint_latitude"]  = ds["waypoint_latitude"].where(wp_mask)
            ds["waypoint_longitude"] = ds["waypoint_longitude"].where(wp_mask)

    ds = ds.sortby("time")

    if clean_year and HAS_PANDAS:
        t_dt = pd.Series(ds.time.values)
        if len(t_dt) > 0:
            mode_year = t_dt.dt.year.mode()[0]
            ds = ds.sel(time=((ds.time.dt.year == mode_year) |
                              (ds.time.dt.year == mode_year - 1)))

    if clean_hemi and "latitude" in ds:
        median_lat = float(np.nanmedian(ds.latitude.values))
        if median_lat < 0:
            ds = ds.where(ds.latitude < 0, drop=True)
            if "waypoint_latitude" in ds:
                ds["waypoint_latitude"]  = ds["waypoint_latitude"].where(ds["waypoint_latitude"] < 0)
                ds["waypoint_longitude"] = ds["waypoint_longitude"].where(ds["waypoint_latitude"] < 0)
        else:
            ds = ds.where(ds.latitude > 0, drop=True)
            if "waypoint_latitude" in ds:
                ds["waypoint_latitude"]  = ds["waypoint_latitude"].where(ds["waypoint_latitude"] > 0)
                ds["waypoint_longitude"] = ds["waypoint_longitude"].where(ds["waypoint_latitude"] > 0)

    n_clean = len(ds.time)
    print(f"  Pre-cleaning: {n_orig} -> {n_clean} observations "
          f"(removed {n_orig - n_clean})")
    return ds


# ============================================================
# GLIDERTOOLS-STYLE QC FUNCTIONS
# ============================================================

def outlier_bounds_iqr(data, multiplier=1.5):
    out = data.copy()
    valid = out[np.isfinite(out)]
    if len(valid) < 10:
        return out, 0
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
    bad = (out < lo) | (out > hi)
    n_bad = int(np.sum(bad))
    out[bad] = np.nan
    return out, n_bad


def despike_median(data, window=5):
    baseline = data.copy()
    valid = np.isfinite(baseline)
    if np.sum(valid) < window:
        return baseline, np.zeros_like(data)
    filled = baseline.copy()
    if np.any(~valid):
        filled[~valid] = np.interp(
            np.flatnonzero(~valid), np.flatnonzero(valid),
            baseline[valid], left=np.nan, right=np.nan)
    med = median_filter(filled, size=window)
    spikes = data - med
    baseline = np.where(valid, med, np.nan)
    return baseline, spikes


def smooth_savgol_per_profile(data, profile_index, window=11, order=2, depth=None):
    out = data.copy()
    valid_mask = np.isfinite(data) & np.isfinite(profile_index)
    if np.sum(valid_mask) < window:
        return out
    profiles = np.unique(profile_index[valid_mask])
    for p in profiles:
        p_mask = valid_mask & (profile_index == p)
        p_idx = np.where(p_mask)[0]
        p_vals = data[p_idx]
        if len(p_vals) < window:
            continue
        p_valid = np.isfinite(p_vals)
        if depth is not None:
            p_depth = depth[p_idx]
            depth_order = np.argsort(p_depth)
        else:
            depth_order = np.arange(len(p_vals))
        p_sorted = p_vals[depth_order]
        s_valid = p_valid[depth_order]
        filled = p_sorted.copy()
        if np.any(~s_valid):
            filled[~s_valid] = np.interp(
                np.flatnonzero(~s_valid), np.flatnonzero(s_valid),
                p_sorted[s_valid], left=np.nan, right=np.nan)
        if len(filled) >= window:
            smoothed = savgol_filter(filled, window, order)
            undo = np.argsort(depth_order)
            out[p_idx] = np.where(p_valid, smoothed[undo], np.nan)
    return out


def backscatter_zhang_correction(beta, temperature, salinity,
                                  theta_deg=124, wavelength=700, chi_factor=1.076):
    out = np.full_like(beta, np.nan)
    valid = np.isfinite(beta) & np.isfinite(temperature) & np.isfinite(salinity)
    if np.sum(valid) < 2:
        return out
    T = temperature[valid]
    S = salinity[valid]
    lam = wavelength
    lam_ref = 500.0
    beta_sw_ref = 1.38e-4 * (lam_ref / lam) ** 4.32
    beta_sw = beta_sw_ref * (1 + 0.3 * S / 37.0) * (1 + (T - 20) * 0.002)
    beta_p = beta[valid] - beta_sw
    bbp = 2 * np.pi * chi_factor * beta_p
    out[valid] = bbp
    return out


def insitu_dark_count(data, depth, deep_min=200, deep_max=400, percentile=95,
                      shallow_threshold=100):
    valid = np.isfinite(data) & np.isfinite(depth)
    max_valid_depth = float(np.nanmax(depth[valid])) if np.any(valid) else 0.0
    deep = valid & (depth >= deep_min) & (depth <= deep_max)
    used_fallback = False
    if np.sum(deep) < 5:
        if np.sum(valid) < 10:
            return data, False
        depth_thresh = np.percentile(depth[valid], 90)
        if depth_thresh < shallow_threshold:
            return data, False
        deep = valid & (depth >= depth_thresh)
        used_fallback = True
    if np.sum(deep) < 3:
        return data, False
    dark = np.percentile(data[deep], percentile)
    out = data.copy()
    out[valid] = data[valid] - dark
    return out, used_fallback


def find_bad_profiles(data, depth, profile_index, depth_threshold=300, multiplier=2.0):
    mask = np.zeros(len(data), dtype=bool)
    valid = np.isfinite(data) & np.isfinite(depth) & np.isfinite(profile_index)
    if np.sum(valid) < 10:
        return mask, 0
    profiles = np.unique(profile_index[valid])
    deep_means = {}
    for p in profiles:
        in_prof = valid & (profile_index == p) & (depth > depth_threshold)
        if np.sum(in_prof) >= 3:
            deep_means[p] = np.nanmean(data[in_prof])
    if len(deep_means) < 5:
        return mask, 0
    means = np.array(list(deep_means.values()))
    med = np.median(means)
    std = np.std(means)
    threshold = med + multiplier * std
    n_flagged = 0
    for p, m in deep_means.items():
        if m > threshold:
            prof_mask = (profile_index == p)
            mask[prof_mask] = True
            n_flagged += int(np.sum(prof_mask))
    return mask, n_flagged


def horizontal_diff_outliers(data, depth, profile_index, depth_bins=None,
                              max_frac=0.1, multiplier=3.0):
    mask = np.zeros(len(data), dtype=bool)
    valid = np.isfinite(data) & np.isfinite(depth) & np.isfinite(profile_index)
    if np.sum(valid) < 20:
        return mask, 0
    if depth_bins is None:
        depth_bins = np.arange(0, np.nanmax(depth[valid]) + 10, 10)
    profiles = np.sort(np.unique(profile_index[valid]))
    if len(profiles) < 5:
        return mask, 0
    n_prof = len(profiles)
    n_bins = len(depth_bins) - 1
    vi = np.where(valid)[0]
    pi_arr = np.searchsorted(profiles, profile_index[vi])
    di_arr = np.clip(np.searchsorted(depth_bins, depth[vi]) - 1, 0, n_bins - 1)
    sums = np.zeros((n_prof, n_bins))
    counts = np.zeros((n_prof, n_bins))
    np.add.at(sums, (pi_arr, di_arr), data[vi])
    np.add.at(counts, (pi_arr, di_arr), 1)
    grid = np.where(counts > 0, sums / counts, np.nan)
    hdiff = np.abs(np.diff(grid, axis=0))
    all_diffs = hdiff[np.isfinite(hdiff)]
    if len(all_diffs) < 10:
        return mask, 0
    med_diff = np.median(all_diffs)
    mad = np.median(np.abs(all_diffs - med_diff))
    threshold = med_diff + multiplier * 1.4826 * mad
    n_flagged = 0
    for i in range(hdiff.shape[0]):
        row = hdiff[i]
        n_valid_bins = np.sum(np.isfinite(row))
        if n_valid_bins < 3:
            continue
        frac = np.sum(row > threshold) / n_valid_bins
        if frac > max_frac:
            for pi in [i, i + 1]:
                p = profiles[pi]
                prof_mask = (profile_index == p)
                mask[prof_mask] = True
                n_flagged += int(np.sum(prof_mask))
    return mask, n_flagged


def quenching_correction(chl, bbp, depth, time_vals, latitude, longitude, profile_index):
    chl_corr = chl.copy()
    valid = (np.isfinite(chl) & np.isfinite(bbp) & np.isfinite(depth)
             & np.isfinite(profile_index))
    if np.sum(valid) < 50:
        return chl_corr, 0
    try:
        time_ns = time_vals.astype("datetime64[ns]")
        day_start = time_ns.astype("datetime64[D]")
        hour_utc = (time_ns - day_start) / np.timedelta64(1, "h")
    except Exception:
        return chl_corr, 0
    mean_lon = float(np.nanmean(longitude)) if np.any(np.isfinite(longitude)) else 75.0
    solar_hour = (hour_utc + mean_lon / 15.0) % 24.0
    is_day = (solar_hour > 6) & (solar_hour < 18)
    profiles = np.unique(profile_index[valid])
    night_ratios = []
    for p in profiles:
        in_prof = valid & (profile_index == p)
        if np.sum(in_prof) < 5 or np.mean(is_day[in_prof]) > 0.5:
            continue
        subsurface = in_prof & (depth >= 20) & (depth <= 200) & (bbp > 1e-5)
        if np.sum(subsurface) < 3:
            continue
        ratio = np.nanmedian(chl[subsurface] / bbp[subsurface])
        if np.isfinite(ratio) and ratio > 0:
            night_ratios.append(ratio)
    if len(night_ratios) < 3:
        return chl_corr, 0
    ref_ratio = np.median(night_ratios)
    n_corrected = 0
    for p in profiles:
        in_prof = valid & (profile_index == p)
        if np.sum(in_prof) < 5 or np.mean(is_day[in_prof]) <= 0.5:
            continue
        prof_depth = depth[in_prof]
        prof_chl = chl[in_prof]
        prof_bbp = bbp[in_prof]
        prof_ratio = np.where(prof_bbp > 1e-5, prof_chl / prof_bbp, np.nan)
        quench_candidates = (prof_depth < 200) & np.isfinite(prof_ratio) & (prof_ratio < 0.5 * ref_ratio)
        if not np.any(quench_candidates):
            continue
        quench_depth = np.max(prof_depth[quench_candidates])
        indices = np.where(in_prof)[0]
        for idx in indices:
            if depth[idx] < quench_depth and bbp[idx] > 1e-5:
                corrected = bbp[idx] * ref_ratio
                if corrected > chl[idx]:
                    chl_corr[idx] = corrected
                    n_corrected += 1
    return chl_corr, n_corrected


# ============================================================
# PROCESSING PIPELINES
# ============================================================

def apply_optics_correction(ds):
    print("  Optics Processing (GliderTools-style)...")
    depth = ds["depth"].values if "depth" in ds else ds["pressure"].values
    profile_index = ds["profile_index"].values if "profile_index" in ds else np.zeros(len(ds.time))

    if "chlorophyll" in ds:
        chl = ds["chlorophyll"].values.copy()
        chl, n_iqr = outlier_bounds_iqr(chl, multiplier=3.0)
        chl, fb = insitu_dark_count(chl, depth)
        if fb:
            print("    WARNING: chlorophyll dark count - no deep data, using fallback")
        chl = np.where(np.isfinite(chl), np.maximum(chl, 0.0), np.nan)  # clip negatives
        chl_base, chl_spikes = despike_median(chl, window=7)
        ds["chlorophyll"] = xr.DataArray(chl_base, dims=["time"],
            attrs={"long_name": "chlorophyll", "units": "mg m-3",
                   "comment": "QC: IQR(3x), dark count corrected, clipped to 0, despiked"})
        ds["chlorophyll_spikes"] = xr.DataArray(chl_spikes, dims=["time"],
            attrs={"long_name": "chlorophyll spikes", "units": "mg m-3"})
        print(f"    chlorophyll: IQR removed {n_iqr}, dark count corrected, despiked")

    if "cdom" in ds:
        cdom = ds["cdom"].values.copy()
        cdom, n_iqr = outlier_bounds_iqr(cdom, multiplier=3.0)
        cdom, fb = insitu_dark_count(cdom, depth)
        if fb:
            print("    WARNING: cdom dark count - no deep data, using fallback")
        cdom = np.where(np.isfinite(cdom), np.maximum(cdom, 0.0), np.nan)  # clip negatives
        cdom_base, _ = despike_median(cdom, window=7)
        ds["cdom"] = xr.DataArray(cdom_base, dims=["time"],
            attrs={"long_name": "CDOM", "units": "ppb",
                   "comment": "QC: IQR(3x), dark count corrected, clipped to 0, despiked"})
        print(f"    cdom: IQR removed {n_iqr}, processed")

    if "backscatter_700" in ds:
        bb = ds["backscatter_700"].values.copy()
        bb, n_iqr = outlier_bounds_iqr(bb, multiplier=3.0)
        temp = ds["temperature"].values if "temperature" in ds else None
        sal = ds["salinity"].values if "salinity" in ds else None
        if temp is not None and sal is not None:
            bbp = backscatter_zhang_correction(bb, temp, sal)
            bbp, fb = insitu_dark_count(bbp, depth)
            if fb:
                print("    WARNING: backscatter dark count - no deep data, using fallback")
            bbp_base, bbp_spikes = despike_median(bbp, window=7)
            ds["backscatter_700"] = xr.DataArray(bbp_base, dims=["time"],
                attrs={"long_name": "particulate backscatter 700nm", "units": "m-1",
                       "comment": "Zhang et al. (2009) corrected, dark count removed, despiked"})
            ds["backscatter_700_spikes"] = xr.DataArray(bbp_spikes, dims=["time"],
                attrs={"long_name": "backscatter spikes", "units": "m-1"})
            print(f"    backscatter: IQR removed {n_iqr}, Zhang corrected, dark count, despiked")
        else:
            bb_base, _ = despike_median(bb, window=7)
            ds["backscatter_700"] = xr.DataArray(bb_base, dims=["time"],
                attrs={"long_name": "backscatter 700nm", "units": "1",
                       "comment": "Despiked only (no T/S for Zhang correction)"})
            print("    backscatter: despiked only (no T/S available)")

    if "profile_index" in ds:
        prof_idx = ds["profile_index"].values
        for var in ["backscatter_700", "chlorophyll"]:
            if var in ds:
                bad_mask, n_bad = find_bad_profiles(
                    ds[var].values, depth, prof_idx, depth_threshold=300, multiplier=2.0)
                if n_bad > 0:
                    vals = ds[var].values.copy()
                    vals[bad_mask] = np.nan
                    ds[var].values = vals
                    print(f"    {var}: flagged {n_bad} pts in bad profiles")

    if all(v in ds for v in ["chlorophyll", "backscatter_700", "profile_index"]):
        lat = ds["latitude"].values if "latitude" in ds else np.full(len(ds.time), 12.0)
        lon = ds["longitude"].values if "longitude" in ds else np.full(len(ds.time), 75.0)
        chl_corr, n_corr = quenching_correction(
            ds["chlorophyll"].values, ds["backscatter_700"].values,
            depth, ds.time.values, lat, lon, ds["profile_index"].values)
        if n_corr > 0:
            ds["chlorophyll_unquenched"] = ds["chlorophyll"].copy()
            ds["chlorophyll"] = xr.DataArray(chl_corr, dims=["time"],
                attrs={**ds["chlorophyll"].attrs,
                       "comment": ds["chlorophyll"].attrs.get("comment", "") +
                       ", quenching corrected (Thomalla et al. 2017)"})
            print(f"    chlorophyll: quenching corrected {n_corr} points")

    return ds


def apply_physics_qc(ds):
    print("  Physics QC (GliderTools-style)...")
    profile_index = ds["profile_index"].values if "profile_index" in ds else np.zeros(len(ds.time))
    depth = ds["depth"].values if "depth" in ds else ds["pressure"].values if "pressure" in ds else None

    # For glider T/S/O2 we do NOT use IQR — it is fundamentally wrong for full-depth
    # profiles where cold deep water dominates the distribution and would flag warm
    # surface water as outliers. Instead we use:
    #   1. Physical range limits (removes truly impossible values)
    #   2. Median despike (removes point-to-point spikes within a profile)
    #   3. Savitzky-Golay smoothing per profile (reduces sensor noise)
    # IQR is reserved for optical variables (chlorophyll, backscatter) where the
    # distribution is more uniform and global outliers are meaningful.

    phys_limits = {
        "temperature":          (PHYS_TEMP_MIN, PHYS_TEMP_MAX),
        "salinity":             (PHYS_SAL_MIN,  PHYS_SAL_MAX),
        "oxygen_concentration": (PHYS_OXY_MIN,  PHYS_OXY_MAX),
    }

    for var, sg_win in [
        ("temperature",          11),
        ("salinity",             11),
        ("oxygen_concentration", 11),
    ]:
        if var not in ds:
            continue
        vals = ds[var].values.copy()
        n_orig = int(np.sum(np.isfinite(vals)))

        # Step 1: physical range guard
        if var in phys_limits:
            lo_phys, hi_phys = phys_limits[var]
            n_phys = int(np.sum(np.isfinite(vals) & ((vals < lo_phys) | (vals > hi_phys))))
            vals[(vals < lo_phys) | (vals > hi_phys)] = np.nan
            if n_phys > 0:
                print(f"    {var}: physical range removed {n_phys}")

        # Step 2: median despike (catches point spikes, not vertical structure)
        vals, spikes = despike_median(vals, window=5)

        # Step 3: Savitzky-Golay smoothing per profile (depth-sorted)
        vals = smooth_savgol_per_profile(vals, profile_index, window=sg_win, order=2, depth=depth)

        # Clip oxygen to physically meaningful minimum
        if var == "oxygen_concentration":
            vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), np.nan)

        attrs = dict(ds[var].attrs) if var in ds else {}
        attrs["comment"] = (
            f"QC: physical range check, median despiked (w=5), "
            f"SG smoothed per-profile (w={sg_win})"
        )
        ds[var] = xr.DataArray(vals, dims=["time"], attrs=attrs)
        n_final = int(np.sum(np.isfinite(vals)))
        print(f"    {var}: {n_orig} -> {n_final} valid")

    if "salinity" in ds and "profile_index" in ds:
        sal_depth = ds["depth"].values if "depth" in ds else ds["pressure"].values
        sal_mask, n_hdiff = horizontal_diff_outliers(
            ds["salinity"].values, sal_depth, ds["profile_index"].values,
            max_frac=0.5, multiplier=6.0)
        if n_hdiff > 0:
            sal_vals = ds["salinity"].values.copy()
            sal_vals[sal_mask] = np.nan
            ds["salinity"].values = sal_vals
            print(f"    salinity: horizontal diff flagged {n_hdiff} pts")

    return ds


def oxygen_lag_correction(ds):
    print("  Applying Oxygen Lag Correction...")
    if "oxygen_concentration" not in ds:
        print("   Oxygen variable not found.")
        return ds

    tau = OXYGEN_TAU
    t = ds.time.values.astype("datetime64[s]").astype(float)
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0.0)
    oxy = ds["oxygen_concentration"].values
    doxy = np.diff(oxy)
    doxy = np.insert(doxy, 0, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        oxy_rate = doxy / dt
    oxy_rate[dt <= 0] = 0
    oxy_rate[dt > 60] = 0
    oxy_rate[np.isnan(oxy_rate)] = 0
    oxy_rate[np.isinf(oxy_rate)] = 0

    if "profile_index" in ds:
        pi = ds["profile_index"].values
        dpi = np.diff(pi)
        dpi = np.insert(dpi, 0, 0)
        oxy_rate[dpi != 0] = 0

    oxy_corrected = oxy + (tau * oxy_rate)
    oxy_corrected[oxy_corrected < 0] = 0.0
    oxy_corrected[~np.isfinite(oxy)] = np.nan

    ds["oxygen_concentration_lag_corrected"] = xr.DataArray(
        oxy_corrected, dims=["time"],
        attrs={"long_name": "lag corrected oxygen concentration",
               "units": "umol l-1",
               "comment": f"First-order lag correction, tau={tau}s, profile-aware"})
    print(f"   Added oxygen_concentration_lag_corrected (tau={tau}s)")
    return ds


# ============================================================
# ARGO RTQC TESTS
# ============================================================

def test_impossible_date(ds, qc_dict):
    flagged = 0
    try:
        time_ns = ds.time.values.astype("datetime64[ns]")
        juld_ref = np.datetime64("1997-01-01", "ns")
        now = np.datetime64("now", "ns")
        bad = (time_ns < juld_ref) | (time_ns > now)
        n_bad = int(np.sum(bad))
        if n_bad > 0:
            for var in qc_dict:
                qc_dict[var][bad] = 4
            flagged = n_bad
    except Exception:
        pass
    return flagged


def test_impossible_location(ds, qc_dict):
    lat = ds["latitude"].values if "latitude" in ds else None
    lon = ds["longitude"].values if "longitude" in ds else None
    if lat is not None and "latitude" in qc_dict:
        bad_lat = ~np.isfinite(lat) | (np.abs(lat) > 90)
        qc_dict["latitude"][bad_lat] = 4
    if lon is not None and "longitude" in qc_dict:
        bad_lon = ~np.isfinite(lon) | (np.abs(lon) > 180)
        qc_dict["longitude"][bad_lon] = 4
    return 0


def test_impossible_speed(ds, qc_dict, max_speed_ms=3.0):
    lat = ds["latitude"].values if "latitude" in ds else None
    lon = ds["longitude"].values if "longitude" in ds else None
    if lat is None or lon is None:
        return 0
    time_s = ds.time.values.astype("datetime64[s]").astype(float)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(time_s)
    if np.sum(valid) < 2:
        return 0
    lat_v = lat[valid]
    lon_v = lon[valid]
    t_v = time_s[valid]
    dlat = np.diff(lat_v) * 111320
    dlon = np.diff(lon_v) * 111320 * np.cos(np.radians(float(np.mean(lat_v))))
    dist = np.sqrt(dlat**2 + dlon**2)
    dt = np.diff(t_v)
    dt[dt <= 0] = 1.0
    speed = dist / dt
    bad_speed = speed > max_speed_ms
    valid_idx = np.where(valid)[0]
    flagged = 0
    for i in np.where(bad_speed)[0]:
        idx1 = valid_idx[i]
        idx2 = valid_idx[i + 1]
        for var in qc_dict:
            qc_dict[var][idx1] = 3
            qc_dict[var][idx2] = 3
        flagged += 2
    return flagged


def test_global_range(ds, qc_dict):
    ranges = {
        "temperature": (-2.5, 40.0),
        "salinity": (2.0, 41.0),
        "pressure": (-5.0, 2000.0),
        "oxygen_concentration": (-5.0, 500.0),
        "chlorophyll": (-0.5, 50.0),
        "cdom": (-5.0, 375.0),
        "backscatter_700": (-0.01, 0.1),
        "density": (1000.0, 1060.0),
    }
    flagged = 0
    for var, (vmin, vmax) in ranges.items():
        if var not in ds:
            continue
        vals = ds[var].values
        bad = (vals < vmin) | (vals > vmax)
        n_bad = int(np.sum(bad))
        if var in qc_dict:
            qc_dict[var][bad] = 4
        if var == "pressure" and n_bad > 0:
            pres_bad = vals < -5.0
            for v2 in ["temperature", "salinity"]:
                if v2 in qc_dict:
                    qc_dict[v2][pres_bad] = 4
        flagged += n_bad
    return flagged


def test_pressure_increasing(ds, qc_dict, reversal_threshold=20.0):
    if "pressure" not in ds:
        return 0
    p = ds["pressure"].values.copy()
    valid = np.isfinite(p)
    if np.sum(valid) < 10:
        return 0
    profile_index = ds["profile_index"].values if "profile_index" in ds else np.zeros(len(p))
    profiles = np.unique(profile_index[valid])
    bad = np.zeros(len(p), dtype=bool)
    for prof in profiles:
        prof_mask = (profile_index == prof) & valid
        prof_idx = np.where(prof_mask)[0]
        if len(prof_idx) < 4:
            continue
        mid = len(prof_idx) // 2
        segment = prof_idx[:mid][::-1]
        if len(segment) > 1:
            p_seg = p[segment]
            running_min = p_seg[0]
            for i in range(1, len(segment)):
                if p_seg[i] >= running_min + reversal_threshold:
                    bad[segment[i]] = True
                running_min = min(running_min, p_seg[i])
        segment = prof_idx[mid:]
        if len(segment) > 1:
            p_seg = p[segment]
            running_max = p_seg[0]
            for i in range(1, len(segment)):
                if p_seg[i] <= running_max - reversal_threshold:
                    bad[segment[i]] = True
                running_max = max(running_max, p_seg[i])
    n_bad = int(np.sum(bad))
    if "pressure" in qc_dict:
        qc_dict["pressure"][bad] = 4
    return n_bad


def test_spike(ds, qc_dict):
    spike_thresholds = {
        "temperature": {"shallow": 6.0, "deep": 2.0},
        "salinity": {"shallow": 0.9, "deep": 0.3},
        "pressure": {"shallow": 20.0, "deep": 20.0},
        "oxygen_concentration": {"shallow": 50.0, "deep": 50.0},
    }
    pvals = ds["pressure"].values if "pressure" in ds else None
    profile_index = ds["profile_index"].values if "profile_index" in ds else np.zeros(len(ds.time))
    total_flagged = 0
    for var, thresh in spike_thresholds.items():
        if var not in ds:
            continue
        vals = ds[var].values
        bad = np.zeros(len(vals), dtype=bool)
        profiles = np.unique(profile_index[np.isfinite(profile_index)])
        for prof in profiles:
            prof_mask = (profile_index == prof)
            prof_idx = np.where(prof_mask)[0]
            if len(prof_idx) < 3:
                continue
            for i in range(1, len(prof_idx) - 1):
                idx_prev = prof_idx[i - 1]
                idx_curr = prof_idx[i]
                idx_next = prof_idx[i + 1]
                v1 = vals[idx_prev]
                v2 = vals[idx_curr]
                v3 = vals[idx_next]
                if not (np.isfinite(v1) and np.isfinite(v2) and np.isfinite(v3)):
                    continue
                spike = abs(v2 - (v3 + v1) / 2.0) - abs((v3 - v1) / 2.0)
                if pvals is not None:
                    th = thresh["shallow"] if pvals[idx_curr] < 500 else thresh["deep"]
                else:
                    th = thresh["shallow"]
                if spike > th:
                    bad[idx_curr] = True
        n_bad = int(np.sum(bad))
        if var in qc_dict:
            qc_dict[var][bad] = 4
        total_flagged += n_bad
    return total_flagged


def test_stuck_value(ds, qc_dict, run_length=10):
    stuck_exempt = {"chlorophyll", "cdom", "backscatter_700"}
    total_flagged = 0
    for var in ["temperature", "salinity", "pressure", "oxygen_concentration", "density"]:
        if var not in ds or var in stuck_exempt:
            continue
        vals = ds[var].values
        n = len(vals)
        if n < run_length + 1:
            continue
        diffs = np.diff(vals)
        zero_diffs = np.concatenate([[False], diffs == 0])
        kernel = np.ones(run_length)
        runs = np.convolve(zero_diffs.astype(float), kernel, mode="same")
        bad = runs >= run_length
        bad[:run_length] = False
        bad[-run_length:] = False
        n_bad = int(np.sum(bad))
        if var in qc_dict:
            qc_dict[var][bad] = 4
        total_flagged += n_bad
    return total_flagged


def test_density_inversion(ds, qc_dict, threshold=0.03):
    if not HAS_GSW:
        return 0
    if not all(v in ds for v in ["temperature", "salinity", "pressure"]):
        return 0
    T = ds["temperature"].values
    SP = ds["salinity"].values
    P = ds["pressure"].values
    profile_index = ds["profile_index"].values if "profile_index" in ds else np.zeros_like(T)
    lat_m = float(np.nanmean(ds["latitude"].values)) if "latitude" in ds else 12.0
    lon_m = float(np.nanmean(ds["longitude"].values)) if "longitude" in ds else 75.0
    with np.errstate(invalid="ignore"):
        SA = gsw.SA_from_SP(SP, P, lon_m, lat_m)
        CT = gsw.CT_from_t(SA, T, P)
        sigma0 = gsw.sigma0(SA, CT)
    inv_mask = np.zeros(len(sigma0), dtype=bool)
    profiles = np.unique(profile_index[np.isfinite(profile_index)])
    for prof in profiles:
        prof_mask = (profile_index == prof)
        prof_idx = np.where(prof_mask)[0]
        if len(prof_idx) < 2:
            continue
        for i in range(len(prof_idx) - 1):
            idx_curr = prof_idx[i]
            idx_next = prof_idx[i + 1]
            dp = P[idx_next] - P[idx_curr]
            dsig = sigma0[idx_next] - sigma0[idx_curr]
            if np.isfinite(dp) and np.isfinite(dsig):
                if dp > 1 and dsig < -threshold:
                    inv_mask[idx_curr] = True
                if dp < -1 and dsig > threshold:
                    inv_mask[idx_curr] = True
    n_bad = int(np.sum(inv_mask))
    for v in ["temperature", "salinity", "density"]:
        if v in qc_dict:
            qc_dict[v][inv_mask] = 4
    return n_bad


def test_gross_sensor_drift(ds, qc_dict):
    if "profile_index" not in ds:
        return 0
    profile_index = ds["profile_index"].values
    depth = ds["depth"].values if "depth" in ds else ds["pressure"].values
    profiles = np.sort(np.unique(profile_index[np.isfinite(profile_index)]))
    if len(profiles) < 2:
        return 0
    flagged = 0
    deep_means_s = {}
    deep_means_t = {}
    for p in profiles:
        p_mask = (profile_index == p)
        p_depth = depth[p_mask]
        p_valid = np.isfinite(p_depth)
        if "salinity" in ds:
            p_sal = ds["salinity"].values[p_mask]
            deep_mask = p_valid & np.isfinite(p_sal) & (p_depth > np.nanmax(p_depth) - 100)
            if np.sum(deep_mask) >= 3:
                deep_means_s[p] = np.nanmean(p_sal[deep_mask])
        if "temperature" in ds:
            p_temp = ds["temperature"].values[p_mask]
            deep_mask = p_valid & np.isfinite(p_temp) & (p_depth > np.nanmax(p_depth) - 100)
            if np.sum(deep_mask) >= 3:
                deep_means_t[p] = np.nanmean(p_temp[deep_mask])
    prof_list = sorted(deep_means_s.keys())
    for i in range(1, len(prof_list)):
        p_prev = prof_list[i - 1]
        p_curr = prof_list[i]
        if p_curr in deep_means_s and p_prev in deep_means_s:
            delta_s = abs(deep_means_s[p_curr] - deep_means_s[p_prev])
            if delta_s > 1.0:   # raised from 0.5 — real horizontal salinity gradients exist
                p_mask = (profile_index == p_curr)
                if "salinity" in qc_dict:
                    qc_dict["salinity"][p_mask] = 3
                flagged += int(np.sum(p_mask))
    prof_list_t = sorted(deep_means_t.keys())
    for i in range(1, len(prof_list_t)):
        p_prev = prof_list_t[i - 1]
        p_curr = prof_list_t[i]
        if p_curr in deep_means_t and p_prev in deep_means_t:
            delta_t = abs(deep_means_t[p_curr] - deep_means_t[p_prev])
            if delta_t > 3.0:   # raised from 1.0 — gliders cross real horizontal gradients
                p_mask = (profile_index == p_curr)
                if "temperature" in qc_dict:
                    qc_dict["temperature"][p_mask] = 3
                flagged += int(np.sum(p_mask))
    return flagged


def test_deepest_pressure(ds, qc_dict, config_pressure_dbar=1000.0):
    if "pressure" not in ds:
        return 0
    p = ds["pressure"].values
    if config_pressure_dbar <= 10:
        tolerance_pct = 150.0
    elif config_pressure_dbar <= 1000:
        tolerance_pct = 150.0 - (config_pressure_dbar - 10) * (140.0 / 990.0)
    else:
        tolerance_pct = 10.0
    if config_pressure_dbar > 1000:
        tolerance = 100.0
    else:
        tolerance = config_pressure_dbar * tolerance_pct / 100.0
    threshold = config_pressure_dbar + tolerance
    bad = p > threshold
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        if "pressure" in qc_dict:
            qc_dict["pressure"][bad] = 3
        if "temperature" in qc_dict:
            qc_dict["temperature"][bad] = 3
        if "salinity" in qc_dict:
            qc_dict["salinity"][bad] = 3
    return n_bad


def pressure_cascade(ds, qc_dict):
    if "pressure" not in qc_dict:
        return 0
    pres_bad = (qc_dict["pressure"] == 4) | (qc_dict["pressure"] == 9)
    n_casc = 0
    for var, qc in qc_dict.items():
        if var == "pressure":
            continue
        cascade_mask = pres_bad & (qc != 4) & (qc != 9)
        n_casc += int(np.sum(cascade_mask))
        qc[cascade_mask] = 4
    return n_casc


def apply_argo_qc(ds, config_pressure_dbar=1000.0):
    print("  Applying ARGO QC Flags (Manual v3.9)...")
    vars_to_qc = [
        "temperature", "salinity", "pressure", "oxygen_concentration",
        "chlorophyll", "cdom", "backscatter_700", "density",
        "latitude", "longitude",
    ]
    qc_dict = {}
    n = len(ds.time)
    for var in vars_to_qc:
        if var in ds:
            qc = np.ones(n, dtype=np.int8)
            qc[np.isnan(ds[var].values)] = 9
            qc_dict[var] = qc

    n2 = test_impossible_date(ds, qc_dict)
    if n2 > 0:
        print(f"   Test 2  (Impossible date):    flagged {n2}")
    n3 = test_impossible_location(ds, qc_dict)
    if n3 > 0:
        print(f"   Test 3  (Impossible location): flagged {n3}")
    n5 = test_impossible_speed(ds, qc_dict)
    if n5 > 0:
        print(f"   Test 5  (Impossible speed):    flagged {n5}")
    n6 = test_global_range(ds, qc_dict)
    print(f"   Test 6  (Global range):          flagged {n6}")
    n8 = test_pressure_increasing(ds, qc_dict)
    if n8 > 0:
        print(f"   Test 8  (Pressure increasing): flagged {n8}")
    n9 = test_spike(ds, qc_dict)
    print(f"   Test 9  (Spike test):            flagged {n9}")
    n13 = test_stuck_value(ds, qc_dict)
    if n13 > 0:
        print(f"   Test 13 (Stuck value):         flagged {n13}")
    n14 = test_density_inversion(ds, qc_dict)
    if n14 > 0:
        print(f"   Test 14 (Density inversion):   flagged {n14}")
    n16 = test_gross_sensor_drift(ds, qc_dict)
    if n16 > 0:
        print(f"   Test 16 (Gross sensor drift):  flagged {n16}")
    n19 = test_deepest_pressure(ds, qc_dict, config_pressure_dbar)
    if n19 > 0:
        print(f"   Test 19 (Deepest pressure):    flagged {n19}")
    nc = pressure_cascade(ds, qc_dict)
    if nc > 0:
        print(f"   Cascade (PRES_QC -> all):      flagged {nc}")

    if "temperature" in qc_dict and "salinity" in qc_dict:
        t_qc = qc_dict["temperature"]
        s_cascade = (t_qc == 4) & (qc_dict["salinity"] != 4) & (qc_dict["salinity"] != 9)
        m_cascade = (t_qc == 9) & (qc_dict["salinity"] != 9)
        n_tc = int(np.sum(s_cascade))
        n_tm = int(np.sum(m_cascade))
        qc_dict["salinity"][s_cascade] = 4
        qc_dict["salinity"][m_cascade] = 9
        if n_tc + n_tm > 0:
            print(f"   Cascade (TEMP_QC -> SAL_QC):   flagged {n_tc + n_tm} (bad:{n_tc}, miss:{n_tm})")

    for var, qc in qc_dict.items():
        ds[f"{var}_QC"] = xr.DataArray(qc, dims=["time"])

    qc_attrs = {
        "long_name": "Quality flag",
        "standard_name": "status_flag",
        "flag_values": np.array([1, 2, 3, 4, 9], dtype=np.int8),
        "flag_meanings": "good probably_good probably_bad bad missing",
        "conventions": "ARGO Reference Table 2",
    }
    for var in list(ds.data_vars):
        if var.endswith("_QC"):
            ds[var].attrs.update(qc_attrs)

    return ds


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_step23(l0_path=None):
    print("=" * 60)
    print("  STEP 2/3: L0 -> L1 (QC + ARGO flags)")
    print("=" * 60)
    t0 = time.time()

    if l0_path is None:
        l0_path = os.path.join(OUTPUT_DIR, f"incois_glider_{GLIDER_ID}_L0.nc")

    if not os.path.exists(l0_path):
        print(f"ERROR: L0 file not found: {l0_path}")
        sys.exit(1)

    print(f"  Loading L0: {l0_path}")
    ds = xr.open_dataset(l0_path)
    print(f"  Observations: {len(ds.time):,}")
    print(f"  Variables: {len(ds.data_vars)}")

    ds = pre_clean(ds)
    ds = apply_optics_correction(ds)
    ds = apply_physics_qc(ds)
    ds = oxygen_lag_correction(ds)
    ds = apply_argo_qc(ds, config_pressure_dbar=MAX_DEPTH_DBAR)

    ds.attrs["processing_level"] = "L1 - GliderTools QC + ARGO RTQC flags applied"
    ds.attrs["history"] = f"L1 processed on {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} using pipeline/step23.py"
    ds.attrs["processing_software"] = "pipeline/step23.py v1.0"

    l1_dir = os.path.join(OUTPUT_DIR, "l1")
    os.makedirs(l1_dir, exist_ok=True)
    l1_path = os.path.join(l1_dir, f"incois_glider_{GLIDER_ID}_L1.nc")
    print(f"\n  Saving L1: {l1_path}")
    ds.to_netcdf(l1_path, mode="w", format="NETCDF4")

    elapsed = time.time() - t0
    print(f"\n  L1 saved: {os.path.getsize(l1_path) / 1024 / 1024:.1f} MB")
    print(f"  QC Flag Summary:")
    for var in sorted(ds.data_vars):
        if var.endswith("_QC"):
            qc = ds[var].values.astype(int)
            good = int(np.sum(qc == 1))
            prob_bad = int(np.sum(qc == 3))
            bad = int(np.sum(qc == 4))
            missing = int(np.sum(qc == 9))
            pct_good = 100.0 * good / len(qc) if len(qc) > 0 else 0
            print(f"    {var:40s} Good:{good:>8,} ({pct_good:5.1f}%)  PBad:{prob_bad:>6,}  Bad:{bad:>6,}  Miss:{missing:>6,}")

    ds.close()
    print(f"\n  STEP 2/3 COMPLETE in {elapsed:.1f}s")
    return l1_path


if __name__ == "__main__":
    l0 = sys.argv[1] if len(sys.argv) > 1 else None
    run_step23(l0)
