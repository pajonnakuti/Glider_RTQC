"""
config.py — Glider processing pipeline configuration.

HOW TO USE
----------
1. Change DATA_DIR to your glider folder (the only line you need to edit).
2. Run:  python run_pipeline.py

Everything else — GPS bounds, deployment year, hemisphere, factory-test
location, max depth — is auto-detected from the folder contents:

  Priority order for each setting
  --------------------------------
  1. deployment.yml  (explicit metadata wins)
  2. GPS fixes in *.mlg log files  (actual track data)
  3. Binary file headers  (timestamps, filenames)
  4. Safe fallback defaults  (global bounds, no filtering)
"""
import os
import sys
import glob
import re
import numpy as np

# ============================================================
# DATA_DIR — set your glider data folder here
# Windows:  DATA_DIR = r"T:\glider_data\890_2"
# Linux:    DATA_DIR = "/data/glider/890_2"
#
# BETTER: pass it at runtime and never edit this file:
#   bash run_pipeline.sh /path/to/your/data
#   python run_pipeline.py --data-dir /path/to/your/data
#
# The run_pipeline.sh auto-detects L0-timeseries/ folders
# and skips binary decoding automatically.
# ============================================================
DATA_DIR = os.environ.get(
    "GLIDER_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "Raw_Data", "1130-Mar-2025")
)

# ============================================================
# Auto-derived paths  (do not edit)
# ============================================================
# Allow runtime override via environment variable
if os.environ.get("GLIDER_DATA_DIR"):
    DATA_DIR = os.environ["GLIDER_DATA_DIR"]

DATA_DIR    = os.path.abspath(DATA_DIR)
GLIDER_ID   = os.path.basename(DATA_DIR)
OUTPUT_DIR  = os.path.join(DATA_DIR, "output")
CACHE_DIR   = os.path.join(DATA_DIR, "cache")
BINARY_DIR  = os.path.join(DATA_DIR, "combined_binary")
DEPLOY_YAML = os.path.join(DATA_DIR, "deployment.yml")

# Binary extensions we collect
_BINARY_EXTS = (".dcd", ".ecd", ".dbd", ".ebd")

# ============================================================
# Processing parameters  (rarely need changing)
# ============================================================
DEPTH_BIN         = 1.0
PLOT_DEPTH_MAX    = 1000.0
OXYGEN_TAU        = 30.0
PROFILE_FILT_SECS = 100
PROFILE_MIN_SECS  = 300

# Physical limits for pre-filtering
TEMP_MIN = -2.0
TEMP_MAX = 40.0
COND_MIN = 0.005
PRES_MIN = -2.0

# Pipeline step control
RUN_STEP1  = True
RUN_STEP23 = True
RUN_STEP4  = True
RUN_STEP5  = True
RUN_VERIFY = True

# ============================================================
# These are filled by detect_deployment() at startup
# ============================================================
GPS_LAT_MIN = -90.0
GPS_LAT_MAX =  90.0
GPS_LON_MIN = -180.0
GPS_LON_MAX =  180.0

MAX_DEPTH_DBAR = 1000.0

CLEAN_FACTORY_TESTS = False
FACTORY_LAT_MIN =  40.0
FACTORY_LAT_MAX =  43.0
FACTORY_LON_MIN = -72.0
FACTORY_LON_MAX = -69.0

CLEAN_ZERO_GPS   = True
CLEAN_HEMISPHERE = True
CLEAN_MODE_YEAR  = True


# ============================================================
# Auto-detection helpers
# ============================================================

def _ddmm_to_dd(val, hemi):
    """Convert DDMM.MMM (Slocum format) to decimal degrees."""
    deg  = int(val / 100)
    mins = val - deg * 100
    dd   = deg + mins / 60.0
    if hemi in ("S", "W"):
        dd = -dd
    return dd


def _gps_from_mlg_files(data_dir):
    """
    Walk data_dir for *.mlg files and extract GPS Location fixes.
    Returns (lats, lons) as numpy arrays, or (None, None) if none found.
    """
    mlg_files = []
    for root, _dirs, files in os.walk(data_dir):
        rel = os.path.relpath(root, data_dir).lower()
        if rel.startswith("output") or rel.startswith("cache"):
            continue
        for fn in files:
            if fn.lower().endswith(".mlg"):
                mlg_files.append(os.path.join(root, fn))

    if not mlg_files:
        return None, None

    lats, lons = [], []
    pat = re.compile(
        r"GPS Location:\s+([\d.]+)\s+([NS])\s+([\d.]+)\s+([EW])"
    )
    for fpath in mlg_files:
        try:
            with open(fpath, "r", errors="replace") as fh:
                for line in fh:
                    m = pat.search(line)
                    if m:
                        lat = _ddmm_to_dd(float(m.group(1)), m.group(2))
                        lon = _ddmm_to_dd(float(m.group(3)), m.group(4))
                        # Sanity check — skip obviously bad fixes
                        if abs(lat) > 0.5 and abs(lon) > 0.5:
                            lats.append(lat)
                            lons.append(lon)
        except Exception:
            pass

    if not lats:
        return None, None
    return np.array(lats), np.array(lons)


def _deployment_dates_from_yaml(yaml_path):
    """Return (start_year, end_year) from deployment.yml, or (None, None)."""
    if not os.path.exists(yaml_path):
        return None, None
    try:
        import yaml
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        meta = cfg.get("metadata", {})
        start = meta.get("deployment_start", "")
        end   = meta.get("deployment_end",   "")
        sy = int(start[:4]) if start and len(start) >= 4 else None
        ey = int(end[:4])   if end   and len(end)   >= 4 else None
        return sy, ey
    except Exception:
        return None, None


def _max_depth_from_yaml(yaml_path):
    """Try to read max depth from deployment.yml pressure valid_max."""
    if not os.path.exists(yaml_path):
        return None
    try:
        import yaml
        with open(yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        ncvars = cfg.get("netcdf_variables", {})
        pres = ncvars.get("pressure", {})
        vmax = pres.get("valid_max", None)
        if vmax is not None:
            return float(vmax)
    except Exception:
        pass
    return None


def _deployment_year_from_binary_headers(data_dir):
    """
    Read fileopen_time from the first few binary file headers.
    Returns the most common year, or None.
    """
    years = []
    pat = re.compile(r"fileopen_time:\s+\w+\s+\w+\s+\d+\s+\d+:\d+:\d+\s+(\d{4})")
    count = 0
    for root, _dirs, files in os.walk(data_dir):
        rel = os.path.relpath(root, data_dir).lower()
        if rel.startswith("output") or rel.startswith("cache"):
            continue
        for fn in sorted(files):
            if fn.lower().endswith((".dbd", ".dcd", ".ebd", ".ecd")):
                try:
                    with open(os.path.join(root, fn), "rb") as fh:
                        hdr = fh.read(300).decode("ascii", errors="replace")
                    m = pat.search(hdr)
                    if m:
                        years.append(int(m.group(1)))
                        count += 1
                        if count >= 20:
                            break
                except Exception:
                    pass
        if count >= 20:
            break

    if not years:
        return None
    vals, cnts = np.unique(years, return_counts=True)
    return int(vals[np.argmax(cnts)])


def detect_deployment(verbose=True):
    """
    Auto-detect deployment parameters from the folder contents.

    Detection order for GPS bounds:
      1. GPS fixes from *.mlg files  →  median ± padding
      2. deployment.yml dates        →  used for year filter only
      3. Binary file headers         →  fallback year
      4. Global defaults             →  no filtering

    Updates the module-level config variables in place.
    Returns a dict of what was detected.
    """
    global GPS_LAT_MIN, GPS_LAT_MAX, GPS_LON_MIN, GPS_LON_MAX
    global MAX_DEPTH_DBAR
    global CLEAN_FACTORY_TESTS, FACTORY_LAT_MIN, FACTORY_LAT_MAX
    global FACTORY_LON_MIN, FACTORY_LON_MAX
    global CLEAN_HEMISPHERE, CLEAN_MODE_YEAR

    detected = {}

    if verbose:
        print("  Auto-detecting deployment parameters...")

    # ------------------------------------------------------------------
    # 1. GPS bounds from MLG files
    # ------------------------------------------------------------------
    lats, lons = _gps_from_mlg_files(DATA_DIR)

    if lats is not None and len(lats) >= 5:
        # Use IQR-clipped range + generous padding (2°) to avoid cutting
        # edge profiles while still filtering pre-deployment test fixes
        lat_p5,  lat_p95  = np.percentile(lats, 5),  np.percentile(lats, 95)
        lon_p5,  lon_p95  = np.percentile(lons, 5),  np.percentile(lons, 95)
        lat_pad = max(2.0, (lat_p95 - lat_p5) * 0.15)
        lon_pad = max(2.0, (lon_p95 - lon_p5) * 0.15)

        GPS_LAT_MIN = round(lat_p5  - lat_pad, 2)
        GPS_LAT_MAX = round(lat_p95 + lat_pad, 2)
        GPS_LON_MIN = round(lon_p5  - lon_pad, 2)
        GPS_LON_MAX = round(lon_p95 + lon_pad, 2)

        # Clamp to valid ranges
        GPS_LAT_MIN = max(GPS_LAT_MIN, -90.0)
        GPS_LAT_MAX = min(GPS_LAT_MAX,  90.0)
        GPS_LON_MIN = max(GPS_LON_MIN, -180.0)
        GPS_LON_MAX = min(GPS_LON_MAX,  180.0)

        median_lat = float(np.median(lats))
        median_lon = float(np.median(lons))
        detected["gps_fixes"]   = len(lats)
        detected["gps_lat_min"] = GPS_LAT_MIN
        detected["gps_lat_max"] = GPS_LAT_MAX
        detected["gps_lon_min"] = GPS_LON_MIN
        detected["gps_lon_max"] = GPS_LON_MAX
        detected["median_lat"]  = round(median_lat, 4)
        detected["median_lon"]  = round(median_lon, 4)

        # Hemisphere cleaning: only meaningful if deployment is clearly
        # in one hemisphere (median lat > 5° or < -5°)
        CLEAN_HEMISPHERE = abs(median_lat) > 5.0
        detected["clean_hemisphere"] = CLEAN_HEMISPHERE

        if verbose:
            print(f"    GPS: {len(lats)} fixes  "
                  f"lat [{GPS_LAT_MIN}, {GPS_LAT_MAX}]  "
                  f"lon [{GPS_LON_MIN}, {GPS_LON_MAX}]")
    else:
        detected["gps_fixes"] = 0
        if verbose:
            print("    GPS: no fixes found — using global bounds (no filtering)")

    # ------------------------------------------------------------------
    # 2. Factory test location
    #    If the deployment is clearly NOT near the default NJ coast box,
    #    disable factory-test cleaning (avoids false positives).
    #    If it IS near NJ, keep it enabled.
    # ------------------------------------------------------------------
    if lats is not None and len(lats) >= 5:
        # Check if any GPS fixes fall inside the default factory box
        in_factory = (
            (lats >= FACTORY_LAT_MIN) & (lats <= FACTORY_LAT_MAX) &
            (lons >= FACTORY_LON_MIN) & (lons <= FACTORY_LON_MAX)
        )
        frac_in_factory = float(np.mean(in_factory))

        if frac_in_factory > 0.05:
            # >5% of fixes in the factory box → keep cleaning enabled
            CLEAN_FACTORY_TESTS = True
            detected["factory_test_cleaning"] = "enabled (fixes found in NJ box)"
        else:
            # Deployment is elsewhere — check if there's a distinct cluster
            # far from the main deployment area (pre-deployment test)
            median_lat = float(np.median(lats))
            median_lon = float(np.median(lons))
            far = np.sqrt((lats - median_lat)**2 + (lons - median_lon)**2)
            outlier_fixes = lats[far > 10]   # fixes > 10° from median

            if len(outlier_fixes) > 3:
                # There are outlier clusters — find their centroid and use
                # that as the factory test box
                out_lats = lats[far > 10]
                out_lons = lons[far > 10]
                FACTORY_LAT_MIN = float(np.min(out_lats)) - 1.0
                FACTORY_LAT_MAX = float(np.max(out_lats)) + 1.0
                FACTORY_LON_MIN = float(np.min(out_lons)) - 1.0
                FACTORY_LON_MAX = float(np.max(out_lons)) + 1.0
                CLEAN_FACTORY_TESTS = True
                detected["factory_test_cleaning"] = (
                    f"enabled (outlier cluster at "
                    f"lat [{FACTORY_LAT_MIN:.1f},{FACTORY_LAT_MAX:.1f}] "
                    f"lon [{FACTORY_LON_MIN:.1f},{FACTORY_LON_MAX:.1f}])"
                )
                if verbose:
                    print(f"    Factory tests: outlier cluster detected, "
                          f"will clean lat [{FACTORY_LAT_MIN:.1f},{FACTORY_LAT_MAX:.1f}] "
                          f"lon [{FACTORY_LON_MIN:.1f},{FACTORY_LON_MAX:.1f}]")
            else:
                CLEAN_FACTORY_TESTS = False
                detected["factory_test_cleaning"] = "disabled (no outlier cluster)"
                if verbose:
                    print("    Factory tests: no outlier cluster — cleaning disabled")
    else:
        CLEAN_FACTORY_TESTS = False
        detected["factory_test_cleaning"] = "disabled (no GPS data)"

    # ------------------------------------------------------------------
    # 3. Max depth from deployment.yml
    # ------------------------------------------------------------------
    depth_from_yaml = _max_depth_from_yaml(DEPLOY_YAML)
    if depth_from_yaml is not None and depth_from_yaml > 10:
        MAX_DEPTH_DBAR = depth_from_yaml
        detected["max_depth_source"] = "deployment.yml"
    else:
        MAX_DEPTH_DBAR = 1000.0
        detected["max_depth_source"] = "default"
    detected["max_depth_dbar"] = MAX_DEPTH_DBAR

    if verbose:
        print(f"    Max depth: {MAX_DEPTH_DBAR} dbar  "
              f"(source: {detected['max_depth_source']})")

    # ------------------------------------------------------------------
    # 4. Deployment year (for CLEAN_MODE_YEAR)
    # ------------------------------------------------------------------
    sy, ey = _deployment_dates_from_yaml(DEPLOY_YAML)
    if sy is not None:
        detected["deployment_year"] = sy
        detected["deployment_year_source"] = "deployment.yml"
    else:
        yr = _deployment_year_from_binary_headers(DATA_DIR)
        if yr is not None:
            detected["deployment_year"] = yr
            detected["deployment_year_source"] = "binary headers"
        else:
            detected["deployment_year"] = None
            detected["deployment_year_source"] = "unknown"

    if verbose:
        yr_info = detected.get("deployment_year", "unknown")
        print(f"    Deployment year: {yr_info}  "
              f"(source: {detected['deployment_year_source']})")

    return detected


# ============================================================
# Binary file discovery
# ============================================================

def _find_binary_files(root):
    """
    Walk root recursively and return all binary glider files.
    Skips output/ and cache/ trees.
    """
    found = []
    for dirpath, _dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root).lower()
        if rel.startswith("output") or rel.startswith("cache"):
            continue
        for fn in files:
            if fn.lower().endswith(_BINARY_EXTS):
                found.append(os.path.join(dirpath, fn))
    return found


def setup_binary_dir():
    """
    Ensure BINARY_DIR exists and contains all binary files from DATA_DIR.

    1. If combined_binary/ already exists and is non-empty → use it as-is.
    2. Otherwise walk DATA_DIR for all binary files and hard-link/copy them.

    Returns the path to the binary directory.
    """
    import shutil

    combined = BINARY_DIR

    if os.path.isdir(combined):
        existing = [f for f in os.listdir(combined)
                    if f.lower().endswith(_BINARY_EXTS)]
        if existing:
            print(f"  combined_binary/ already exists "
                  f"({len(existing)} files) — skipping collection")
            return combined

    print(f"\n  Collecting binary files from {DATA_DIR} ...")
    os.makedirs(combined, exist_ok=True)

    all_files = _find_binary_files(DATA_DIR)
    all_files = [f for f in all_files
                 if not os.path.abspath(f).startswith(
                     os.path.abspath(combined))]

    if not all_files:
        print("  WARNING: no binary files found anywhere under DATA_DIR")
        return combined

    copied = skipped = 0
    for src in all_files:
        dst = os.path.join(combined, os.path.basename(src))
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            try:
                os.link(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"  WARNING: could not copy {src}: {e}")

    n_total = sum(1 for f in os.listdir(combined)
                  if f.lower().endswith(_BINARY_EXTS))
    print(f"  Copied {copied} new files, skipped {skipped} existing "
          f"-> {n_total} total in combined_binary/")
    return combined


# ============================================================
# Helpers used by run_pipeline.py
# ============================================================

def ensure_dirs():
    """Create all required output directories."""
    dirs = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, "l1"),
        os.path.join(OUTPUT_DIR, "profiles"),
        os.path.join(OUTPUT_DIR, "gridfiles"),
        os.path.join(OUTPUT_DIR, "plots"),
        CACHE_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def get_l0_path():
    """
    Find the L0 NetCDF timeseries for this deployment.

    Search order:
      1. output/incois_glider_{ID}_L0.nc   (step1 output — always preferred)
      2. Any *.nc in DATA_DIR tree with 'L0' in filename (largest wins)
      3. Any *.nc whose processing_level attribute contains 'L0' (largest wins)
    Skips output/ and cache/ subdirectories.
    """
    import xarray as xr

    # 1. Standard step1 output
    p1 = os.path.join(OUTPUT_DIR, f"incois_glider_{GLIDER_ID}_L0.nc")
    if os.path.exists(p1) and os.path.getsize(p1) > 100_000:
        return p1

    # Collect all NC files outside output/ and cache/
    all_nc = []
    for root, _dirs, files in os.walk(DATA_DIR):
        rel = os.path.relpath(root, DATA_DIR).lower()
        if rel.startswith("output") or rel.startswith("cache"):
            continue
        for fn in files:
            if fn.endswith(".nc"):
                fp = os.path.join(root, fn)
                sz = os.path.getsize(fp) if os.path.exists(fp) else 0
                all_nc.append((fn, sz, fp))

    # Priority 2: explicit 'L0' in filename (not 'L1')
    name_l0 = [(sz, fp) for fn, sz, fp in all_nc
               if ("L0" in fn or "l0" in fn)
               and "L1" not in fn and "l1" not in fn
               and "grid" not in fn.lower()
               and "profile" not in fn.lower()]
    if name_l0:
        name_l0.sort(reverse=True)
        return name_l0[0][1]

    # Priority 3: check processing_level attribute for 'L0'
    attr_l0 = []
    for fn, sz, fp in sorted(all_nc, key=lambda x: x[1], reverse=True):
        if sz < 100_000:
            continue
        if "grid" in fn.lower() or "profile" in fn.lower():
            continue
        try:
            with xr.open_dataset(fp) as ds:
                lvl = ds.attrs.get("processing_level", "")
                if "L0" in lvl and "L1" not in lvl:
                    attr_l0.append((sz, fp))
        except Exception:
            pass
        if len(attr_l0) >= 3:   # stop after finding a few candidates
            break
    if attr_l0:
        attr_l0.sort(reverse=True)
        return attr_l0[0][1]

    return p1  # fallback


def get_l1_path():
    return os.path.join(OUTPUT_DIR, "l1", f"incois_glider_{GLIDER_ID}_L1.nc")


def print_config():
    print("=" * 60)
    print("  PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  Glider ID:      {GLIDER_ID}")
    print(f"  Data dir:       {DATA_DIR}")
    print(f"  Binary dir:     {BINARY_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Cache dir:      {CACHE_DIR}")
    print(f"  GPS bounds:     lat [{GPS_LAT_MIN}, {GPS_LAT_MAX}]  "
          f"lon [{GPS_LON_MIN}, {GPS_LON_MAX}]")
    print(f"  Max depth:      {MAX_DEPTH_DBAR} dbar")
    print(f"  Depth bin:      {DEPTH_BIN} m")
    print(f"  Factory clean:  {CLEAN_FACTORY_TESTS}")
    if CLEAN_FACTORY_TESTS:
        print(f"    box: lat [{FACTORY_LAT_MIN},{FACTORY_LAT_MAX}] "
              f"lon [{FACTORY_LON_MIN},{FACTORY_LON_MAX}]")
    print(f"  Hemisphere:     {CLEAN_HEMISPHERE}")
    print(f"  Mode year:      {CLEAN_MODE_YEAR}")
    print("=" * 60)
