"""
config.py - Central configuration for the glider processing pipeline.

All deployment-specific settings live here. Edit these before running the pipeline.
"""
import os
import sys

# ============================================================
# DEPLOYMENT SETTINGS - EDIT THESE FOR YOUR GLIDER
# ============================================================

# Glider identifier (used in output filenames)
GLIDER_ID = os.environ.get("GLIDER_ID", "890_2")

# Paths - use raw strings for Windows paths
# Change these to match your data location
BASE_DIR = os.environ.get("GLIDER_BASE_DIR", r"T:\890_2")

# Input: directory containing raw .dbd/.ecd binary files
BINARY_DIR = os.path.join(BASE_DIR, "combined_binary")

# Input: deployment YAML config (optional, provides metadata)
DEPLOY_YAML = os.path.join(BASE_DIR, "deployment.yml")

# Output: central directory for all processed data
OUTPUT_DIR = os.environ.get("GLIDER_OUTPUT_DIR", r"T:\glider_data\output")

# Cache directory for dbdreader
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# ============================================================
# DEPLOYMENT AREA BOUNDS (for GPS filtering in Step 1)
# Set to None to skip geographic filtering
# ============================================================
GPS_LAT_MIN = float(os.environ.get("GPS_LAT_MIN", "10.0"))
GPS_LAT_MAX = float(os.environ.get("GPS_LAT_MAX", "18.0"))
GPS_LON_MIN = float(os.environ.get("GPS_LON_MIN", "68.0"))
GPS_LON_MAX = float(os.environ.get("GPS_LON_MAX", "85.0"))

# ============================================================
# PHYSICAL LIMITS (for pre-filtering in Step 1)
# ============================================================
TEMP_MIN = -2.0
TEMP_MAX = 40.0
COND_MIN = 0.005  # S/m
PRES_MIN = -2.0   # dbar

# ============================================================
# PROCESSING PARAMETERS
# ============================================================

# Max profile depth for ARGO Test 19 (dbar)
MAX_DEPTH_DBAR = float(os.environ.get("MAX_DEPTH_DBAR", "1000.0"))

# Depth bin size for grid generation (meters)
DEPTH_BIN = float(os.environ.get("DEPTH_BIN", "1.0"))

# Max depth to plot (meters) - None = auto
PLOT_DEPTH_MAX = float(os.environ.get("PLOT_DEPTH_MAX", "200.0"))

# Oxygen lag correction time constant (seconds)
OXYGEN_TAU = 30.0

# Profile detection parameters
PROFILE_FILT_SECS = 100   # smoothing window
PROFILE_MIN_SECS = 300    # minimum profile duration

# ============================================================
# PIPELINE STEP CONTROL
# Set to False to skip individual steps
# ============================================================
RUN_STEP1 = os.environ.get("RUN_STEP1", "1") == "1"   # Binary -> L0
RUN_STEP23 = os.environ.get("RUN_STEP23", "1") == "1"  # L0 -> L1 (QC)
RUN_STEP4 = os.environ.get("RUN_STEP4", "1") == "1"    # Profiles + Grid
RUN_STEP5 = os.environ.get("RUN_STEP5", "1") == "1"    # Plotting
RUN_VERIFY = os.environ.get("RUN_VERIFY", "1") == "1"   # Verification

# ============================================================
# PRE-CLEANING (remove factory/lab tests before QC)
# ============================================================
# Remove known factory test locations (e.g., New Jersey coast)
CLEAN_FACTORY_TESTS = True
FACTORY_LAT_MIN = 40.0
FACTORY_LAT_MAX = 43.0
FACTORY_LON_MIN = -72.0
FACTORY_LON_MAX = -69.0

# Remove (0, 0) sentinel positions
CLEAN_ZERO_GPS = True

# Filter by dominant hemisphere (separate pre-deployment tests)
CLEAN_HEMISPHERE = True

# Filter by mode year (keep only deployment year and year-1)
CLEAN_MODE_YEAR = True

# ============================================================
# Helper functions
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
    """Get expected L0 output path from Step 1."""
    return os.path.join(OUTPUT_DIR, f"incois_glider_{GLIDER_ID}_L0.nc")


def get_l1_path():
    """Get expected L1 output path from Step 2/3."""
    return os.path.join(OUTPUT_DIR, "l1", f"incois_glider_{GLIDER_ID}_L1.nc")


def print_config():
    """Print current configuration for verification."""
    print("=" * 60)
    print("  PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"  Glider ID:      {GLIDER_ID}")
    print(f"  Base dir:       {BASE_DIR}")
    print(f"  Binary dir:     {BINARY_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Cache dir:      {CACHE_DIR}")
    print(f"  GPS bounds:     [{GPS_LAT_MIN},{GPS_LAT_MAX}] [{GPS_LON_MIN},{GPS_LON_MAX}]")
    print(f"  Max depth:      {MAX_DEPTH_DBAR} dbar")
    print(f"  Depth bin:      {DEPTH_BIN} m")
    print(f"  Steps:          1={RUN_STEP1} 2/3={RUN_STEP23} 4={RUN_STEP4} 5={RUN_STEP5} Verify={RUN_VERIFY}")
    print("=" * 60)
