#!/usr/bin/env python3
"""
run_pipeline.py — INCOIS Glider L1 Processing Pipeline
=======================================================

Produces a complete set of outputs from raw glider data:

  L0 products (raw / no QC):
    output/L0-timeseries/incois_glider_{ID}_L0.nc
    output/L0-profiles/incois_glider_{ID}_profile_NNNN.nc  (one per dive)
    output/L0-gridfiles/incois_glider_{ID}_L0_grid.nc

  L1 products (ARGO RTQC flags applied):
    output/L1-timeseries/incois_glider_{ID}_L1.nc
    output/L1-profiles/incois_glider_{ID}_L1_profile_NNNN.nc
    output/L1-gridfiles/incois_glider_{ID}_L1_grid.nc

  Plots (19 diagnostic PNG files):
    output/plots/

  Reports:
    output/reports/incois_glider_{ID}_summary.txt

Usage
-----
  bash run_pipeline.sh /path/to/data             # run from data folder
  bash run_pipeline.sh /path/to/data /path/L0.nc # explicit L0, skip decode
  python run_pipeline.py --data-dir /path/to/data
  python run_pipeline.py --data-dir /path --l0-path /path/L0.nc
  python run_pipeline.py --skip-step1            # use existing L0
"""
import os
import sys
import time
import argparse

# ── Parse CLI args BEFORE importing config ──────────────────────
parser = argparse.ArgumentParser(
    description="INCOIS Glider L1 Processing Pipeline",
    add_help=True)
parser.add_argument("--data-dir",   default=None,
                    help="Glider data folder (overrides config.py DATA_DIR)")
parser.add_argument("--output-dir", default=None,
                    help="Output folder (default: <data-dir>/output)")
parser.add_argument("--l0-path",    default=None,
                    help="Explicit L0 NetCDF timeseries path. "
                         "Skips Step 1 (binary decode) automatically.")
parser.add_argument("--skip-step1", action="store_true",
                    help="Skip binary decode — use existing L0 timeseries")
parser.add_argument("--deploy-yml", default=None,
                    help="Path to deployment.yml (if not in data-dir)")
_args, _ = parser.parse_known_args()

if _args.data_dir:
    os.environ["GLIDER_DATA_DIR"] = os.path.abspath(_args.data_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Configure ───────────────────────────────────────────────────
import config as _cfg
if "GLIDER_DATA_DIR" in os.environ:
    _cfg.DATA_DIR    = os.environ["GLIDER_DATA_DIR"]
    _cfg.GLIDER_ID   = os.path.basename(_cfg.DATA_DIR)
    _cfg.OUTPUT_DIR  = os.path.join(_cfg.DATA_DIR, "output")
    _cfg.CACHE_DIR   = os.path.join(_cfg.DATA_DIR, "cache")
    _cfg.BINARY_DIR  = os.path.join(_cfg.DATA_DIR, "combined_binary")
    _cfg.DEPLOY_YAML = os.path.join(_cfg.DATA_DIR, "deployment.yml")

if _args.output_dir:
    _cfg.OUTPUT_DIR = os.path.abspath(_args.output_dir)

if _args.deploy_yml:
    _cfg.DEPLOY_YAML = os.path.abspath(_args.deploy_yml)

from config import (
    OUTPUT_DIR, GLIDER_ID, print_config, ensure_dirs,
    get_l0_path, setup_binary_dir, detect_deployment,
    RUN_STEP1, RUN_STEP23, RUN_STEP4, RUN_STEP5, RUN_VERIFY,
)


# ── Output directory helpers ─────────────────────────────────────
def _dirs():
    """Create and return all output directory paths."""
    d = {
        "L0_ts":       os.path.join(OUTPUT_DIR, "L0-timeseries"),
        "L0_profiles": os.path.join(OUTPUT_DIR, "L0-profiles"),
        "L0_grid":     os.path.join(OUTPUT_DIR, "L0-gridfiles"),
        "L1_ts":       os.path.join(OUTPUT_DIR, "L1-timeseries"),
        "L1_profiles": os.path.join(OUTPUT_DIR, "L1-profiles"),
        "L1_grid":     os.path.join(OUTPUT_DIR, "L1-gridfiles"),
        "plots":       os.path.join(OUTPUT_DIR, "plots"),
        "reports":     os.path.join(OUTPUT_DIR, "reports"),
        "cache":       _cfg.CACHE_DIR,
    }
    for path in d.values():
        os.makedirs(path, exist_ok=True)
    return d


def _l0_nc(dirs):
    return os.path.join(dirs["L0_ts"],
                        f"incois_glider_{GLIDER_ID}_L0.nc")

def _l1_nc(dirs):
    return os.path.join(dirs["L1_ts"],
                        f"incois_glider_{GLIDER_ID}_L1.nc")

def _l0_grid_nc(dirs):
    return os.path.join(dirs["L0_grid"],
                        f"incois_glider_{GLIDER_ID}_L0_grid.nc")

def _l1_grid_nc(dirs):
    return os.path.join(dirs["L1_grid"],
                        f"incois_glider_{GLIDER_ID}_L1_grid.nc")


# ── Main ─────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    # 1. Auto-detect deployment parameters
    detect_deployment(verbose=True)

    # 2. Collect binary files if needed
    setup_binary_dir()

    print_config()
    dirs = _dirs()

    base = f"incois_glider_{GLIDER_ID}"

    # ── Resolve the explicit L0 path ──────────────────────────────
    _explicit_l0 = getattr(_args, "l0_path", None)
    _skip1       = _args.skip_step1 or (_explicit_l0 is not None)

    # Auto-skip if dbdreader not installed
    if not _skip1 and RUN_STEP1:
        try:
            import dbdreader  # noqa: F401
        except ImportError:
            print("  WARNING: dbdreader not installed — skipping Step 1")
            print("  Install: pip install dbdreader")
            _skip1 = True

    # ── Step 1: Binary → L0 timeseries ───────────────────────────
    l0_path = None
    if RUN_STEP1 and not _skip1:
        print()
        from step1 import run_step1
        _cfg.OUTPUT_DIR = dirs["L0_ts"]   # write L0 into L0-timeseries/
        l0_path = run_step1()
        _cfg.OUTPUT_DIR = OUTPUT_DIR       # restore
        print()
    else:
        # Prefer explicit path, then auto-detect
        if _explicit_l0 and os.path.exists(_explicit_l0):
            l0_path = _explicit_l0
            print(f"  Using specified L0: {l0_path}")
        else:
            # Try the output L0-timeseries dir first, then auto-detect
            default = _l0_nc(dirs)
            l0_path = default if os.path.exists(default) else get_l0_path()
            if l0_path and os.path.exists(l0_path):
                print(f"  Using existing L0:  {l0_path}")
            else:
                print(f"  Step 1 skipped. L0 not found.")
                l0_path = None

    if not l0_path or not os.path.exists(l0_path):
        print("  ERROR: No L0 file available. Cannot continue.")
        print("  Either run without --skip-step1 or provide --l0-path.")
        sys.exit(1)

    # ── Step 1b: L0 profile splitting ────────────────────────────
    print("=" * 60)
    print("  STEP 1b: L0 Profile Splitting")
    print("=" * 60)
    from step4 import split_profiles
    n_l0_profiles = split_profiles(
        l0_path,
        dirs["L0_profiles"],
        base,
        apply_qc=False)
    print(f"  Created {n_l0_profiles} L0 profile files")
    print()

    # ── Step 1c: L0 gridding ─────────────────────────────────────
    print("=" * 60)
    print("  STEP 1c: L0 Grid")
    print("=" * 60)
    from step4 import make_grid
    l0_grid_path = make_grid(
        l0_path,
        dirs["L0_grid"],
        base + "_L0_grid.nc",
        apply_qc=False)
    print(f"  L0 grid: {l0_grid_path}")
    print()

    # ── Step 2/3: L0 → L1 (QC + ARGO flags) ─────────────────────
    l1_path = None
    if RUN_STEP23:
        from step23 import run_step23
        # Write L1 into L1-timeseries/
        _cfg.OUTPUT_DIR = dirs["L1_ts"]
        l1_path = run_step23(l0_path)
        _cfg.OUTPUT_DIR = OUTPUT_DIR
        # Move/copy to correct name if needed
        expected = _l1_nc(dirs)
        if l1_path and os.path.exists(l1_path) and l1_path != expected:
            import shutil
            shutil.move(l1_path, expected)
            l1_path = expected
        print()

    if not l1_path or not os.path.exists(l1_path):
        print("  ERROR: L1 file not produced.")
        sys.exit(1)

    # ── Step 3b: L1 profile splitting ────────────────────────────
    print("=" * 60)
    print("  STEP 3b: L1 Profile Splitting")
    print("=" * 60)
    n_l1_profiles = split_profiles(
        l1_path,
        dirs["L1_profiles"],
        base + "_L1",
        apply_qc=True)
    print(f"  Created {n_l1_profiles} L1 profile files")
    print()

    # ── Step 4: L1 gridding ───────────────────────────────────────
    l1_grid_path = None
    if RUN_STEP4:
        from step4 import make_grid
        l1_grid_path = make_grid(
            l1_path,
            dirs["L1_grid"],
            base + "_L1_grid.nc",
            apply_qc=True)
        print()

    # ── Step 5: Time-depth plots (L0 raw + L1 QC) ────────────────
    if RUN_STEP5:
        _cfg.OUTPUT_DIR = OUTPUT_DIR
        from step5 import run_step5
        run_step5(l1_grid_path,
                  l1_path=l1_path,
                  l0_path=l0_path)
        print()

    # ── Step 6: Summary, track, T-S, coverage ────────────────────
    from step6 import run_step6
    run_step6(l0_path=l0_path,
              l1_path=l1_path,
              grid_path=l1_grid_path)
    print()

    # ── Step 7: Oceanographic section plots ──────────────────────
    if l1_grid_path and os.path.exists(l1_grid_path):
        from step7 import run_step7
        run_step7(grid_path=l1_grid_path, l1_path=l1_path)
        print()

    # ── Verify ───────────────────────────────────────────────────
    if RUN_VERIFY and l1_path:
        from verify import main as verify_main
        print("  Running verification...")
        sys.argv = ["verify.py", "--input", l1_path]
        if l0_path and os.path.exists(l0_path):
            sys.argv.extend(["--l0", l0_path])
        verify_main()
        print()

    # ── Done ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  All outputs in: {OUTPUT_DIR}")
    print()
    print("  L0 products:")
    print(f"    timeseries:  {dirs['L0_ts']}")
    print(f"    profiles:    {dirs['L0_profiles']}")
    print(f"    gridfiles:   {dirs['L0_grid']}")
    print()
    print("  L1 products:")
    print(f"    timeseries:  {dirs['L1_ts']}")
    print(f"    profiles:    {dirs['L1_profiles']}")
    print(f"    gridfiles:   {dirs['L1_grid']}")
    print()
    print(f"  Plots:   {dirs['plots']}")
    print(f"  Reports: {dirs['reports']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
