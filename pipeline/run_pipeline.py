#!/usr/bin/env python3
"""
run_pipeline.py - Main orchestrator for the full glider processing pipeline.

Runs all steps sequentially: Binary -> L0 -> L1 -> Profiles/Grid -> Plots -> Verify

Usage:
    python run_pipeline.py
    python run_pipeline.py --data-dir T:\\some_other_glider
"""
import os
import sys
import time
import argparse

# Allow overriding DATA_DIR and skipping steps from the command line
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data-dir",    default=None, help="Override DATA_DIR in config.py")
parser.add_argument("--output-dir",  default=None, help="Override output directory")
parser.add_argument("--l0-path",     default=None, help="Explicit L0 NetCDF path (skips auto-detection)")
parser.add_argument("--skip-step1",  action="store_true",
                    help="Skip Step 1 (binary decode)")
_args, _ = parser.parse_known_args()

if _args.data_dir:
    os.environ["GLIDER_DATA_DIR"] = os.path.abspath(_args.data_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg_module
if "GLIDER_DATA_DIR" in os.environ:
    _cfg_module.DATA_DIR   = os.environ["GLIDER_DATA_DIR"]
    _cfg_module.GLIDER_ID  = os.path.basename(_cfg_module.DATA_DIR)
    _cfg_module.OUTPUT_DIR = os.path.join(_cfg_module.DATA_DIR, "output")
    _cfg_module.CACHE_DIR  = os.path.join(_cfg_module.DATA_DIR, "cache")
    _cfg_module.BINARY_DIR = os.path.join(_cfg_module.DATA_DIR, "combined_binary")
    _cfg_module.DEPLOY_YAML = os.path.join(_cfg_module.DATA_DIR, "deployment.yml")

# --output-dir overrides the auto-derived output directory
if _args.output_dir:
    _cfg_module.OUTPUT_DIR = os.path.abspath(_args.output_dir)

from config import (
    OUTPUT_DIR, GLIDER_ID, print_config, ensure_dirs,
    get_l0_path, get_l1_path, setup_binary_dir,
    RUN_STEP1, RUN_STEP23, RUN_STEP4, RUN_STEP5, RUN_VERIFY,
    detect_deployment,
)

# Step 6 always runs if L1 exists
RUN_STEP6 = True


def main():
    t_start = time.time()

    # 1. Auto-detect GPS bounds, depth, factory-test location, etc.
    detect_deployment(verbose=True)

    # 2. Collect binary files into combined_binary/ if needed
    setup_binary_dir()

    print_config()
    ensure_dirs()

    # --l0-path flag overrides all L0 auto-detection
    _explicit_l0 = getattr(_args, "l0_path", None)

    l0_path   = None
    l1_path   = None
    grid_path = None

    # ------------------------------------------------------------------
    # Step 1: Binary -> L0
    # ------------------------------------------------------------------
    # Auto-skip if dbdreader is not installed, --skip-step1 was passed,
    # or an explicit L0 path was provided
    _skip_step1 = _args.skip_step1 or (_explicit_l0 is not None)
    if not _skip_step1 and RUN_STEP1:
        try:
            import dbdreader  # noqa: F401
        except ImportError:
            print("  WARNING: dbdreader not installed — skipping Step 1.")
            print("  To enable binary decoding, install Microsoft C++ Build Tools")
            print("  then run:  pip install dbdreader")
            print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print()
            _skip_step1 = True

    if RUN_STEP1 and not _skip_step1:
        from step1 import run_step1
        l0_path = run_step1()
        print()
    else:
        # Use explicit path if given, otherwise auto-detect
        if _explicit_l0 and os.path.exists(_explicit_l0):
            l0_path = _explicit_l0
            print(f"  Using specified L0: {l0_path}")
        else:
            l0_path = get_l0_path()
            if os.path.exists(l0_path):
                print(f"  Step 1 skipped. Using existing L0: {l0_path}")
            else:
                print(f"  Step 1 skipped. L0 not found at {l0_path}")
                l0_path = None

    # ------------------------------------------------------------------
    # Step 2/3: L0 -> L1 (QC)
    # ------------------------------------------------------------------
    if RUN_STEP23:
        from step23 import run_step23
        l1_path = run_step23(l0_path)
        print()
    else:
        l1_path = get_l1_path()
        if not os.path.exists(l1_path):
            print(f"  Step 2/3 skipped. L1 not found at {l1_path}")
            l1_path = None

    # ------------------------------------------------------------------
    # Step 4: Profiles + Grid
    # ------------------------------------------------------------------
    if RUN_STEP4 and l1_path:
        from step4 import run_step4
        grid_path = run_step4(l1_path)
        print()
    elif RUN_STEP4:
        print("  Step 4 skipped (no L1 file)")

    # ------------------------------------------------------------------
    # Step 5: Plots
    # ------------------------------------------------------------------
    if RUN_STEP5:
        from step5 import run_step5
        run_step5(grid_path, l1_path=l1_path, l0_path=l0_path)
        print()

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    if RUN_VERIFY and l1_path:
        from verify import main as verify_main
        print("  Running verification...")
        sys.argv = ["verify.py", "--input", l1_path]
        if l0_path and os.path.exists(l0_path):
            sys.argv.extend(["--l0", l0_path])
        verify_main()
        print()

    # ------------------------------------------------------------------
    # Step 6: Summary report, track map, T-S diagram, coverage matrix
    # ------------------------------------------------------------------
    if RUN_STEP6 and l1_path:
        from step6 import run_step6
        run_step6(l0_path=l0_path, l1_path=l1_path, grid_path=grid_path)
        print()

    # ------------------------------------------------------------------
    # Step 7: Oceanographic section plots (contours, envelopes, etc.)
    # ------------------------------------------------------------------
    if grid_path and os.path.exists(grid_path):
        from step7 import run_step7
        run_step7(grid_path=grid_path, l1_path=l1_path)
        print()

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
