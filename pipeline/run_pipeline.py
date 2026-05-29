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

# Allow overriding DATA_DIR from the command line before importing config
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data-dir", default=None)
_args, _ = parser.parse_known_args()

if _args.data_dir:
    # Inject into environment so config.py picks it up
    os.environ["GLIDER_DATA_DIR"] = os.path.abspath(_args.data_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Support optional env-var override of DATA_DIR before config loads
import importlib, types
import config as _cfg_module
if "GLIDER_DATA_DIR" in os.environ:
    _cfg_module.DATA_DIR   = os.environ["GLIDER_DATA_DIR"]
    _cfg_module.GLIDER_ID  = os.path.basename(_cfg_module.DATA_DIR)
    _cfg_module.OUTPUT_DIR = os.path.join(_cfg_module.DATA_DIR, "output")
    _cfg_module.CACHE_DIR  = os.path.join(_cfg_module.DATA_DIR, "cache")
    _cfg_module.BINARY_DIR = os.path.join(_cfg_module.DATA_DIR, "combined_binary")
    _cfg_module.DEPLOY_YAML = os.path.join(_cfg_module.DATA_DIR, "deployment.yml")

from config import (
    OUTPUT_DIR, GLIDER_ID, print_config, ensure_dirs,
    get_l0_path, get_l1_path, setup_binary_dir,
    RUN_STEP1, RUN_STEP23, RUN_STEP4, RUN_STEP5, RUN_VERIFY,
    detect_deployment,
)


def main():
    t_start = time.time()

    # 1. Auto-detect GPS bounds, depth, factory-test location, etc.
    detect_deployment(verbose=True)

    # 2. Collect binary files into combined_binary/ if needed
    setup_binary_dir()

    print_config()
    ensure_dirs()

    l0_path   = None
    l1_path   = None
    grid_path = None

    # ------------------------------------------------------------------
    # Step 1: Binary -> L0
    # ------------------------------------------------------------------
    if RUN_STEP1:
        from step1 import run_step1
        l0_path = run_step1()
        print()
    else:
        l0_path = get_l0_path()
        if not os.path.exists(l0_path):
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

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Outputs in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
