#!/usr/bin/env python3
"""
run_pipeline.py - Main orchestrator for the full glider processing pipeline.

Runs all steps sequentially: Binary->L0->L1->Profiles/Grid->Plots->Verify
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RUN_STEP1, RUN_STEP23, RUN_STEP4, RUN_STEP5, RUN_VERIFY,
    OUTPUT_DIR, GLIDER_ID, print_config, ensure_dirs,
    get_l0_path, get_l1_path,
)


def main():
    t_start = time.time()
    print_config()
    ensure_dirs()

    l0_path = None
    l1_path = None
    grid_path = None

    if RUN_STEP1:
        from step1 import run_step1
        l0_path = run_step1()
        print()
    else:
        l0_path = get_l0_path()
        if not os.path.exists(l0_path):
            print(f"  Step 1 skipped. L0 not found at {l0_path}")
            l0_path = None

    if RUN_STEP23:
        from step23 import run_step23
        l1_path = run_step23(l0_path)
        print()
    else:
        l1_path = get_l1_path()
        if not os.path.exists(l1_path):
            print(f"  Step 2/3 skipped. L1 not found at {l1_path}")
            l1_path = None

    if RUN_STEP4 and l1_path:
        from step4 import run_step4
        grid_path = run_step4(l1_path)
        print()
    elif RUN_STEP4:
        print("  Step 4 skipped (no L1 file)")

    if RUN_STEP5:
        from step5 import run_step5
        run_step5(grid_path)
        print()

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
