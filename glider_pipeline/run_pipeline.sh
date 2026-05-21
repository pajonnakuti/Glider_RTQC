#!/bin/bash
# ============================================================
# Glider Data Processing Pipeline - Linux/Mac Entry Point
# ============================================================
# Run the full pipeline: Binary -> L0 -> L1 -> Profiles/Grid -> Plots
#
# Usage:
#   ./run_pipeline.sh                    - Run all steps with default config
#   ./run_pipeline.sh --step 1           - Run only Step 1 (Binary -> L0)
#   ./run_pipeline.sh --step 23          - Run only Step 2/3 (L0 -> L1)
#   ./run_pipeline.sh --step 4           - Run only Step 4 (Profiles + Grid)
#   ./run_pipeline.sh --step 5           - Run only Step 5 (Plots)
#
# Environment variables (optional):
#   GLIDER_ID         - Glider identifier (default: 890_2)
#   GLIDER_BASE_DIR   - Base directory with binary data
#   GLIDER_OUTPUT_DIR - Output directory for processed data
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${SCRIPT_DIR}/pipeline"

echo ""
echo "============================================================"
echo "  GLIDER DATA PROCESSING PIPELINE"
echo "============================================================"
echo "  Pipeline: ${PIPELINE_DIR}"
echo ""

# Check for step override
STEP_OVERRIDE=""
if [ "$1" = "--step" ]; then
    STEP_OVERRIDE="$2"
fi

if [ -n "$STEP_OVERRIDE" ]; then
    echo "  Running Step ${STEP_OVERRIDE} only"
    echo ""
    case "$STEP_OVERRIDE" in
        1)
            export RUN_STEP1=1 RUN_STEP23=0 RUN_STEP4=0 RUN_STEP5=0 RUN_VERIFY=0
            ;;
        23)
            export RUN_STEP1=0 RUN_STEP23=1 RUN_STEP4=0 RUN_STEP5=0 RUN_VERIFY=0
            ;;
        4)
            export RUN_STEP1=0 RUN_STEP23=0 RUN_STEP4=1 RUN_STEP5=0 RUN_VERIFY=0
            ;;
        5)
            export RUN_STEP1=0 RUN_STEP23=0 RUN_STEP4=0 RUN_STEP5=1 RUN_VERIFY=0
            ;;
        *)
            echo "ERROR: Unknown step '$STEP_OVERRIDE'. Use 1, 23, 4, or 5."
            exit 1
            ;;
    esac
else
    echo "  Running full pipeline (all steps)"
    echo ""
    export RUN_STEP1=1 RUN_STEP23=1 RUN_STEP4=1 RUN_STEP5=1 RUN_VERIFY=1
fi

# Run the pipeline
export PYTHONIOENCODING=utf-8
python3 "${PIPELINE_DIR}/run_pipeline.py"

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"
