#!/usr/bin/env bash
# ============================================================
#  run_pipeline.sh  -  Glider Processing Pipeline
#
#  USAGE
#  -----
#  Basic (uses DATA_DIR set in pipeline/config.py):
#    ./run_pipeline.sh
#
#  Override data folder at runtime:
#    ./run_pipeline.sh /path/to/glider_data/890_2
#    ./run_pipeline.sh /path/to/glider_data/1126
#
#  Skip step 1 (binary decode) if L0 already exists:
#    RUN_STEP1=0 ./run_pipeline.sh /path/to/glider_data/890_2
#
#  REQUIREMENTS
#  ------------
#  Python environment with:
#    numpy, xarray, scipy, matplotlib, gsw, dbdreader, pyyaml, pandas, netCDF4
#
#  Install missing packages:
#    pip install numpy xarray scipy matplotlib gsw dbdreader pyyaml pandas netCDF4
#
#  On Linux/Mac with conda:
#    conda activate base
#    ./run_pipeline.sh /data/glider/890_2
# ============================================================

set -e

# Resolve the directory this script lives in (works with symlinks too)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline"

# ---- Python interpreter ----
# Edit this line if your Python is somewhere else.
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: python not found. Activate your conda/venv environment first."
    exit 1
fi

echo "Using Python: $($PYTHON --version 2>&1)"
echo "Pipeline dir: $PIPELINE_DIR"
echo ""

# ---- Run ----
if [ -z "$1" ]; then
    "$PYTHON" "$PIPELINE_DIR/run_pipeline.py"
else
    "$PYTHON" "$PIPELINE_DIR/run_pipeline.py" --data-dir "$1"
fi