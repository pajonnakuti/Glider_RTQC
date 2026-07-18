#!/usr/bin/env bash
# ============================================================
#  run_pipeline.sh  -  Glider Processing Pipeline
#
#  USAGE
#  -----
#  Basic (uses DATA_DIR set in pipeline/config.py):
#    bash run_pipeline.sh
#
#  Override data folder:
#    bash run_pipeline.sh /path/to/glider_data/890_2
#
#  With explicit L0 (skip binary decode, use existing L0):
#    bash run_pipeline.sh /path/to/data/890_2 /path/to/L0.nc
#
#  PYTHON ENVIRONMENT
#  ------------------
#  Looks for Python in this order:
#    1. conda 'glider' or 'base' environment
#    2. ~/glider_env/bin/python  (venv)
#    3. Any python3/python with required packages
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline"

# ---- Find Python with all required packages ----
_has_packages() {
    "$1" -c "import dbdreader, xarray, gsw, scipy, matplotlib, numpy, netCDF4" \
        >/dev/null 2>&1
}

PYTHON=""

# 1. conda environments
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    for ENV in glider base; do
        PY="$CONDA_BASE/envs/$ENV/bin/python"
        [ "$ENV" = "base" ] && PY="$CONDA_BASE/bin/python"
        if [ -x "$PY" ] && _has_packages "$PY"; then
            PYTHON="$PY"; break
        fi
    done
fi

# 2. ~/glider_env venv
if [ -z "$PYTHON" ] && [ -x "$HOME/glider_env/bin/python" ]; then
    _has_packages "$HOME/glider_env/bin/python" && PYTHON="$HOME/glider_env/bin/python"
fi

# 3. system python3/python
if [ -z "$PYTHON" ]; then
    for PY in python3 python; do
        if command -v "$PY" &>/dev/null && _has_packages "$PY"; then
            PYTHON="$PY"; break
        fi
    done
fi

# 4. fall back to whatever python3 is
if [ -z "$PYTHON" ]; then
    command -v python3 &>/dev/null && PYTHON=python3 || { echo "ERROR: No Python found."; exit 1; }
    echo "WARNING: Python found but may be missing packages. Run:"
    echo "  pip install dbdreader xarray gsw scipy matplotlib pandas netCDF4"
    echo ""
fi

echo "Python:   $($PYTHON --version 2>&1)"
echo "Pipeline: $PIPELINE_DIR"
echo ""

DATA_DIR_ARG=""
L0_ARG=""
SKIP_ARG=""

# --- Parse arguments ---
if [ -n "$1" ]; then
    DATA_DIR_ARG="--data-dir $1"
fi
if [ -n "$2" ]; then
    # Explicit L0 path provided — skip step 1
    L0_ARG="--l0-path $2"
    SKIP_ARG="--skip-step1"
fi

# --- Run ---
"$PYTHON" "$PIPELINE_DIR/run_pipeline.py" $DATA_DIR_ARG $L0_ARG $SKIP_ARG