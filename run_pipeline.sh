#!/usr/bin/env bash
# ============================================================
#  run_pipeline.sh  -  Glider Processing Pipeline
#
#  USAGE
#  -----
#  From the SSH machine or WSL:
#    bash run_pipeline.sh                          # uses DATA_DIR in config.py
#    bash run_pipeline.sh /path/to/data/890_2023   # override data folder
#
#  The script auto-finds Python with all dependencies installed.
#  It checks (in order):
#    1. conda 'glider' or 'base' environment
#    2. ~/glider_env/bin/python  (venv)
#    3. System python3 (may lack packages)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline"

# ---- Find Python with all required packages ----
PYTHON=""

# Helper: test if a python executable has all required packages
_has_packages() {
    local py="$1"
    "$py" -c "import dbdreader, xarray, gsw, scipy, matplotlib, numpy, netCDF4" \
        >/dev/null 2>&1
}

# 1. conda 'glider' env
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null)
    for ENV in glider base; do
        PY="$CONDA_BASE/envs/$ENV/bin/python"
        if [ ! -x "$PY" ] && [ "$ENV" = "base" ]; then
            PY="$CONDA_BASE/bin/python"
        fi
        if [ -x "$PY" ] && _has_packages "$PY"; then
            PYTHON="$PY"
            break
        fi
    done
fi

# 2. ~/glider_env venv
if [ -z "$PYTHON" ] && [ -x "$HOME/glider_env/bin/python" ]; then
    if _has_packages "$HOME/glider_env/bin/python"; then
        PYTHON="$HOME/glider_env/bin/python"
    fi
fi

# 3. Any python3 with packages
if [ -z "$PYTHON" ]; then
    for PY in python3 python; do
        if command -v "$PY" &>/dev/null && _has_packages "$PY"; then
            PYTHON="$PY"
            break
        fi
    done
fi

# 4. Fall back to whatever python3 is (warn about missing packages)
if [ -z "$PYTHON" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON=python3
        echo "WARNING: Could not find Python with all glider packages."
        echo "  Install with:  pip install dbdreader xarray gsw scipy matplotlib pandas netCDF4"
        echo "  Or activate your conda/venv environment first."
        echo ""
    else
        echo "ERROR: No Python found."
        exit 1
    fi
fi

echo "Python:   $($PYTHON --version 2>&1)"
echo "Pipeline: $PIPELINE_DIR"
echo ""

if [ -z "$1" ]; then
    "$PYTHON" "$PIPELINE_DIR/run_pipeline.py"
else
    "$PYTHON" "$PIPELINE_DIR/run_pipeline.py" --data-dir "$1"
fi