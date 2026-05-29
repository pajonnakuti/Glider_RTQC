@echo off
REM ============================================================
REM  run_pipeline.bat  -  Glider Processing Pipeline
REM
REM  USAGE:
REM    run_pipeline.bat                        (uses DATA_DIR in config.py)
REM    run_pipeline.bat T:\glider_data\1126    (override data folder)
REM
REM  REQUIREMENTS:
REM    conda activate base   (or any env with numpy, xarray, gsw, dbdreader)
REM ============================================================

setlocal

REM -- locate this script's directory
set SCRIPT_DIR=%~dp0

REM -- optional: activate conda env (edit env name if needed)
REM call conda activate base

REM -- run the pipeline
if "%~1"=="" (
    python "%SCRIPT_DIR%pipeline\run_pipeline.py"
) else (
    python "%SCRIPT_DIR%pipeline\run_pipeline.py" --data-dir "%~1"
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  Pipeline exited with errors. Check output above.
    pause
)
