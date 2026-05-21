@echo off
REM ============================================================
REM Glider Data Processing Pipeline - Windows Entry Point
REM ============================================================
REM Run the full pipeline: Binary -> L0 -> L1 -> Profiles/Grid -> Plots
REM
REM Usage:
REM   run_pipeline.bat                    - Run all steps with default config
REM   run_pipeline.bat --step 1           - Run only Step 1 (Binary -> L0)
REM   run_pipeline.bat --step 23          - Run only Step 2/3 (L0 -> L1)
REM   run_pipeline.bat --step 4           - Run only Step 4 (Profiles + Grid)
REM   run_pipeline.bat --step 5           - Run only Step 5 (Plots)
REM
REM Environment variables (optional):
REM   GLIDER_ID         - Glider identifier (default: 890_2)
REM   GLIDER_BASE_DIR   - Base directory with binary data
REM   GLIDER_OUTPUT_DIR - Output directory for processed data
REM ============================================================

setlocal enabledelayedexpansion

REM Set UTF-8 encoding to prevent charmap errors
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

REM Determine script directory
set "SCRIPT_DIR=%~dp0"
set "PIPELINE_DIR=%SCRIPT_DIR%pipeline"

echo.
echo ============================================================
echo   GLIDER DATA PROCESSING PIPELINE
echo ============================================================
echo   Pipeline: %PIPELINE_DIR%
echo.

REM Check for step override
set "STEP_OVERRIDE="
if "%~1"=="--step" set "STEP_OVERRIDE=%~2"

if defined STEP_OVERRIDE (
    echo   Running Step %STEP_OVERRIDE% only
    echo.
    if "!STEP_OVERRIDE!"=="1" (
        set "RUN_STEP1=1"
        set "RUN_STEP23=0"
        set "RUN_STEP4=0"
        set "RUN_STEP5=0"
        set "RUN_VERIFY=0"
    ) else if "!STEP_OVERRIDE!"=="23" (
        set "RUN_STEP1=0"
        set "RUN_STEP23=1"
        set "RUN_STEP4=0"
        set "RUN_STEP5=0"
        set "RUN_VERIFY=0"
    ) else if "!STEP_OVERRIDE!"=="4" (
        set "RUN_STEP1=0"
        set "RUN_STEP23=0"
        set "RUN_STEP4=1"
        set "RUN_STEP5=0"
        set "RUN_VERIFY=0"
    ) else if "!STEP_OVERRIDE!"=="5" (
        set "RUN_STEP1=0"
        set "RUN_STEP23=0"
        set "RUN_STEP4=0"
        set "RUN_STEP5=1"
        set "RUN_VERIFY=0"
    )
) else (
    echo   Running full pipeline (all steps)
    echo.
    set "RUN_STEP1=1"
    set "RUN_STEP23=1"
    set "RUN_STEP4=1"
    set "RUN_STEP5=1"
    set "RUN_VERIFY=1"
)

REM Run the pipeline
python "%PIPELINE_DIR%\run_pipeline.py"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo   PIPELINE FAILED - Check errors above
    echo ============================================================
    exit /b 1
)

echo.
echo ============================================================
echo   ALL DONE
echo ============================================================
pause
