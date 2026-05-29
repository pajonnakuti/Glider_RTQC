# Glider Processing Pipeline

Processes Slocum glider data from raw binary files to QC-flagged L1 NetCDF
with time-depth plots. Works for any deployment folder — auto-detects GPS
bounds, deployment year, hemisphere, and factory-test locations.

## Quick Start

**Windows:**
```
run_pipeline.bat T:\glider_data\890_2
```

**Linux / Mac:**
```
./run_pipeline.sh /data/glider/890_2
```

Or edit `pipeline/config.py` and set `DATA_DIR`, then run:
```
python pipeline/run_pipeline.py
```

## Folder Structure Expected

```
DATA_DIR/
  aft/logs/          <- Slocum binary logs (.dbd/.dcd)
  aft/sentlogs/      <- Slocum sent logs (.ebd/.ecd)
  deployment.yml     <- optional metadata (GPS bounds, depth, dates)
  cache/             <- auto-created: dbdreader cache files
  combined_binary/   <- auto-created: all binary files collected here
  output/            <- auto-created: all outputs
    incois_glider_{ID}_L0.nc
    l1/incois_glider_{ID}_L1.nc
    gridfiles/incois_glider_{ID}_grid.nc
    profiles/
    plots/
      incois_glider_{ID}_L0_gridplot.png   <- raw data, no QC
      incois_glider_{ID}_L1_gridplot.png   <- QC-filtered (flags 1 & 2)
```

Also works if the folder already contains a pre-processed L0 NetCDF
(e.g. from another pipeline). Step 1 can be skipped by setting
`RUN_STEP1 = False` in config.py.

## Pipeline Steps

| Step | Input | Output | Description |
|------|-------|--------|-------------|
| 1 | Binary (.dbd/.ecd) | L0 NetCDF | Decode, sync, derive T/S/depth/profiles |
| 2/3 | L0 NetCDF | L1 NetCDF | GliderTools QC + full ARGO RTQC flags |
| 4 | L1 NetCDF | Grid + profiles | 1m depth bins, per-profile NetCDFs |
| 5 | L0 + Grid | Two PNG plots | L0 raw plot + L1 QC-filtered plot |
| verify | L1 NetCDF | Console report | GPS, T-S, QC flag diagnostics |

## Auto-Detection

`detect_deployment()` runs before any processing and sets:
- **GPS bounds** from GPS fixes in `*.mlg` log files (p5–p95 + padding)
- **Factory test box** from outlier GPS clusters (> 10° from deployment median)
- **Max depth** from `deployment.yml` pressure valid_max
- **Deployment year** from `deployment.yml` or binary file headers
- **Hemisphere cleaning** enabled only if median lat > 5° from equator

## Requirements

```
pip install numpy xarray scipy matplotlib gsw dbdreader pyyaml pandas netCDF4
```

## Running for a Different Glider

Just change one line in `pipeline/config.py`:
```python
DATA_DIR = r"T:\glider_data\1126"
```
Or pass it on the command line — no other changes needed.
