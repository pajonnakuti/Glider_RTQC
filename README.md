# Glider RTQC Pipeline

**INCOIS Slocum Glider L1 Processing Pipeline**

A fully automated pipeline that processes raw Slocum glider binary data into
science-ready L1 NetCDF files with ARGO RTQC flags, gridded products, and a
comprehensive set of oceanographic plots — for any glider deployment, anywhere.

---

## Quick Start

```bash
# On the SSH / Linux machine:
git clone https://github.com/pajonnakuti/Glider_RTQC.git
cd Glider_RTQC/Glider_RTQC

# Run on a specific data folder:
bash run_pipeline.sh /path/to/Raw_Data/1130-Mar-2025

# Or on Windows:
run_pipeline.bat T:\glider_data\890_2
```

The script auto-detects your Python environment, finds the data, and runs all
steps — no configuration required.

---

## What This Pipeline Does

The pipeline takes raw Slocum glider binary files (`.dbd`/`.dcd` flight,
`.ebd`/`.ecd` science) and produces a fully quality-controlled, ARGO-compliant
L1 dataset with 19 diagnostic plots.

### Processing Steps

| Step | Input | Output | Description |
|------|-------|--------|-------------|
| **Step 1** | Binary `.dbd/.dcd/.ebd/.ecd` | L0 NetCDF | Decode binaries, sync to science time axis, compute T/S/depth/profiles |
| **Step 2/3** | L0 NetCDF | L1 NetCDF | GliderTools-style QC + full ARGO RTQC flag tests 5–16 |
| **Step 4** | L1 NetCDF | Grid + per-profile NetCDFs | 1 m depth bins, QC flags applied before gridding |
| **Step 5** | L0 + Grid | L0 & L1 time-depth plots | Two PNG sections: raw data vs QC-filtered |
| **Step 6** | L0 + L1 + Grid | Track map, T-S diagram, coverage matrix, summary report | Deployment overview diagnostics |
| **Step 7** | Grid | 9 oceanographic section plots | Contour sections, envelopes, Hovmöller, gradients, isotherm depths |

---

## Outputs

All outputs go to `<DATA_DIR>/output/`:

```
output/
├── incois_glider_{ID}_L0.nc              # Raw decoded timeseries
├── l1/
│   └── incois_glider_{ID}_L1.nc          # QC-flagged L1 timeseries
├── gridfiles/
│   └── incois_glider_{ID}_grid.nc        # 2D time × depth grid (1 m bins)
├── profiles/
│   └── incois_glider_{ID}_profile_NNNN.nc  # One NetCDF per profile (NGDAC format)
├── plots/
│   ├── _L0_gridplot.png                  # Raw T/S/O2/Chl/CDOM time-depth
│   ├── _L1_gridplot.png                  # QC-filtered time-depth
│   ├── _track.png                        # GPS track on world map
│   ├── _ts_diagram.png                   # T-S scatter coloured by depth
│   ├── _ts_density.png                   # T-S with σ₀ contours, coloured by time & depth
│   ├── _data_coverage.png                # Sensor availability matrix
│   ├── _mld.png                          # Mixed layer depth timeseries
│   ├── _contour_temp.png                 # Potential temperature section + isopycnals
│   ├── _contour_salinity.png             # Salinity section + isopycnals
│   ├── _contour_oxygen.png               # Oxygen (raw + lag-corrected, dual panel)
│   ├── _contour_optics.png               # Chlorophyll, CDOM, backscatter sections
│   ├── _contour_density.png              # Potential density section
│   ├── _overview_section.png             # 5-panel summary: T/S/O2/Chl/density
│   ├── _profiles_envelope.png            # Min/IQR/median/max profile envelopes
│   ├── _surface_properties.png           # SST, SSS, surface O2, Chl timeseries
│   ├── _depth_timeseries.png             # Isotherm (10/15/20/25/28°C) depths
│   ├── _hovmoller.png                    # Temperature anomaly (T − time-mean)
│   └── _vertical_gradient.png           # dT/dz and dS/dz (thermocline structure)
└── reports/
    ├── incois_glider_{ID}_summary.txt    # Deployment summary (profiles, QC stats, gaps)
    └── incois_glider_{ID}_gt_comparison.txt  # GliderTools comparison (if installed)
```

---

## QC Methods

### Step 1 — Binary Decode
- Reads `.dcd`/`.dbd` (flight) and `.ecd`/`.ebd` (science) files via `dbdreader`
- Handles non-ASCII sensor names (e.g. French/Latin-1 characters) by skipping
  bad files and loading the rest one-by-one
- Handles corrupt/truncated files gracefully — skips them, reports count
- Suppresses dbdreader's internal "could not be loaded" noise
- GPS filtering using auto-detected deployment bounds
- Derives salinity (GSW), potential temperature, density, depth, profiles

### Step 2/3 — QC + ARGO Flags
**GliderTools-style processing (same methods, extended):**
- IQR outlier removal (global for optics, per-profile for T/S/O2)
- Median despike (window = 5 points)
- Savitzky-Golay smoothing per profile (window = 11 points, order 2)
- Horizontal diff filter for salinity
- Optics dark-count correction (in-situ deep-water baseline)
- Fluorescence quenching correction (Thomalla et al. 2017)
- Zhang et al. (2009) backscatter correction
- Variable corruption detection (nulls channels identical to pressure/oxygen)

**ARGO RTQC tests (Manual v3.9):**

| Test | Description |
|------|-------------|
| Test 5  | Impossible speed (>3 m/s between GPS fixes) |
| Test 6  | Global range check |
| Test 8  | Pressure monotonicity (per-profile) |
| Test 9  | Spike test (per-profile, shallow/deep thresholds) |
| Test 13 | Stuck value detection |
| Test 14 | Density inversion |
| Test 16 | Gross sensor drift |

**Additional:**
- Oxygen lag correction (first-order, τ = 30 s, profile-aware)
- Pressure → all variables cascade
- Temperature QC → salinity QC cascade

**QC Flags (ARGO convention):**
- `1` = Good  `2` = Probably good  `3` = Probably bad  `4` = Bad  `9` = Missing

### Step 4 — Gridding
- 1 m depth bins, `binned_statistic` mean per bin
- **QC flags applied before gridding** — only flag 1 (good) and 2 (probably good)
  values contribute to each bin
- Large time gaps (> 48 h) masked in plots to prevent pcolormesh stretching

---

## Auto-Detection

When you point the pipeline at a new data folder, it automatically detects:

| Setting | Source | Method |
|---------|--------|--------|
| GPS bounds | `*.mlg` log files | Parses `GPS Location:` lines, p5–p95 + 15% padding |
| Factory test location | GPS clusters > 10° from deployment median | Auto-sets cleaning box |
| Hemisphere cleaning | GPS median latitude | Enabled if |lat| > 5° |
| Max depth | `deployment.yml` `pressure.valid_max` | Falls back to 1000 dbar |
| Deployment year | `deployment.yml` or binary file headers | For mode-year filtering |
| L0 timeseries | `L0-timeseries/` subfolder (largest NC file) | Skips Step 1 if found |
| Binary files | Recursive walk: `aft/logs/`, `aft/sentlogs/`, any subfolder | Hard-links to `combined_binary/` |

---

## Comparison with GliderTools

Both pipelines share the same core QC methods. Ours extends GliderTools with:

| Feature | GliderTools | This Pipeline |
|---------|-------------|---------------|
| IQR outlier removal | ✓ | ✓ |
| Median despike | ✓ | ✓ |
| Savitzky-Golay smoothing | ✓ | ✓ |
| Horizontal diff filter | ✓ | ✓ |
| Optics corrections | ✓ | ✓ |
| ARGO RTQC flag tests | · | ✓ |
| QC flags in output files | · | ✓ |
| Oxygen lag correction | · | ✓ |
| Density inversion test | · | ✓ |
| Per-profile NetCDF (NGDAC) | · | ✓ |
| GPS track map | · | ✓ |
| 9 oceanographic section plots | · | ✓ |
| Deployment summary report | · | ✓ |
| Mixed layer depth | ✓ | ✓ |
| Bottle calibration | ✓ | · |
| Thermal lag correction | ✓ | · |

**Key finding:** On the 890_2 Arabian Sea deployment, both pipelines agree within
0.01°C for temperature (94% of co-located points) and within 0.01 PSU for
salinity (99.7%). The ~5-10% difference in valid point counts is from ARGO-specific
tests that GliderTools does not apply (pressure monotonicity, density inversion,
impossible speed).

**Important:** GliderTools' global IQR filter incorrectly removes warm surface
water (~30°C SST) when deep cold water dominates the distribution. This pipeline
uses per-profile QC for T/S/O2 to avoid this problem.

---

## Requirements

```bash
pip install numpy xarray scipy matplotlib gsw dbdreader pyyaml pandas netCDF4
# Optional (better plots):
pip install cartopy cmocean
# Optional (comparison):
pip install glidertools
```

`dbdreader` requires a C compiler on Windows. On Linux it installs cleanly.
Use the provided `run_pipeline.sh` which auto-finds the correct Python.

---

## Running on Different Data Products

```bash
# Any glider folder — the pipeline adapts automatically
bash run_pipeline.sh /path/to/Raw_Data/1130-Mar-2025
bash run_pipeline.sh /path/to/Raw_Data/1131-Data(Dec-2024)

# If you already have an L0 timeseries (skip binary decode):
bash run_pipeline.sh /path/to/data /path/to/L0-timeseries/file.nc

# If the data has an L0-timeseries/ subfolder:
# Step 1 is skipped automatically — the script detects and uses it
```

---

## Repository Structure

```
Glider_RTQC/
├── Glider_RTQC/                   ← pipeline code (this repo)
│   ├── run_pipeline.sh            ← Linux/Mac launcher (auto-finds Python)
│   ├── run_pipeline.bat           ← Windows launcher
│   ├── README.md                  ← this file
│   └── pipeline/
│       ├── config.py              ← auto-detection + deployment settings
│       ├── run_pipeline.py        ← orchestrator (--data-dir, --l0-path, --skip-step1)
│       ├── step1.py               ← binary → L0 (robust error handling)
│       ├── step23.py              ← L0 → L1 QC + ARGO flags
│       ├── step4.py               ← L1 → grid + profiles (QC-masked)
│       ├── step5.py               ← L0 + L1 time-depth plots
│       ├── step6.py               ← track map, T-S, coverage, MLD, summary
│       ├── step7.py               ← oceanographic section plots (9 types)
│       └── verify.py              ← L1 diagnostics (GPS, T-S, QC flags)
└── Raw_Data/                      ← your glider data goes here
    └── 1130-Mar-2025/
        ├── aft/logs/              ← Slocum binary files
        ├── aft/sentlogs/
        ├── deployment.yml         ← metadata (optional but recommended)
        ├── combined_binary/       ← auto-created: all binaries collected here
        ├── cache/                 ← auto-created: dbdreader cache
        └── output/                ← auto-created: all results
```

---

## Known Limitations

- **`dbdreader` on Windows** requires Microsoft C++ Build Tools to install.
  Use WSL or the SSH machine for deployments with raw binary files.
- **Optics variables** (chlorophyll, CDOM, backscatter) in some pre-processed
  L0 files have data mapping bugs (channels filled with pressure values).
  The pipeline auto-detects and nulls these via `detect_variable_corruption()`.
- **GliderTools `horizontal_diff_outliers`** crashes on NumPy ≥ 2.0 due to
  `numpy.NaN` removal. Our implementation is unaffected (uses `np.nan`).

---

## Citation / Acknowledgements

Pipeline developed at INCOIS (Indian National Centre for Ocean Information Services).  
QC methodology follows ARGO Data Management Team (2022), *Argo Quality Control Manual*, v3.9.  
Optics processing follows GliderTools (Gregor et al., 2019, *Front. Mar. Sci.*).  
Backscatter correction: Zhang et al. (2009), *Opt. Express*.  
Quenching correction: Thomalla et al. (2017), *Front. Mar. Sci.*
