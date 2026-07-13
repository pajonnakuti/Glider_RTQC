# INCOIS Glider RTQC Pipeline — Design Document

## Architecture

```
Raw Binary Files (.dbd/.dcd/.ebd/.ecd)
        │
        │  Step 0: pyglider.slocum.binary_to_timeseries()
        │  (external — generates L0 products)
        ▼
┌──────────────────────────────────────────────────────────┐
│  L0 Products (input to our pipeline)                     │
│  ├── L0-timeseries/incois_glider_{ID}_L0.nc              │
│  ├── L0-profiles/ (one .nc per profile)                  │
│  └── L0-gridfiles/ (time × depth grid, all vars)         │
└──────────────────────────────────────────────────────────┘
        │
        │  Our Pipeline (Steps 2-7)
        ▼
┌──────────────────────────────────────────────────────────┐
│  L1 Products (our output)                                │
│  ├── L1-timeseries/incois_glider_{ID}_L1.nc              │
│  ├── L1-profiles/ (one .nc per profile, QC applied)      │
│  ├── L1-gridfiles/ (time × depth, flags 1&2 only)        │
│  ├── plots/ (19 diagnostic PNGs)                         │
│  └── reports/ (summary.txt, gt_comparison.txt)           │
└──────────────────────────────────────────────────────────┘
```

---

## How L0 is Generated

**Tool:** `pyglider` (pyglider.slocum module)  
**Function:** `pyglider.slocum.binary_to_timeseries()`

This is an EXTERNAL tool — our pipeline does NOT generate L0. It consumes it.

What pyglider does:
1. Reads raw Slocum `.dbd`/`.dcd` (flight) and `.ebd`/`.ecd` (science) files
2. Decodes binary format using sensor list cache
3. Syncs flight and science onto a common time axis
4. Computes derived variables (salinity, depth, density via GSW)
5. Detects profiles (dive/climb) from pressure direction changes
6. Writes L0 timeseries NetCDF with CF-1.8 conventions
7. Optionally writes L0 profiles and L0 grid

Our pipeline has a fallback `step1.py` that does the same thing using `dbdreader` — but the primary workflow is: **pyglider generates L0, our pipeline generates L1.**

---

## How L1 is Generated — Our Pipeline vs GliderTools

### GliderTools L1 Processing Flow

GliderTools provides individual QC functions — it is NOT a pipeline. The user calls each function manually. The typical workflow:

```python
import glidertools as gt

# 1. Load data
dives = ds.profile_index.values
depth = ds.depth.values
temp  = ds.temperature.values

# 2. Physics QC (temperature/salinity)
temp_clean = gt.calc_physics(temp, dives, depth,
    spike_window=3, spike_method='minmax',
    iqr=1.5, savitzky_golay_window=11,
    depth_threshold=400, mask_frac=0.2)

# 3. Backscatter processing
bbp = gt.calc_backscatter(bb_raw, temp, salt, dives, depth,
    wavelength=700, dark_count=50, scale_factor=3.17e-5)

# 4. Fluorescence + quenching correction
chl = gt.calc_fluorescence(flr_raw, bbp, dives, depth, time, lat, lon,
    dark_count=50, scale_factor=0.012)

# 5. Grid
temp_grid = gt.grid_data(dives, depth, temp_clean, bins=1.0)
```

**What GliderTools does internally in `calc_physics`:**
1. `outlier_bounds_iqr(var, multiplier=1.5)` — global IQR filter
2. `despike(var, window=3, method='minmax')` — spike removal
3. `horizontal_diff_outliers(dives, depth, var)` — cross-profile outliers
4. `savitzky_golay(var, window=11, order=2)` — smoothing

**What GliderTools does internally in `calc_backscatter`:**
1. `outlier_bounds_iqr(bb, multiplier=3)` — global IQR
2. `despike(bb, window=7, method='median')` — despiking
3. Scale raw counts → physical units using dark count + scale factor
4. Zhang et al. (2009) seawater subtraction
5. `find_bad_profiles(dives, depth, bbp)` — bad profile removal
6. In-situ dark count correction

**What GliderTools does internally in `calc_fluorescence`:**
1. Same as backscatter (IQR, despike, scale, dark count)
2. `quenching_correction(flr, bbp, dives, depth, time, lat, lon)` — Thomalla et al. 2017

### Our Pipeline L1 Processing Flow (step23.py)

```
L0 timeseries
    │
    ├── Pre-cleaning
    │   ├── Remove factory-test GPS locations
    │   ├── Remove (0,0) sentinel positions
    │   ├── Mode-year filtering (remove pre-deployment data)
    │   └── Hemisphere filtering
    │
    ├── Variable corruption detection
    │   └── If any optics variable = pressure (corr > 0.999), null it
    │
    ├── Optics processing (same as GliderTools)
    │   ├── IQR outlier removal (multiplier=3)
    │   ├── In-situ dark count correction
    │   ├── Negative clipping (≥0)
    │   ├── Median despike (window=7)
    │   ├── Zhang backscatter correction
    │   ├── Bad profile detection
    │   └── Quenching correction (Thomalla et al. 2017)
    │
    ├── Physics QC (different from GliderTools!)
    │   ├── Physical range guard (NOT global IQR — avoids warm water problem)
    │   ├── Median despike (window=5)
    │   ├── Per-profile Savitzky-Golay smoothing (window=11, order=2)
    │   ├── Horizontal diff outliers for salinity
    │   └── Oxygen negative clipping
    │
    ├── Oxygen lag correction (tau=30s, profile-aware)
    │   └── GliderTools does NOT do this
    │
    └── ARGO RTQC flag tests (GliderTools does NOT do this)
        ├── Test 5: Impossible speed
        ├── Test 6: Global range
        ├── Test 8: Pressure increasing (per-profile)
        ├── Test 9: Spike test
        ├── Test 13: Stuck value detection
        ├── Test 14: Density inversion
        ├── Test 16: Gross sensor drift
        ├── Pressure → all variables cascade
        └── Temperature → salinity cascade
```

---

## Key Differences: Our Pipeline vs GliderTools

| Aspect | GliderTools | Our Pipeline |
|--------|-------------|--------------|
| **Type** | Library of functions — user assembles workflow | Automated end-to-end pipeline |
| **T/S QC method** | Global IQR (cuts warm surface water!) | Physical range + per-profile despike (preserves structure) |
| **Optics QC** | Same methods | Same methods (ported from GT) |
| **Formal QC flags** | None — just masks data | ARGO flags 1/2/3/4/9 in output |
| **Oxygen correction** | Only unit conversion | Lag correction (τ=30s) |
| **Output format** | User's choice | CF-1.8 NetCDF with NGDAC attributes |
| **Profiles** | Not generated | Per-profile NetCDF files |
| **Grid** | Yes (user calls `grid_data`) | Yes + QC applied before binning |
| **Plots** | None built-in | 19 diagnostic PNGs |
| **Data validation** | None | Variable corruption detection |
| **Auto-config** | None — manual | GPS bounds, depth, year from data |

### Why We Don't Use Global IQR for Temperature/Salinity

GliderTools' `calc_physics` uses `outlier_bounds_iqr(temp, multiplier=1.5)` which computes bounds over the ENTIRE deployment. For a glider profiling from surface (~30°C) to 1000m (~8°C):
- Q1 ≈ 10°C, Q3 ≈ 22°C, IQR = 12°C
- Upper bound = 22 + 1.5×12 = **40°C** ✓
- Lower bound = 10 - 1.5×12 = **-8°C** ✓

But after smoothing, the bounds tighten. The deeper you go, the more dominant cold water becomes. On the 890_2 deployment, GliderTools max temp was 21.9°C while reality is 31°C. **30°C surface water gets flagged as outlier.**

Our approach: physical range limits + per-profile despike + Savitzky-Golay. No global statistical filter that can be fooled by the depth distribution.

---

## Quantitative Comparison (890_2 deployment)

```
Temperature:
  GliderTools retained:  93.6% (max 21.9°C — cuts warm surface!)
  Our pipeline retained: 90.2% (max 35.4°C — correct)
  Agreement at co-located good points: 94.3% within 0.01°C

Salinity:
  GliderTools retained:  93.8%
  Our pipeline retained: 81.3% (more removed by pressure cascade)
  Agreement: 99.7% within 0.01 PSU

Oxygen:
  GliderTools retained:  95.1%
  Our pipeline retained: 96.6%
  Our pipeline adds: lag-corrected version (tau=30s)
```

---

## What Each File Does

| File | Role |
|------|------|
| `config.py` | Auto-detection of GPS bounds, depth, year; all settings |
| `run_pipeline.py` | Orchestrator — calls all steps in order |
| `step1.py` | Binary → L0 (fallback if pyglider L0 doesn't exist) |
| `step23.py` | L0 → L1 QC (GT-style optics + ARGO flags) |
| `step4.py` | Profile splitting + gridding (works for both L0 and L1) |
| `step5.py` | L0 + L1 time-depth section plots |
| `step6.py` | Track map, T-S diagram, MLD, coverage matrix, GT comparison |
| `step7.py` | Oceanographic contour sections, envelopes, gradients |
| `verify.py` | L1 product diagnostics (GPS, T-S, QC flags) |
| `run_pipeline.sh` | Linux launcher (auto-finds Python, auto-detects L0) |
| `run_pipeline.bat` | Windows launcher |

---

## Usage

```bash
# The full flow:
# 1. Generate L0 with pyglider (or use existing L0)
# 2. Place L0 in: Raw_Data/{deployment}/L0-timeseries/
# 3. Run our pipeline:

bash run_pipeline.sh /path/to/Raw_Data/1130-Mar-2025

# Or with explicit L0 path:
bash run_pipeline.sh /path/to/data /path/to/L0.nc
```

---

## Known Issues

1. **Corrupted optics in some L0 files** — chlorophyll/CDOM/backscatter channels filled with pressure values. Detected automatically and nulled.
2. **Non-ASCII sensor names** — some Slocum firmware writes French characters in sensor lists. Our `_open_multidbd` skips these files.
3. **Missing cache files** — `.dbd` files need `.cac` entries. Pipeline falls back to file-by-file loading.
4. **GliderTools NumPy 2.0 incompatibility** — GT's `horizontal_diff_outliers` imports `numpy.NaN` which was removed. Our implementation is unaffected.

---

## References

- ARGO Data Management Team (2022). *Argo Quality Control Manual*, Version 3.9
- Gregor et al. (2019). *GliderTools: A Python Toolbox for Processing Underwater Glider Data*. Front. Mar. Sci.
- Zhang et al. (2009). *Scattering by pure seawater*. Opt. Express
- Thomalla et al. (2017). *High-resolution fluorescence quenching correction*. Front. Mar. Sci.
- pyglider: https://pyglider.readthedocs.io/
