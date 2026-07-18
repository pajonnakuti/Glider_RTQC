#!/usr/bin/env python3
"""
step5.py - Generates TWO time-depth plots per deployment:

  1. L0_gridplot  — raw data from L0 NetCDF, all finite values, no QC masking.
                    Shows what came off the glider before any processing.

  2. L1_gridplot  — QC-filtered data from L1 NetCDF (grid), only ARGO flag 1
                    (good) and flag 2 (probably good) values included.
                    This is the science-ready product.

Large time gaps (> GAP_THRESHOLD_HOURS) are masked in both plots so
pcolormesh does not stretch a cell across a multi-week blank period.
"""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID, PLOT_DEPTH_MAX, DEPTH_BIN, GRID_TIME_BIN_H

# Gaps longer than this (hours) are masked in pcolormesh
GAP_THRESHOLD_HOURS = 48

VARS_TO_PLOT = ["potential_temperature", "salinity",
                "oxygen_concentration", "chlorophyll", "cdom"]
CMAPS        = ["RdYlBu_r", "viridis", "viridis", "viridis", "RdBu_r"]

# Fallback variable names: if primary name not in dataset, try these
VAR_FALLBACKS = {
    "potential_temperature": "temperature",
    "salinity": None,  # no fallback
}

VAR_LABELS   = {
    "potential_temperature": "water potential temperature [Celsius]",
    "temperature":           "water temperature [Celsius]",
    "salinity":              "water salinity [1e-3]",
    "oxygen_concentration":  "oxygen concentration [umol l-1]",
    "chlorophyll":           "chlorophyll [mg m-3]",
    "cdom":                  "CDOM [ppb]",
}


# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

def _resolve_var(ds, var_name):
    """Resolve a variable name, checking fallbacks if primary doesn't exist."""
    if var_name in ds:
        return var_name
    fallback = VAR_FALLBACKS.get(var_name)
    if fallback and fallback in ds:
        return fallback
    return None


def _pcolormesh_edges(centers):
    """Compute cell edges from center values (handles datetime64)."""
    c = np.asarray(centers)
    if np.issubdtype(c.dtype, np.datetime64):
        c_num = mdates.date2num(c)
        edges = np.empty(len(c) + 1)
        edges[1:-1] = (c_num[1:] + c_num[:-1]) / 2.0
        edges[0]    = c_num[0]  - (c_num[1]  - c_num[0])  / 2.0
        edges[-1]   = c_num[-1] + (c_num[-1] - c_num[-2]) / 2.0
        return edges
    edges = np.empty(len(c) + 1)
    edges[1:-1] = (c[1:] + c[:-1]) / 2.0
    edges[0]    = c[0]  - (c[1]  - c[0])  / 2.0
    edges[-1]   = c[-1] + (c[-1] - c[-2]) / 2.0
    return edges


def _mask_time_gaps(data, time_values, gap_hours=GAP_THRESHOLD_HOURS):
    """NaN out the first profile after each large time gap."""
    if len(time_values) < 2:
        return data
    t_h = time_values.astype("datetime64[h]").astype(float)
    dt  = np.diff(t_h)
    gap_starts = np.where(dt > gap_hours)[0] + 1
    if len(gap_starts) == 0:
        return data
    masked = data.copy()
    for idx in gap_starts:
        masked[idx, :] = np.nan
    return masked


def _max_data_depth(ds, vars_list, depth_vals, coverage_threshold=0.10):
    """
    Find the deepest depth bin with meaningful data coverage.

    Universal approach: deepest bin where at least `coverage_threshold`
    fraction of profiles have data. With bin-averaging (no interpolation),
    this works correctly for any glider depth range.
    """
    n_depth = len(depth_vals)
    total_coverage = np.zeros(n_depth, dtype=float)
    n_profiles_max = 0

    for var in vars_list:
        if var in ds:
            v = ds[var].values
            if v.ndim == 2 and v.shape[1] == n_depth:
                n_profiles = v.shape[0]
                n_profiles_max = max(n_profiles_max, n_profiles)
                col_counts = np.sum(np.isfinite(v), axis=0).astype(float)
                total_coverage = np.maximum(total_coverage, col_counts)

    if n_profiles_max == 0:
        return float(depth_vals.max()) if len(depth_vals) > 0 else 1000.0

    frac = total_coverage / n_profiles_max
    bins_with_coverage = np.where(frac >= coverage_threshold)[0]
    if len(bins_with_coverage) > 0:
        max_d = float(depth_vals[bins_with_coverage[-1]])
        padding = max(10.0, max_d * 0.05)
        return min(max_d + padding, float(depth_vals.max()))

    # Fallback: any bin with data
    any_data = np.where(total_coverage > 0)[0]
    if len(any_data) > 0:
        return float(depth_vals[any_data[-1]])

    return float(depth_vals.max()) if len(depth_vals) > 0 else 1000.0


def _report_gaps(t_vals):
    if len(t_vals) < 2:
        return
    dt_h = np.diff(t_vals.astype("datetime64[h]").astype(float))
    big  = np.where(dt_h > GAP_THRESHOLD_HOURS)[0]
    if len(big):
        print(f"  Time gaps > {GAP_THRESHOLD_HOURS}h: {len(big)}")
        for gi in big:
            print(f"    {str(t_vals[gi])[:19]} -> {str(t_vals[gi+1])[:19]}"
                  f"  ({dt_h[gi]:.0f}h)")


def _draw_pcolormesh(ax, t_vals, depth_vals, V, cmap, label, max_depth):
    """Draw one pcolormesh panel. Returns True if data was plotted."""
    # Guard: V must be exactly 2D (n_time × n_depth)
    if V.ndim != 2 or V.shape != (len(t_vals), len(depth_vals)):
        ax.set_title(f"{label} (not 2D)", fontsize=13)
        ax.set_ylim(max_depth, 0)
        ax.set_ylabel("Depth (m)", fontsize=11)
        return False

    depth_mask = depth_vals <= max_depth
    V_trim = V[:, depth_mask].copy()
    d_trim = depth_vals[depth_mask]

    # Suppress isolated depth artefacts: NaN out depth bins where fewer than
    # 10% of profiles have data (removes interpolation artefacts and pressure
    # spike horizontal lines)
    n_profiles = V_trim.shape[0]
    if n_profiles > 5:
        col_counts = np.sum(np.isfinite(V_trim), axis=0).astype(float)
        sparse_bins = col_counts < max(n_profiles * 0.10, 3)
        V_trim[:, sparse_bins] = np.nan

    V_trim = _mask_time_gaps(V_trim, t_vals, GAP_THRESHOLD_HOURS)

    valid = np.isfinite(V_trim)
    if np.all(~valid):
        ax.set_title(f"{label} (ALL NaN)", fontsize=13)
        ax.set_ylim(max_depth, 0)
        ax.set_ylabel("Depth (m)", fontsize=11)
        return False

    n_valid = int(np.sum(valid))
    print(f"    {label}: {n_valid}/{V_trim.size} filled cells")

    v_min = np.nanpercentile(V_trim[valid], 2)
    v_max = np.nanpercentile(V_trim[valid], 98)

    t_edges = _pcolormesh_edges(t_vals)
    d_edges = _pcolormesh_edges(d_trim)

    mesh = ax.pcolormesh(t_edges, d_edges, V_trim.T,
                         cmap=cmap, vmin=v_min, vmax=v_max, shading="flat")

    # Add contour lines for structure (matching team's plot style)
    try:
        import numpy.ma as ma
        V_ma = ma.masked_invalid(V_trim)
        X, Y = np.meshgrid(mdates.date2num(t_vals), d_trim)
        n_contours = 10
        levels = np.linspace(v_min, v_max, n_contours)
        ax.contour(X, Y, V_ma.T, levels=levels,
                   colors="k", linewidths=0.4, alpha=0.5)
    except Exception:
        pass

    ax.set_ylim(max_depth, 0)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar_label = VAR_LABELS.get(label, label)
    cbar.set_label(cbar_label, fontsize=11)
    return True


def _finalise_fig(fig, axes, plot_path):
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, fontsize=11)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")
    plt.close(fig)


# ----------------------------------------------------------------
# L0 plot  — raw data, no QC masking
# ----------------------------------------------------------------

def _make_l1_grid_from_ts(l1_ds, depth_bin=None, time_bin_h=None):
    """
    Build a 2D grid from L1 timeseries using UNIFORM TIME BINS.
    Each grid column covers a fixed time interval (default 3h), aggregating
    all observations in that window regardless of profile boundaries.
    This eliminates the picket-fence pattern from irregular profile spacing.
    Returns (grid_data dict, time_arr, depth_centers) or (None, None, None).
    """
    if depth_bin is None:
        depth_bin = DEPTH_BIN
    if time_bin_h is None:
        time_bin_h = GRID_TIME_BIN_H

    depth_raw = l1_ds["depth"].values if "depth" in l1_ds else None
    if depth_raw is None:
        return None, None, None

    finite_depths = depth_raw[np.isfinite(depth_raw)]
    if len(finite_depths) == 0:
        return None, None, None
    max_depth = float(np.percentile(finite_depths, 99.5))
    if max_depth < 10:
        max_depth = PLOT_DEPTH_MAX or 1000.0
    depth_centers = np.arange(depth_bin / 2.0, max_depth + depth_bin / 2.0, depth_bin)
    n_depth = len(depth_centers)

    # Build uniform time axis
    time_vals = l1_ds.time.values
    t_float = time_vals.astype("datetime64[s]").astype(float)
    finite_mask = np.isfinite(t_float)
    if np.sum(finite_mask) == 0:
        return None, None, None
    t_start = time_vals[finite_mask][0]
    t_end = time_vals[finite_mask][-1]
    time_bin_td = np.timedelta64(int(time_bin_h * 3600), 's')
    time_edges = np.arange(t_start, t_end + time_bin_td, time_bin_td)
    time_centers = time_edges[:-1] + time_bin_td // 2
    n_time = len(time_centers)

    if n_time == 0:
        return None, None, None

    # Digitize time and depth for fast bin-aggregation
    t_edge_float = time_edges.astype("datetime64[s]").astype(float)
    t_bin_idx = np.clip(np.digitize(t_float, t_edge_float) - 1, 0, n_time - 1)

    d_edge = np.arange(0, max_depth + depth_bin, depth_bin)
    d_bin_idx = np.clip(np.digitize(depth_raw, d_edge) - 1, 0, n_depth - 1)

    grid_2d = {}
    for var in VARS_TO_PLOT:
        actual_var = _resolve_var(l1_ds, var)
        if actual_var is None:
            continue
        v_arr = l1_ds[actual_var].values

        # Apply QC mask
        qc_var = f"{actual_var}_QC"
        if qc_var not in l1_ds:
            # Try the canonical name QC var too
            qc_var = f"{var}_QC"
        if qc_var in l1_ds:
            qc = l1_ds[qc_var].values.astype(float)
            qc_ok = (qc == 1) | (qc == 2)
        else:
            qc_ok = np.ones(len(v_arr), dtype=bool)

        valid = np.isfinite(depth_raw) & np.isfinite(v_arr) & qc_ok
        if np.sum(valid) == 0:
            continue

        # Aggregate into grid
        sums = np.zeros((n_time, n_depth))
        counts = np.zeros((n_time, n_depth))
        vi = np.where(valid)[0]
        np.add.at(sums, (t_bin_idx[vi], d_bin_idx[vi]), v_arr[vi])
        np.add.at(counts, (t_bin_idx[vi], d_bin_idx[vi]), 1)
        grid_2d[var] = np.where(counts > 0, sums / counts, np.nan)

    return grid_2d, time_centers, depth_centers


def _make_l0_grid(l0_ds, depth_bin=None, time_bin_h=None):
    """
    Build a 2D grid from L0 timeseries using UNIFORM TIME BINS.
    All finite values included, no QC masking.
    Returns (grid_data dict, time_arr, depth_centers).
    """
    if depth_bin is None:
        depth_bin = DEPTH_BIN
    if time_bin_h is None:
        time_bin_h = GRID_TIME_BIN_H

    depth_raw = l0_ds["depth"].values if "depth" in l0_ds else None
    if depth_raw is None:
        return None, None, None

    finite_depths = depth_raw[np.isfinite(depth_raw)]
    if len(finite_depths) == 0:
        return None, None, None
    max_depth = float(np.percentile(finite_depths, 99.5))
    if max_depth < 10:
        max_depth = PLOT_DEPTH_MAX or 1000.0
    depth_centers = np.arange(depth_bin / 2.0, max_depth + depth_bin / 2.0, depth_bin)
    n_depth = len(depth_centers)

    # Build uniform time axis
    time_vals = l0_ds.time.values
    t_float = time_vals.astype("datetime64[s]").astype(float)
    finite_mask = np.isfinite(t_float)
    if np.sum(finite_mask) == 0:
        return None, None, None
    t_start = time_vals[finite_mask][0]
    t_end = time_vals[finite_mask][-1]
    time_bin_td = np.timedelta64(int(time_bin_h * 3600), 's')
    time_edges = np.arange(t_start, t_end + time_bin_td, time_bin_td)
    time_centers = time_edges[:-1] + time_bin_td // 2
    n_time = len(time_centers)

    if n_time == 0:
        return None, None, None

    # Digitize time and depth
    t_edge_float = time_edges.astype("datetime64[s]").astype(float)
    t_bin_idx = np.clip(np.digitize(t_float, t_edge_float) - 1, 0, n_time - 1)

    d_edge = np.arange(0, max_depth + depth_bin, depth_bin)
    d_bin_idx = np.clip(np.digitize(depth_raw, d_edge) - 1, 0, n_depth - 1)

    grid_2d = {}
    for var in VARS_TO_PLOT:
        actual_var = _resolve_var(l0_ds, var)
        if actual_var is None:
            continue
        v_arr = l0_ds[actual_var].values
        valid = np.isfinite(depth_raw) & np.isfinite(v_arr)
        if np.sum(valid) == 0:
            continue

        sums = np.zeros((n_time, n_depth))
        counts = np.zeros((n_time, n_depth))
        vi = np.where(valid)[0]
        np.add.at(sums, (t_bin_idx[vi], d_bin_idx[vi]), v_arr[vi])
        np.add.at(counts, (t_bin_idx[vi], d_bin_idx[vi]), 1)
        grid_2d[var] = np.where(counts > 0, sums / counts, np.nan)

    return grid_2d, time_centers, depth_centers


def plot_l0(l0_path, plot_path=None):
    """Generate the L0 raw gridplot."""
    print("  Generating L0 plot (raw, no QC)...")

    if not os.path.exists(l0_path):
        print(f"  WARNING: L0 file not found: {l0_path}")
        return None

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_L0_gridplot.png")

    ds = xr.open_dataset(l0_path)

    grid_2d, t_arr, depth_centers = _make_l0_grid(ds)
    if grid_2d is None:
        print("  WARNING: could not build L0 grid (no profile_index or depth)")
        ds.close()
        return None

    _report_gaps(t_arr)

    # Determine max plot depth — universal approach:
    # Find the deepest depth bin where at least 10% of profiles have data.
    # With bin-averaging (no interpolation), only bins with actual
    # measurements are non-NaN, so this works for any depth range.
    n_profiles = len(t_arr)
    max_depth = 0.0
    for var in VARS_TO_PLOT:
        if var in grid_2d:
            v = grid_2d[var]
            col_counts = np.sum(np.isfinite(v), axis=0).astype(float)
            frac = col_counts / max(n_profiles, 1)
            bins_ok = np.where(frac >= 0.10)[0]
            if len(bins_ok) > 0:
                max_depth = max(max_depth, float(depth_centers[bins_ok[-1]]))
    if max_depth < 10:
        max_depth = float(depth_centers.max())
    else:
        # Add small padding
        padding = max(10.0, max_depth * 0.05)
        max_depth = min(max_depth + padding, float(depth_centers.max()))

    print(f"  L0 plot depth: {max_depth:.0f} m")

    fig, axes = plt.subplots(len(VARS_TO_PLOT), 1,
                             figsize=(14, 4 * len(VARS_TO_PLOT)), sharex=True)
    fig.suptitle(f"Glider {GLIDER_ID}  —  L0 Raw Data (no QC)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, (var, cmap) in enumerate(zip(VARS_TO_PLOT, CMAPS)):
        ax = axes[i]
        if var not in grid_2d:
            ax.set_title(f"{var} (NOT IN L0)", fontsize=13)
            ax.set_ylim(max_depth, 0)
            ax.set_ylabel("Depth (m)", fontsize=11)
            continue
        _draw_pcolormesh(ax, t_arr, depth_centers, grid_2d[var],
                         cmap, var, max_depth)

    _finalise_fig(fig, axes, plot_path)
    ds.close()
    return plot_path


# ----------------------------------------------------------------
# L1 plot  — QC-filtered grid (flags 1 & 2 only)
# ----------------------------------------------------------------

def plot_l1(grid_path, plot_path=None, l1_path=None):
    """Generate the L1 QC-filtered gridplot."""
    print("  Generating L1 plot (QC flags 1 & 2 only)...")

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Try to find the grid file (multiple naming conventions)
    if grid_path is None or not os.path.exists(grid_path):
        # Search common grid file patterns
        candidates = [
            grid_path,
            os.path.join(OUTPUT_DIR, "L1-gridfiles",
                         f"incois_glider_{GLIDER_ID}_L1_grid.nc"),
            os.path.join(OUTPUT_DIR, "gridfiles",
                         f"incois_glider_{GLIDER_ID}_L1_grid.nc"),
            os.path.join(OUTPUT_DIR, "gridfiles",
                         f"incois_glider_{GLIDER_ID}_grid.nc"),
        ]
        grid_path = None
        for c in candidates:
            if c and os.path.exists(c):
                grid_path = c
                break

    ds = None
    is_1d = False

    # Prefer building uniform-time-bin grid from L1 timeseries for plotting.
    # This eliminates picket-fence artifacts from irregular profile spacing.
    if l1_path is not None and os.path.exists(l1_path):
        print(f"  Building L1 plot grid from timeseries (uniform time bins)...")
        l1_ds = xr.open_dataset(l1_path)
        if "depth" in l1_ds:
            grid_2d, t_arr, depth_centers = _make_l1_grid_from_ts(l1_ds)
            if grid_2d is not None:
                l1_ds.close()
                ds = xr.Dataset(coords={"time": t_arr, "depth": depth_centers})
                for var in VARS_TO_PLOT:
                    if var in grid_2d:
                        ds[var] = xr.DataArray(grid_2d[var],
                                               dims=["time", "depth"])
                is_1d = False
            else:
                l1_ds.close()

    # Fallback: use pre-built grid file if timeseries approach failed
    if ds is None and grid_path and os.path.exists(grid_path):
        print(f"  Loading grid: {grid_path}")
        ds = xr.open_dataset(grid_path)
        if "depth" in ds and "time" in ds:
            has_2d = False
            for var in VARS_TO_PLOT:
                if var in ds and ds[var].ndim == 2:
                    has_2d = True
                    break
            if not has_2d:
                print(f"  WARNING: grid file has no 2D variables — "
                      f"falling back to L1 timeseries")
                ds.close()
                ds = None

    if ds is None and l1_path is not None and os.path.exists(l1_path):
        print(f"  Loading L1 timeseries (grid not usable): {l1_path}")
        ds = xr.open_dataset(l1_path)
        # Build uniform-time-bin grid on-the-fly from timeseries
        if "depth" in ds:
            print(f"  Building L1 grid on-the-fly (uniform time bins)...")
            grid_2d, t_arr, depth_centers = _make_l1_grid_from_ts(ds)
            if grid_2d is not None:
                ds.close()
                gds = xr.Dataset(coords={"time": t_arr, "depth": depth_centers})
                for var in VARS_TO_PLOT:
                    if var in grid_2d:
                        gds[var] = xr.DataArray(grid_2d[var],
                                                dims=["time", "depth"])
                ds = gds
                is_1d = False
            else:
                is_1d = ("depth" in ds and ds["depth"].ndim == 1
                         and ds.depth.dims[0] == "time")
        else:
            is_1d = ("depth" in ds and ds["depth"].ndim == 1
                     and ds.depth.dims[0] == "time")

    if ds is None:
        print(f"  WARNING: no grid or L1 file found for L1 plot")
        return None

    if "potential_density" not in ds and "density" in ds:
        ds["potential_density"] = ds["density"]

    ds = ds.sortby("time")

    # Diagnostic: show what's in the loaded dataset
    if not is_1d:
        print(f"  Grid dims: time={len(ds.time)} depth={len(ds.depth)}")
        for var in VARS_TO_PLOT:
            if var in ds:
                print(f"    {var}: shape={ds[var].shape} dims={ds[var].dims}")

    if plot_path is None:
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_L1_gridplot.png")

    t_vals     = ds.time.values
    depth_vals = ds.depth.values if not is_1d else None

    _report_gaps(t_vals)

    if not is_1d:
        max_depth = _max_data_depth(ds, VARS_TO_PLOT, depth_vals)
        if np.isnan(max_depth) or max_depth < 10:
            max_depth = PLOT_DEPTH_MAX or 1000.0
    else:
        max_depth = float(np.nanmax(ds.depth.values))
        if np.isnan(max_depth) or max_depth < 10:
            max_depth = PLOT_DEPTH_MAX or 1000.0

    print(f"  L1 plot depth: {max_depth:.0f} m")

    fig, axes = plt.subplots(len(VARS_TO_PLOT), 1,
                             figsize=(14, 4 * len(VARS_TO_PLOT)), sharex=True)
    fig.suptitle(f"Glider {GLIDER_ID}  —  L1 QC-Filtered (flags 1 & 2)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, (var, cmap) in enumerate(zip(VARS_TO_PLOT, CMAPS)):
        ax = axes[i]
        if var not in ds:
            ax.set_title(f"{var} (NOT IN DATASET)", fontsize=13)
            ax.set_ylim(max_depth, 0)
            ax.set_ylabel("Depth (m)", fontsize=11)
            continue

        V = ds[var].values

        # Handle variables that might have different dimension names
        if not is_1d:
            if V.ndim != 2:
                # Try to reshape if it's a 1D variable in a grid dataset
                # (shouldn't happen with make_grid output, but be defensive)
                ax.set_title(f"{var} (NOT 2D: ndim={V.ndim})", fontsize=13)
                ax.set_ylim(max_depth, 0)
                ax.set_ylabel("Depth (m)", fontsize=11)
                continue
            # Accept any 2D variable — use its actual shape for depth axis
            if V.shape[0] == len(t_vals) and V.shape[1] == len(depth_vals):
                pass  # perfect match
            elif V.shape[0] == len(t_vals):
                # Depth dimension might differ — use variable's actual depth
                # This handles cases where depth coord length doesn't match
                print(f"    {var}: shape {V.shape} (depth_vals={len(depth_vals)})")
                depth_vals_eff = np.arange(V.shape[1]) * DEPTH_BIN
                _draw_pcolormesh(ax, t_vals, depth_vals_eff, V,
                                 cmap, var, max_depth)
                continue
            else:
                ax.set_title(f"{var} (shape mismatch: {V.shape})", fontsize=13)
                ax.set_ylim(max_depth, 0)
                ax.set_ylabel("Depth (m)", fontsize=11)
                continue

        if is_1d:
            depth_1d = ds.depth.values
            valid = np.isfinite(V) & np.isfinite(depth_1d)
            qc_var = f"{var}_QC"
            if qc_var in ds:
                qc    = ds[qc_var].values.astype(int)
                valid = valid & ((qc == 1) | (qc == 2))
            n_valid = int(np.sum(valid))
            print(f"    {var}: {n_valid} points (scatter)")
            if n_valid == 0:
                ax.set_title(f"{var} (NO DATA)", fontsize=13)
                ax.set_ylim(max_depth, 0)
                ax.set_ylabel("Depth (m)", fontsize=11)
                continue
            v_min = np.nanpercentile(V[valid], 2)
            v_max = np.nanpercentile(V[valid], 98)
            sc = ax.scatter(t_vals[valid], depth_1d[valid],
                            c=V[valid], cmap=cmap,
                            vmin=v_min, vmax=v_max,
                            s=15, marker="o", alpha=0.8)
            ax.set_ylim(max_depth, 0)
            ax.set_title(var, fontsize=13, fontweight="bold")
            ax.set_ylabel("Depth (m)", fontsize=11)
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(var, fontsize=11)
        else:
            _draw_pcolormesh(ax, t_vals, depth_vals, V, cmap, var, max_depth)

    _finalise_fig(fig, axes, plot_path)
    ds.close()
    return plot_path


# ----------------------------------------------------------------
# Main entry point — generates both plots
# ----------------------------------------------------------------

def run_step5(grid_path=None, plot_path=None, l1_path=None, l0_path=None):
    print("=" * 60)
    print("  STEP 5: Time-Depth Grid Plots (L0 + L1)")
    print("=" * 60)

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- L0 plot ---
    if l0_path is None:
        l0_path = os.path.join(OUTPUT_DIR,
                               f"incois_glider_{GLIDER_ID}_L0.nc")
    l0_out = plot_l0(l0_path)

    print()

    # --- L1 plot ---
    l1_out = plot_l1(grid_path, plot_path, l1_path)

    print()
    if l0_out:
        print(f"  L0 plot: {l0_out}")
    if l1_out:
        print(f"  L1 plot: {l1_out}")

    # Return L1 path for backward compatibility
    return l1_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid",  default=None)
    p.add_argument("--l0",    default=None)
    p.add_argument("--l1",    default=None)
    p.add_argument("--out",   default=None)
    args = p.parse_args()
    run_step5(grid_path=args.grid, plot_path=args.out,
              l1_path=args.l1, l0_path=args.l0)
