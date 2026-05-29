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
from scipy.stats import binned_statistic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID, PLOT_DEPTH_MAX, DEPTH_BIN

# Gaps longer than this (hours) are masked in pcolormesh
GAP_THRESHOLD_HOURS = 48

VARS_TO_PLOT = ["potential_temperature", "salinity",
                "oxygen_concentration", "chlorophyll", "cdom"]
CMAPS        = ["RdYlBu_r", "viridis", "viridis", "viridis", "viridis"]


# ----------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------

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


def _max_data_depth(ds, vars_list, depth_vals):
    """Find the deepest depth bin that has any valid data."""
    max_d = 0.0
    for var in vars_list:
        if var in ds:
            v = ds[var].values
            if v.ndim == 2:
                col_valid = np.any(np.isfinite(v), axis=0)
                if np.any(col_valid):
                    max_d = max(max_d, float(depth_vals[col_valid].max()))
    return max_d if max_d > 10 else float(depth_vals.max())


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
    depth_mask = depth_vals <= max_depth
    V_trim = V[:, depth_mask]
    d_trim = depth_vals[depth_mask]
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
    ax.set_ylim(max_depth, 0)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=11)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=11)
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

def _make_l0_grid(l0_ds, depth_bin=None):
    """
    Build a quick 2D grid from the L0 timeseries (all finite values, no QC).
    Returns (grid_data dict, time_arr, depth_centers).
    """
    if depth_bin is None:
        depth_bin = DEPTH_BIN

    if "profile_index" not in l0_ds:
        return None, None, None

    pi_vals = l0_ds["profile_index"].values
    unique_profiles = np.unique(pi_vals[np.isfinite(pi_vals)])

    depth_raw = l0_ds["depth"].values if "depth" in l0_ds else None
    if depth_raw is None:
        return None, None, None

    max_depth = float(np.nanmax(depth_raw))
    if np.isnan(max_depth) or max_depth < 10:
        max_depth = PLOT_DEPTH_MAX or 1000.0
    depth_bins    = np.arange(0, max_depth + depth_bin, depth_bin)
    depth_centers = depth_bins[:-1] + depth_bin / 2.0

    gridded = {var: [] for var in VARS_TO_PLOT}
    times   = []

    for p_num in unique_profiles:
        mask   = (pi_vals == p_num)
        d_vals = depth_raw[mask]
        t_vals = l0_ds.time.values[mask].astype("datetime64[s]").astype(float)
        times.append(float(np.nanmean(t_vals)) if len(t_vals) > 0 else np.nan)

        for var in VARS_TO_PLOT:
            if var in l0_ds:
                v_vals = l0_ds[var].values[mask]
                valid  = np.isfinite(d_vals) & np.isfinite(v_vals)
                if np.sum(valid) > 0:
                    stat, _, _ = binned_statistic(
                        d_vals[valid], v_vals[valid],
                        statistic="mean", bins=depth_bins)
                    gridded[var].append(stat)
                else:
                    gridded[var].append(np.full(len(depth_centers), np.nan))
            else:
                gridded[var].append(np.full(len(depth_centers), np.nan))

    time_arr = np.array(times).astype("datetime64[s]")
    grid_2d  = {}
    for var in VARS_TO_PLOT:
        if var in l0_ds and len(gridded[var]) > 0:
            grid_2d[var] = np.vstack(gridded[var])

    return grid_2d, time_arr, depth_centers


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

    max_depth = 0.0
    for var in VARS_TO_PLOT:
        if var in grid_2d:
            v = grid_2d[var]
            col_valid = np.any(np.isfinite(v), axis=0)
            if np.any(col_valid):
                max_depth = max(max_depth, float(depth_centers[col_valid].max()))
    if max_depth < 10:
        max_depth = PLOT_DEPTH_MAX or 1000.0

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

    if grid_path is None:
        grid_path = os.path.join(OUTPUT_DIR, "gridfiles",
                                 f"incois_glider_{GLIDER_ID}_grid.nc")

    if os.path.exists(grid_path):
        print(f"  Loading grid: {grid_path}")
        ds = xr.open_dataset(grid_path)
        is_1d = False
    elif l1_path is not None and os.path.exists(l1_path):
        print(f"  Loading L1 (grid not found): {l1_path}")
        ds = xr.open_dataset(l1_path)
        is_1d = ("depth" in ds and ds["depth"].ndim == 1
                 and ds.depth.dims[0] == "time")
    else:
        print(f"  WARNING: no grid or L1 file found for L1 plot")
        return None

    if "potential_density" not in ds and "density" in ds:
        ds["potential_density"] = ds["density"]

    ds = ds.sortby("time")

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
