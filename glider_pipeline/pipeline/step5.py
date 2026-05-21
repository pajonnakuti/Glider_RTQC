#!/usr/bin/env python3
"""
step5.py - Time-vs-depth scatter plots from gridded L1 data.

Uses scatter plotting instead of pcolormesh to handle irregular grids
and large time gaps that cause blank images with contour methods.
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
from config import OUTPUT_DIR, GLIDER_ID, PLOT_DEPTH_MAX

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def run_step5(grid_path=None, plot_path=None):
    print("=" * 60)
    print("  STEP 5: Grid Plots")
    print("=" * 60)

    if grid_path is None:
        grid_path = os.path.join(OUTPUT_DIR, "gridfiles", f"incois_glider_{GLIDER_ID}_grid.nc")

    if not os.path.exists(grid_path):
        print(f"ERROR: Grid file not found: {grid_path}")
        sys.exit(1)

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"incois_glider_{GLIDER_ID}_gridplot.png")

    print(f"  Loading grid: {grid_path}")
    ds = xr.open_dataset(grid_path)

    if "potential_density" not in ds and "density" in ds:
        ds["potential_density"] = ds["density"]

    if HAS_PANDAS and len(ds.time) > 0:
        t_dt = pd.Series(ds.time.values)
        mode_year = t_dt.dt.year.mode()[0]
        ds = ds.sel(time=ds.time.dt.year == mode_year)

    ds = ds.sortby("time")

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    vars_to_plot = ["potential_temperature", "salinity", "oxygen_concentration", "chlorophyll", "cdom"]
    cmaps = ["RdYlBu_r", "viridis", "viridis", "viridis", "viridis"]

    T, D = np.meshgrid(ds.time.values, ds.depth.values, indexing="ij")

    max_depth = float(ds.depth.max().values)
    if np.isnan(max_depth) or max_depth < 10:
        max_depth = PLOT_DEPTH_MAX if PLOT_DEPTH_MAX else 150.0

    for i, var in enumerate(vars_to_plot):
        if var in ds:
            V = ds[var].values
            valid = np.isfinite(V) & np.isfinite(T) & np.isfinite(D)
            if np.sum(valid) == 0:
                axes[i].set_title(f"{var} (NO DATA)")
                axes[i].set_ylim(max_depth, 0)
                continue
            v_min, v_max = np.nanpercentile(V[valid], 1), np.nanpercentile(V[valid], 99)
            sc = axes[i].scatter(T[valid], D[valid], c=V[valid], cmap=cmaps[i],
                                 vmin=v_min, vmax=v_max, s=5, marker="s")
            axes[i].set_ylim(max_depth, 0)
            axes[i].set_title(var, fontsize=14, fontweight="bold")
            axes[i].set_ylabel("Depth (m)", fontsize=12)
            cbar = plt.colorbar(sc, ax=axes[i], pad=0.02)
            cbar.set_label(var, fontsize=12)
        else:
            axes[i].set_title(f"{var} (NOT IN DATASET)")
            axes[i].set_ylim(max_depth, 0)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path}")

    ds.close()
    plt.close()
    return plot_path


if __name__ == "__main__":
    grid = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_step5(grid, out)
