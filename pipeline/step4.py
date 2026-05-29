#!/usr/bin/env python3
"""
step4.py - Profile splitting and 2D grid generation from L1 timeseries.
"""
import os
import sys
import time
import numpy as np
import xarray as xr
from scipy.stats import binned_statistic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID, DEPTH_BIN


def run_step4(l1_path=None):
    print("=" * 60)
    print("  STEP 4: Profiles + Grid Generation")
    print("=" * 60)
    t0 = time.time()

    if l1_path is None:
        l1_path = os.path.join(OUTPUT_DIR, "l1", f"incois_glider_{GLIDER_ID}_L1.nc")

    if not os.path.exists(l1_path):
        print(f"ERROR: L1 file not found: {l1_path}")
        sys.exit(1)

    print(f"  Loading L1: {l1_path}")
    ds = xr.open_dataset(l1_path)

    if "profile_index" not in ds:
        print("ERROR: profile_index not found in L1 file")
        sys.exit(1)

    p_indices = ds.profile_index.values
    unique_profiles = np.unique(p_indices[np.isfinite(p_indices)])
    n_profiles = len(unique_profiles)
    print(f"  Profiles: {n_profiles}")

    profiles_dir = os.path.join(OUTPUT_DIR, "profiles")
    grid_dir = os.path.join(OUTPUT_DIR, "gridfiles")
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    vars_to_grid = [
        "temperature", "salinity", "oxygen_concentration", "oxygen_concentration_lag_corrected",
        "chlorophyll", "cdom", "backscatter_700", "density", "potential_temperature",
        "potential_density", "latitude", "longitude",
    ]

    gridded_data = {var: [] for var in vars_to_grid}
    profile_times = []

    max_depth = float(np.nanmax(ds.depth.values))
    if np.isnan(max_depth) or max_depth < 10:
        max_depth = 1000.0
    depth_bins = np.arange(0, max_depth + DEPTH_BIN, DEPTH_BIN)
    depth_centers = depth_bins[:-1] + DEPTH_BIN / 2.0
    print(f"  Grid depth bins: 0 to {max_depth:.1f} m (dz={DEPTH_BIN}m)")

    base_name = f"incois_glider_{GLIDER_ID}"

    # Build a per-variable QC mask over the full dataset once.
    # Values with QC flag 3 (probably bad) or 4 (bad) are excluded from gridding.
    # Flag 1 (good) and 2 (probably good) are kept.
    # If no QC variable exists for a given variable, all finite values are used.
    def _qc_good_mask(ds, var):
        qc_var = f"{var}_QC"
        if qc_var in ds:
            qc = ds[qc_var].values.astype(int)
            return (qc == 1) | (qc == 2)
        return np.ones(len(ds.time), dtype=bool)

    qc_masks = {var: _qc_good_mask(ds, var) for var in vars_to_grid if var in ds}

    for i, p_num in enumerate(unique_profiles):
        prof_mask = (ds.profile_index == p_num)
        prof_ds = ds.isel(time=prof_mask)

        prof_out = os.path.join(profiles_dir, f"{base_name}_profile_{int(p_num):04d}.nc")
        prof_ds.attrs["profile_id"] = int(p_num)
        if "profile_direction" in prof_ds:
            dir_val = float(np.nanmean(prof_ds.profile_direction.values))
            prof_ds.attrs["direction"] = "climb" if dir_val > 0 else "dive" if dir_val < 0 else "unknown"
        prof_ds.to_netcdf(prof_out, mode="w")

        t_vals = prof_ds.time.values.astype("datetime64[s]").astype(float)
        p_time = float(np.nanmean(t_vals)) if len(t_vals) > 0 else float("nan")
        profile_times.append(p_time)

        d_vals = prof_ds.depth.values
        # Profile-level QC mask (slice of the full-dataset mask)
        prof_indices = np.where(prof_mask.values)[0]

        for var in vars_to_grid:
            if var in prof_ds:
                v_vals = prof_ds[var].values
                # Apply QC: only use good/probably-good flagged points
                if var in qc_masks:
                    qc_ok = qc_masks[var][prof_indices]
                else:
                    qc_ok = np.ones(len(v_vals), dtype=bool)
                valid = np.isfinite(d_vals) & np.isfinite(v_vals) & qc_ok
                if np.sum(valid) > 0:
                    stat, _, _ = binned_statistic(d_vals[valid], v_vals[valid], statistic="mean", bins=depth_bins)
                    gridded_data[var].append(stat)
                else:
                    gridded_data[var].append(np.full(len(depth_centers), np.nan))
            else:
                gridded_data[var].append(np.full(len(depth_centers), np.nan))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_profiles} profiles...")

    print(f"  Created {n_profiles} profile files in {profiles_dir}")

    print("  Assembling 2D Grid NetCDF...")
    grid_times = np.array(profile_times).astype("datetime64[s]")
    grid_ds = xr.Dataset(coords={"time": grid_times, "depth": depth_centers})

    for var in vars_to_grid:
        if var in ds:
            grid_data_array = np.vstack(gridded_data[var])
            attrs = ds[var].attrs.copy()
            grid_ds[var] = xr.DataArray(grid_data_array, dims=["time", "depth"], attrs=attrs)

    grid_ds.attrs = ds.attrs.copy()
    grid_ds.attrs["processing_level"] = ds.attrs.get("processing_level", "") + " | 2D Gridded (QC flags 1&2 only)"
    grid_ds.attrs["note"] = f"Binned to {DEPTH_BIN}m depth bins, 1 profile per time step. Only QC flag 1 (good) and 2 (probably good) values included."

    grid_out = os.path.join(grid_dir, f"{base_name}_grid.nc")
    grid_ds.to_netcdf(grid_out, mode="w")
    print(f"  Grid saved: {grid_out}")

    ds.close()
    print(f"\n  STEP 4 COMPLETE in {time.time() - t0:.1f}s")
    return grid_out


if __name__ == "__main__":
    l1 = sys.argv[1] if len(sys.argv) > 1 else None
    run_step4(l1)
