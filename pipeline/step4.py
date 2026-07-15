#!/usr/bin/env python3
"""
step4.py - Profile splitting and 2D grid generation.

Exposes two reusable functions:

  split_profiles(nc_path, out_dir, base_name, apply_qc)
    → writes one NetCDF per profile, returns profile count

  make_grid(nc_path, out_dir, grid_filename, apply_qc)
    → writes 2D time×depth grid NetCDF, returns path

Both can be called with apply_qc=False (L0) or apply_qc=True (L1).
When apply_qc=True, only QC flags 1 & 2 (good / probably good) are
included in the depth-bin average.
"""
import os
import sys
import time
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID, DEPTH_BIN

# Variables excluded from gridding (not continuous physical quantities)
# "depth" MUST be excluded — it's used as the grid coordinate and must stay 1D
_SKIP_GRID_VARS = {
    "profile_index", "profile_direction", "distance_over_ground",
    "mission_number", "profile_time_start", "profile_time_end",
    "depth",
}


def _get_vars_to_grid(ds):
    """Return all numeric time-dimension variables suitable for gridding."""
    result = []
    for var in ds.data_vars:
        if var.endswith("_QC"):
            continue
        if var in _SKIP_GRID_VARS:
            continue
        da = ds[var]
        if da.dims == ("time",) and np.issubdtype(da.dtype, np.floating):
            result.append(var)
    return result


def _qc_mask(ds, var):
    """Return boolean mask: True where data is good (QC 1 or 2)."""
    qc_var = f"{var}_QC"
    if qc_var in ds:
        qc = ds[qc_var].values.astype(int)
        return (qc == 1) | (qc == 2)
    return np.ones(len(ds.time), dtype=bool)


# ── Profile splitting ────────────────────────────────────────────

def split_profiles(nc_path, out_dir, base_name, apply_qc=False):
    """
    Split a timeseries NetCDF into one file per profile.

    Parameters
    ----------
    nc_path   : path to input timeseries NetCDF
    out_dir   : directory where profile NetCDFs are written
    base_name : filename prefix (e.g. "incois_glider_890_2023_L1")
    apply_qc  : if True, mask bad-flagged values in the output profiles

    Returns
    -------
    Number of profiles written.
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    if "profile_index" not in ds:
        print("  WARNING: no profile_index — cannot split profiles")
        ds.close()
        return 0

    pi = ds.profile_index.values
    unique = np.unique(pi[np.isfinite(pi)])
    n = len(unique)

    if apply_qc:
        # Pre-build QC masks
        qc_vars = [v for v in ds.data_vars if v.endswith("_QC")]
        # Mask bad values in a copy — flag 3 and 4 → NaN
        ds_masked = ds.copy(deep=True)
        for qv in qc_vars:
            base_v = qv.replace("_QC", "")
            if base_v in ds_masked:
                qc = ds_masked[qv].values.astype(int)
                bad = (qc == 3) | (qc == 4)
                vals = ds_masked[base_v].values.copy().astype(float)
                vals[bad] = np.nan
                ds_masked[base_v].values = vals
    else:
        ds_masked = ds

    for p_num in unique:
        mask = (ds_masked.profile_index == p_num)
        prof = ds_masked.isel(time=mask)
        prof.attrs["profile_id"] = int(p_num)
        if "profile_direction" in prof:
            d = float(np.nanmean(prof.profile_direction.values))
            prof.attrs["direction"] = ("climb" if d > 0
                                       else "dive" if d < 0 else "unknown")
        out = os.path.join(out_dir, f"{base_name}_profile_{int(p_num):04d}.nc")
        prof.to_netcdf(out, mode="w")

    ds.close()
    if apply_qc:
        ds_masked.close()
    print(f"  Split {n} profiles → {out_dir}")
    return n


# ── Grid generation ─────────────────────────────────────────────

def _interp_profile_to_grid(d_arr, v_arr, depth_centers):
    """
    Interpolate a single profile onto the regular depth grid.

    Uses linear interpolation between measurements — this is correct because
    glider data is essentially a continuous profile sampled every few meters.
    Only interpolates WITHIN the measured depth range (no extrapolation).

    Returns array of length len(depth_centers) with NaN outside data range.
    """
    from scipy.interpolate import interp1d

    # Sort by depth (profiles can be ascending or descending)
    order = np.argsort(d_arr)
    d_sorted = d_arr[order]
    v_sorted = v_arr[order]

    # Remove duplicate depths (average values at same depth)
    _, unique_idx = np.unique(d_sorted, return_index=True)
    if len(unique_idx) < len(d_sorted):
        # There are duplicates — use binned mean at each unique depth
        d_unique = d_sorted[unique_idx]
        v_unique = np.empty(len(d_unique))
        for j, idx in enumerate(unique_idx):
            next_idx = unique_idx[j + 1] if j + 1 < len(unique_idx) else len(d_sorted)
            v_unique[j] = np.nanmean(v_sorted[idx:next_idx])
        d_sorted = d_unique
        v_sorted = v_unique

    if len(d_sorted) < 2:
        # Can't interpolate with fewer than 2 points
        result = np.full(len(depth_centers), np.nan)
        if len(d_sorted) == 1:
            # Place single point in nearest bin
            idx = np.argmin(np.abs(depth_centers - d_sorted[0]))
            result[idx] = v_sorted[0]
        return result

    # Interpolate — only within the measured depth range (bounds_error=False
    # with fill_value=nan means no extrapolation beyond data)
    f = interp1d(d_sorted, v_sorted, kind='linear',
                 bounds_error=False, fill_value=np.nan)
    return f(depth_centers)


def make_grid(nc_path, out_dir, grid_filename, apply_qc=False):
    """
    Interpolate a timeseries NetCDF into a 2D time x depth grid.

    Each profile is linearly interpolated onto a regular 1m depth grid.
    This matches what pyglider does — glider measurements are continuous
    profiles sampled every few meters, so linear interpolation between
    points is physically correct.

    Parameters
    ----------
    nc_path        : path to input timeseries NetCDF
    out_dir        : directory where the grid is written
    grid_filename  : output filename (e.g. "incois_glider_890_2023_L1_grid.nc")
    apply_qc       : if True, only flag 1/2 values go into the grid

    Returns
    -------
    Path to the written grid NetCDF.
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = xr.open_dataset(nc_path)

    if "profile_index" not in ds:
        print(f"  WARNING: no profile_index in {nc_path} — cannot grid")
        ds.close()
        return None

    pi = ds.profile_index.values
    unique = np.unique(pi[np.isfinite(pi)])
    n = len(unique)

    vars_to_grid = _get_vars_to_grid(ds)

    max_depth = float(np.nanmax(ds.depth.values))
    if np.isnan(max_depth) or max_depth < 10:
        max_depth = 1000.0
    depth_centers = np.arange(DEPTH_BIN / 2.0, max_depth + DEPTH_BIN / 2.0, DEPTH_BIN)
    label = "QC flags 1&2" if apply_qc else "all finite values"
    print(f"  Grid: {n} profiles x {len(depth_centers)} depth bins "
          f"(0-{max_depth:.0f} m, dz={DEPTH_BIN}m, {label})")

    # QC masks (built once over full dataset)
    qc_masks = {}
    if apply_qc:
        qc_masks = {var: _qc_mask(ds, var) for var in vars_to_grid}

    gridded  = {var: [] for var in vars_to_grid}
    p_times  = []

    for i, p_num in enumerate(unique):
        mask = (pi == p_num)
        t_arr = ds.time.values[mask].astype("datetime64[s]").astype(float)
        p_times.append(float(np.nanmean(t_arr)) if len(t_arr) > 0 else np.nan)

        d_arr = ds.depth.values[mask]
        prof_idx = np.where(mask)[0]

        for var in vars_to_grid:
            if var not in ds:
                gridded[var].append(np.full(len(depth_centers), np.nan))
                continue
            v_arr = ds[var].values[mask]
            # Skip if not 1D float (e.g. string, structured, or wrong shape)
            if v_arr.ndim != 1 or not np.issubdtype(v_arr.dtype, np.floating):
                gridded[var].append(np.full(len(depth_centers), np.nan))
                continue
            if apply_qc and var in qc_masks:
                qc_ok = qc_masks[var][prof_idx]
            else:
                qc_ok = np.ones(len(v_arr), dtype=bool)
            good = np.isfinite(d_arr) & np.isfinite(v_arr) & qc_ok
            if good.sum() >= 2:
                # Interpolate profile onto regular depth grid
                gridded[var].append(
                    _interp_profile_to_grid(d_arr[good], v_arr[good],
                                            depth_centers))
            elif good.sum() == 1:
                row = np.full(len(depth_centers), np.nan)
                idx = np.argmin(np.abs(depth_centers - d_arr[good][0]))
                row[idx] = v_arr[good][0]
                gridded[var].append(row)
            else:
                gridded[var].append(np.full(len(depth_centers), np.nan))

        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{n} profiles")

    # Assemble grid dataset
    grid_times = np.array(p_times).astype("datetime64[s]")
    gds = xr.Dataset(coords={"time": grid_times, "depth": depth_centers})

    for var in vars_to_grid:
        if var in ds:
            arr = np.vstack(gridded[var])
            gds[var] = xr.DataArray(arr, dims=["time", "depth"],
                                    attrs=ds[var].attrs.copy())

    # Carry all global attributes from source
    gds.attrs = ds.attrs.copy()
    if apply_qc:
        gds.attrs["processing_level"] = (
            ds.attrs.get("processing_level", "") +
            " | 2D Gridded (QC flags 1 & 2 only)")
        gds.attrs["qc_applied"] = "Only ARGO QC flags 1 (good) and 2 (probably good) included in depth bins"
    else:
        gds.attrs["processing_level"] = (
            ds.attrs.get("processing_level", "") +
            " | 2D Gridded (all finite values, no QC)")
    gds.attrs["depth_bin_m"] = str(DEPTH_BIN)
    gds.attrs["n_profiles"]  = str(n)

    out_path = os.path.join(out_dir, grid_filename)
    gds.to_netcdf(out_path, mode="w")
    print(f"  Grid saved: {out_path}")

    ds.close()
    return out_path


# ── Legacy entry point (for backward compatibility) ─────────────

def run_step4(l1_path=None):
    """Backward-compatible wrapper used by old run_pipeline.py."""
    print("=" * 60)
    print("  STEP 4: Profiles + Grid Generation")
    print("=" * 60)
    t0 = time.time()

    if l1_path is None:
        l1_path = os.path.join(OUTPUT_DIR, "l1",
                               f"incois_glider_{GLIDER_ID}_L1.nc")
    if not os.path.exists(l1_path):
        print(f"ERROR: L1 file not found: {l1_path}")
        sys.exit(1)

    profiles_dir = os.path.join(OUTPUT_DIR, "profiles")
    grid_dir     = os.path.join(OUTPUT_DIR, "gridfiles")
    base         = f"incois_glider_{GLIDER_ID}"

    split_profiles(l1_path, profiles_dir, base, apply_qc=True)
    grid_out = make_grid(l1_path, grid_dir, f"{base}_grid.nc", apply_qc=True)

    print(f"\n  STEP 4 COMPLETE in {time.time() - t0:.1f}s")
    return grid_out


if __name__ == "__main__":
    l1 = sys.argv[1] if len(sys.argv) > 1 else None
    run_step4(l1)
