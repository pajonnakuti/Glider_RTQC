#!/usr/bin/env python3
"""
step6.py - Deployment summary, track map, and diagnostic plots.

Produces:
  plots/incois_glider_{ID}_track.png          — GPS track on world map
  plots/incois_glider_{ID}_ts_diagram.png     — T-S scatter coloured by depth
  plots/incois_glider_{ID}_data_coverage.png  — sensor availability matrix
  reports/incois_glider_{ID}_summary.txt      — deployment summary report
"""
import os
import sys
import time
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False


# ============================================================
# Track map
# ============================================================

def plot_track(l1_path, l0_path=None, plot_path=None):
    """
    Plot the glider GPS track on a 2D map.
    Uses cartopy if available, falls back to plain matplotlib.
    """
    print("  Generating track map...")

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_track.png")

    ds = xr.open_dataset(l1_path)

    lat = ds["latitude"].values  if "latitude"  in ds else None
    lon = ds["longitude"].values if "longitude" in ds else None
    t   = ds["time"].values

    if lat is None or lon is None:
        print("  WARNING: no lat/lon in L1 — skipping track map")
        ds.close()
        return None

    valid = np.isfinite(lat) & np.isfinite(lon)
    lat_v = lat[valid]
    lon_v = lon[valid]
    t_v   = t[valid]

    if len(lat_v) < 2:
        print("  WARNING: insufficient GPS data for track map")
        ds.close()
        return None

    # Pad the map extent
    lat_min, lat_max = np.nanmin(lat_v), np.nanmax(lat_v)
    lon_min, lon_max = np.nanmin(lon_v), np.nanmax(lon_v)
    pad_lat = max(2.0, (lat_max - lat_min) * 0.2)
    pad_lon = max(2.0, (lon_max - lon_min) * 0.2)
    extent = [lon_min - pad_lon, lon_max + pad_lon,
              lat_min - pad_lat, lat_max + pad_lat]

    # Colour track by time
    t_num = t_v.astype("datetime64[s]").astype(float)
    t_norm = (t_num - t_num.min()) / max(t_num.max() - t_num.min(), 1)

    fig = plt.figure(figsize=(12, 8))

    _use_cartopy = HAS_CARTOPY
    if _use_cartopy:
        try:
            ax = fig.add_subplot(1, 1, 1,
                                 projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND,       facecolor="lightgray",  zorder=1)
            ax.add_feature(cfeature.OCEAN,      facecolor="lightblue",  zorder=0)
            ax.add_feature(cfeature.COASTLINE,  linewidth=0.8,          zorder=2)
            ax.add_feature(cfeature.BORDERS,    linewidth=0.4,
                           linestyle=":",       zorder=2)
            ax.add_feature(cfeature.RIVERS,     linewidth=0.3,
                           edgecolor="blue",    zorder=2)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                              color="gray", alpha=0.5)
            gl.top_labels   = False
            gl.right_labels = False
            sc = ax.scatter(lon_v, lat_v, c=t_norm, cmap="plasma",
                            s=6, transform=ccrs.PlateCarree(),
                            zorder=3, alpha=0.8)
            # Mark start and end
            ax.plot(lon_v[0],  lat_v[0],  "g^", ms=10,
                    transform=ccrs.PlateCarree(), zorder=5, label="Start")
            ax.plot(lon_v[-1], lat_v[-1], "rs", ms=10,
                    transform=ccrs.PlateCarree(), zorder=5, label="End")
        except Exception as e:
            # Cartopy failed (e.g. shapefile download blocked, HTTP 403)
            print(f"  WARNING: cartopy basemap unavailable ({e}), "
                  f"falling back to plain scatter")
            plt.close(fig)
            fig = plt.figure(figsize=(12, 8))
            _use_cartopy = False

    if not _use_cartopy:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor("lightblue")
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        sc = ax.scatter(lon_v, lat_v, c=t_norm, cmap="plasma",
                        s=6, alpha=0.8)
        ax.plot(lon_v[0],  lat_v[0],  "g^", ms=10, label="Start")
        ax.plot(lon_v[-1], lat_v[-1], "rs", ms=10, label="End")

    # Colourbar — show actual dates at ticks
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Time", fontsize=11)
    n_ticks = 5
    tick_pos = np.linspace(0, 1, n_ticks)
    tick_vals = t_num.min() + tick_pos * (t_num.max() - t_num.min())
    tick_labels = [str(np.datetime64(int(v), "s"))[:10] for v in tick_vals]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(tick_labels, fontsize=9)

    # Deployment distance
    dlat = np.diff(lat_v) * 111.32
    dlon = np.diff(lon_v) * 111.32 * np.cos(np.radians(float(np.mean(lat_v))))
    total_dist = float(np.sum(np.sqrt(dlat**2 + dlon**2)))

    t0_str = str(t_v[0])[:10]
    t1_str = str(t_v[-1])[:10]
    duration_days = (t_v[-1].astype("datetime64[D]") -
                     t_v[0].astype("datetime64[D]")).astype(int)

    title = (f"Glider {GLIDER_ID}  —  GPS Track\n"
             f"{t0_str} → {t1_str}  |  "
             f"{duration_days} days  |  "
             f"{total_dist:.0f} km")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    try:
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Track map saved: {plot_path}")
    except Exception as e:
        # Cartopy render failure (shapefile download at render time)
        if _use_cartopy:
            print(f"  WARNING: cartopy render failed ({e}), "
                  f"retrying without basemap")
            plt.close(fig)
            # Redo as plain scatter
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_facecolor("lightblue")
            ax.set_xlabel("Longitude (°E)", fontsize=12)
            ax.set_ylabel("Latitude (°N)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)
            sc = ax.scatter(lon_v, lat_v, c=t_norm, cmap="plasma",
                            s=6, alpha=0.8)
            ax.plot(lon_v[0],  lat_v[0],  "g^", ms=10, label="Start")
            ax.plot(lon_v[-1], lat_v[-1], "rs", ms=10, label="End")
            cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
            cbar.set_label("Time", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(loc="upper right", fontsize=10)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"  Track map saved (plain fallback): {plot_path}")
        else:
            print(f"  ERROR: track map save failed: {e}")
    plt.close(fig)
    ds.close()
    return plot_path


# ============================================================
# T-S diagram
# ============================================================

def plot_ts_diagram(l1_path, plot_path=None):
    """T-S scatter coloured by depth. Standard QC visualization."""
    print("  Generating T-S diagram...")

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_ts_diagram.png")

    ds = xr.open_dataset(l1_path)

    temp = ds["temperature"].values  if "temperature" in ds else None
    sal  = ds["salinity"].values     if "salinity"    in ds else None
    dep  = ds["depth"].values        if "depth"       in ds else None

    if temp is None or sal is None:
        print("  WARNING: no T/S data — skipping T-S diagram")
        ds.close()
        return None

    # Only use QC-good points
    tqc = ds["temperature_QC"].values.astype(int) if "temperature_QC" in ds else None
    sqc = ds["salinity_QC"].values.astype(int)    if "salinity_QC"    in ds else None
    valid = np.isfinite(temp) & np.isfinite(sal)
    if tqc is not None:
        valid &= (tqc == 1) | (tqc == 2)
    if sqc is not None:
        valid &= (sqc == 1) | (sqc == 2)

    n_valid = int(np.sum(valid))
    if n_valid < 10:
        print("  WARNING: too few valid T-S points — skipping T-S diagram")
        ds.close()
        return None

    T = temp[valid]
    S = sal[valid]
    D = dep[valid] if dep is not None else np.zeros(n_valid)

    # Subsample for large datasets to keep plot fast
    if n_valid > 50000:
        idx = np.random.choice(n_valid, 50000, replace=False)
        T, S, D = T[idx], S[idx], D[idx]

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(S, T, c=D, cmap="viridis_r",
                    s=3, alpha=0.6, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Depth (m)", fontsize=11)

    # Density contours (sigma-theta)
    if HAS_GSW:
        s_range = np.linspace(max(S.min() - 0.2, 0), S.max() + 0.2, 100)
        t_range = np.linspace(T.min() - 0.5, T.max() + 0.5, 100)
        SS, TT = np.meshgrid(s_range, t_range)
        lat_m = float(np.nanmean(ds["latitude"].values)) if "latitude" in ds else 12.0
        lon_m = float(np.nanmean(ds["longitude"].values)) if "longitude" in ds else 75.0
        try:
            SA = gsw.SA_from_SP(SS, 0, lon_m, lat_m)
            CT = gsw.CT_from_t(SA, TT, 0)
            sigma0 = gsw.sigma0(SA, CT)
            contour_levels = np.arange(
                np.floor(sigma0.min()), np.ceil(sigma0.max()) + 0.5, 0.5)
            cs = ax.contour(SS, TT, sigma0, levels=contour_levels,
                            colors="gray", linewidths=0.5, alpha=0.6)
            ax.clabel(cs, fmt="%.1f", fontsize=8)
        except Exception:
            pass

    ax.set_xlabel("Salinity (PSU)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(f"Glider {GLIDER_ID}  —  T-S Diagram  "
                 f"({n_valid:,} QC-good points)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  T-S diagram saved: {plot_path}")
    plt.close(fig)
    ds.close()
    return plot_path


# ============================================================
# Sensor data coverage matrix
# ============================================================

def plot_data_coverage(l0_path, l1_path, plot_path=None):
    """
    Show which sensors were active (data available) across the deployment.
    One row per variable, one column per day. Colour = % valid in that day.
    """
    print("  Generating data coverage matrix...")

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_data_coverage.png")

    # Use L1 for coverage (shows what survived QC)
    ds = xr.open_dataset(l1_path)

    vars_to_show = [
        "temperature", "salinity", "pressure",
        "oxygen_concentration", "chlorophyll", "cdom", "backscatter_700",
    ]
    vars_present = [v for v in vars_to_show if v in ds]
    if not vars_present:
        print("  WARNING: no variables found — skipping coverage matrix")
        ds.close()
        return None

    t = ds.time.values.astype("datetime64[D]")
    days = np.unique(t)
    n_days = len(days)
    n_vars = len(vars_present)

    coverage = np.zeros((n_vars, n_days))

    for j, day in enumerate(days):
        mask = (t == day)
        for i, var in enumerate(vars_present):
            v = ds[var].values[mask]
            # For L1 use QC flag if available
            qc_var = var + "_QC"
            if qc_var in ds:
                qc = ds[qc_var].values[mask].astype(int)
                n_good = int(np.sum((qc == 1) | (qc == 2)))
            else:
                n_good = int(np.sum(np.isfinite(v)))
            n_total = len(v)
            coverage[i, j] = n_good / n_total if n_total > 0 else 0.0

    # --- Plot ---
    fig_width = max(12, n_days // 3)
    fig, ax = plt.subplots(figsize=(min(fig_width, 24), n_vars * 0.8 + 2))

    im = ax.imshow(coverage, aspect="auto", cmap="YlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    # X axis — monthly ticks
    day_nums  = np.arange(n_days)
    day_strs  = [str(d) for d in days]
    # Pick roughly monthly tick positions
    tick_step = max(1, n_days // 12)
    tick_pos  = day_nums[::tick_step]
    tick_lbl  = [day_strs[i][:7] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=9)

    # Y axis
    ax.set_yticks(np.arange(n_vars))
    ax.set_yticklabels(vars_present, fontsize=10)

    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cbar.set_label("Fraction of QC-good data", fontsize=10)

    ax.set_title(f"Glider {GLIDER_ID}  —  Sensor Data Coverage (L1 QC-good)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Coverage matrix saved: {plot_path}")
    plt.close(fig)
    ds.close()
    return plot_path


# ============================================================
# Profile count summary
# ============================================================

def _count_profiles(path):
    """Return (n_profiles, profile_list) from a NetCDF timeseries."""
    if not os.path.exists(path):
        return 0, []
    ds = xr.open_dataset(path)
    if "profile_index" not in ds:
        ds.close()
        return 0, []
    pi = ds["profile_index"].values
    profiles = list(np.unique(pi[np.isfinite(pi)]).astype(int))
    ds.close()
    return len(profiles), profiles


# ============================================================
# Deployment summary report
# ============================================================

def write_summary_report(l0_path, l1_path, grid_path, report_path=None):
    """
    Write a plain-text deployment summary report covering:
    - Deployment dates, duration, distance
    - Profile counts (L0 vs L1, how many survived QC)
    - Variable quality summary
    - Data gaps
    - Issues from verify
    """
    print("  Writing deployment summary report...")

    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if report_path is None:
        report_path = os.path.join(reports_dir,
                                   f"incois_glider_{GLIDER_ID}_summary.txt")

    lines = []
    def w(s=""):
        lines.append(s)

    w("=" * 70)
    w(f"  DEPLOYMENT SUMMARY — Glider {GLIDER_ID}")
    w(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    w("=" * 70)

    # ---- L0 stats ----
    w()
    w("  L0 PRODUCT")
    w("  " + "-" * 50)
    if l0_path and os.path.exists(l0_path):
        ds0 = xr.open_dataset(l0_path)
        n0 = len(ds0.time)
        t0_start = str(ds0.time.values[0])[:19]
        t0_end   = str(ds0.time.values[-1])[:19]
        dur_days = float((ds0.time.values[-1].astype("datetime64[D]") -
                          ds0.time.values[0].astype("datetime64[D]")).astype(int))
        n_prof_l0, _ = _count_profiles(l0_path)
        w(f"  File:         {os.path.basename(l0_path)}")
        w(f"  Size:         {os.path.getsize(l0_path)/1024/1024:.1f} MB")
        w(f"  Time range:   {t0_start}  →  {t0_end}")
        w(f"  Duration:     {dur_days:.0f} days")
        w(f"  Observations: {n0:,}")
        w(f"  Profiles:     {n_prof_l0}")
        if "depth" in ds0:
            max_d = float(np.nanmax(ds0["depth"].values))
            w(f"  Max depth:    {max_d:.1f} m")

        # GPS track distance
        if "latitude" in ds0 and "longitude" in ds0:
            lat = ds0["latitude"].values
            lon = ds0["longitude"].values
            valid = np.isfinite(lat) & np.isfinite(lon)
            if np.sum(valid) > 1:
                dlat = np.diff(lat[valid]) * 111.32
                dlon = (np.diff(lon[valid]) * 111.32
                        * np.cos(np.radians(float(np.nanmean(lat[valid])))))
                dist = float(np.sum(np.sqrt(dlat**2 + dlon**2)))
                w(f"  Track dist:   {dist:.0f} km")

        # Variable coverage in L0
        w()
        w("  L0 variable coverage:")
        for var in ["temperature", "salinity", "pressure",
                    "oxygen_concentration", "chlorophyll",
                    "cdom", "backscatter_700"]:
            if var in ds0:
                v = ds0[var].values
                n_v = int(np.sum(np.isfinite(v)))
                pct = 100 * n_v / n0 if n0 > 0 else 0
                w(f"    {var:22s}: {n_v:8,} / {n0:,}  ({pct:5.1f}%)")
        ds0.close()
    else:
        w(f"  L0 file not found: {l0_path}")

    # ---- L1 stats ----
    w()
    w("  L1 PRODUCT")
    w("  " + "-" * 50)
    if l1_path and os.path.exists(l1_path):
        ds1 = xr.open_dataset(l1_path)
        n1 = len(ds1.time)
        t1_start = str(ds1.time.values[0])[:19]
        t1_end   = str(ds1.time.values[-1])[:19]
        n_prof_l1, _ = _count_profiles(l1_path)
        w(f"  File:         {os.path.basename(l1_path)}")
        w(f"  Size:         {os.path.getsize(l1_path)/1024/1024:.1f} MB")
        w(f"  Time range:   {t1_start}  →  {t1_end}")
        w(f"  Observations: {n1:,}")
        w(f"  Profiles:     {n_prof_l1}")

        # Profile retention
        n_prof_l0_ref = n_prof_l0 if (l0_path and os.path.exists(l0_path)) else n_prof_l1
        if n_prof_l0_ref > 0:
            pct_prof = 100 * n_prof_l1 / n_prof_l0_ref
            w(f"  Profile retention: {n_prof_l1}/{n_prof_l0_ref} ({pct_prof:.0f}% of L0 profiles)")

        # QC flag summary
        w()
        w("  L1 QC flag summary (per variable):")
        w(f"  {'Variable':22s}  {'Good%':>6}  {'PBad%':>6}  {'Bad%':>6}  {'Miss%':>6}")
        w("  " + "-" * 52)
        qc_vars = sorted([v for v in ds1.data_vars if v.endswith("_QC")])
        for qv in qc_vars:
            vname = qv.replace("_QC", "")
            qc = ds1[qv].values.astype(int)
            g  = 100 * np.sum(qc == 1) / n1
            pb = 100 * np.sum(qc == 3) / n1
            b  = 100 * np.sum(qc == 4) / n1
            ms = 100 * np.sum(qc == 9) / n1
            w(f"  {vname:22s}  {g:6.1f}  {pb:6.1f}  {b:6.1f}  {ms:6.1f}")

        ds1.close()
    else:
        w(f"  L1 file not found: {l1_path}")

    # ---- Grid stats ----
    if grid_path and os.path.exists(grid_path):
        w()
        w("  GRID PRODUCT")
        w("  " + "-" * 50)
        g = xr.open_dataset(grid_path)
        w(f"  File:    {os.path.basename(grid_path)}")
        w(f"  Size:    {os.path.getsize(grid_path)/1024/1024:.1f} MB")
        w(f"  Dims:    {len(g.time)} profiles × {len(g.depth)} depth bins")
        w(f"  Depth:   {float(g.depth.min()):.0f} – {float(g.depth.max()):.0f} m")
        g.close()

    # ---- Data gaps ----
    if l1_path and os.path.exists(l1_path):
        ds1 = xr.open_dataset(l1_path)
        t_vals = ds1.time.values
        if len(t_vals) > 1:
            dt_h = np.diff(t_vals.astype("datetime64[h]").astype(float))
            gaps = np.where(dt_h > 48)[0]
            if len(gaps):
                w()
                w(f"  DATA GAPS  (> 48 h)")
                w("  " + "-" * 50)
                for gi in gaps:
                    w(f"  {str(t_vals[gi])[:19]}  →  "
                      f"{str(t_vals[gi+1])[:19]}  "
                      f"({dt_h[gi]:.0f} h / {dt_h[gi]/24:.1f} days)")
        ds1.close()

    w()
    w("=" * 70)
    w(f"  Pipeline outputs in: {OUTPUT_DIR}")
    w("=" * 70)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Summary report saved: {report_path}")

    # Also print to console
    for line in lines:
        print(line)

    return report_path


# ============================================================
# Mixed Layer Depth plot  (uses GliderTools if available)
# ============================================================

def plot_mld(l1_path, grid_path=None, plot_path=None):
    """
    Plot mixed layer depth timeseries using density threshold method.
    Uses GliderTools gt.physics.mixed_layer_depth if available,
    otherwise computes it directly from the gridded temperature.
    """
    print("  Generating MLD plot...")

    if plot_path is None:
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir,
                                 f"incois_glider_{GLIDER_ID}_mld.png")

    # Prefer gridded data
    src_path = grid_path if (grid_path and os.path.exists(grid_path or "")) else l1_path
    if not src_path or not os.path.exists(src_path):
        print("  WARNING: no data for MLD — skipping")
        return None

    ds = xr.open_dataset(src_path)
    is_grid = "depth" in ds.dims

    if not is_grid:
        # timeseries — need gridded for MLD
        ds.close()
        print("  WARNING: MLD requires gridded data — skipping")
        return None

    # --- Try GliderTools MLD ---
    mld_times = None
    mld_vals  = None

    try:
        import glidertools as gt
        import warnings
        warnings.filterwarnings("ignore")
        # GT needs xr.Dataset with specific variable names
        # Map our names → GT expects 'temperature', 'salinity', 'depth'
        if "potential_temperature" in ds and "salinity" in ds:
            # Build a minimal dataset GT can use
            gt_ds = xr.Dataset({
                "temperature": ds["potential_temperature"],
                "salinity":    ds["salinity"],
            }, coords={"depth": ds.depth, "time": ds.time})
            result = gt.physics.mixed_layer_depth(
                gt_ds, variable="temperature",
                thresh=0.2, ref_depth=10, verbose=False)
            if result is not None:
                if hasattr(result, "values"):
                    mld_vals = result.values
                else:
                    mld_vals = np.array(result)
                mld_times = ds.time.values
                print(f"    MLD via GliderTools: {len(mld_vals)} profiles")
    except Exception as e:
        print(f"    GliderTools MLD failed ({e}) — using manual calculation")

    # --- Manual fallback: temperature threshold 0.2°C from 10m ---
    if mld_vals is None and "potential_temperature" in ds:
        T_arr = ds["potential_temperature"].values
        D     = ds.depth.values
        # Guard: must be 2D (n_profiles × n_depth)
        if T_arr.ndim != 2 or T_arr.shape[1] != len(D):
            print("  WARNING: temperature not 2D — skipping MLD")
        else:
            ref_idx = np.argmin(np.abs(D - 10))
            mld_vals = np.full(T_arr.shape[0], np.nan)
            for i in range(T_arr.shape[0]):
                prof = T_arr[i, :]
                valid = np.isfinite(prof)
                if valid.sum() < 5:
                    continue
                ref_T = (prof[ref_idx] if np.isfinite(prof[ref_idx])
                         else np.nanmean(prof[:ref_idx + 2]))
                if not np.isfinite(ref_T):
                    continue
                # Depths in this profile that are > 10m and show T change > 0.2°C
                below = np.where((D > 10) & valid & (np.abs(prof - ref_T) > 0.2))[0]
                if len(below) > 0:
                    mld_vals[i] = float(D[below[0]])
                else:
                    # No threshold crossing — MLD = deepest valid point
                    valid_depths = D[valid]
                    mld_vals[i] = float(valid_depths[-1]) if len(valid_depths) > 0 else np.nan
            mld_times = ds.time.values
            print(f"    MLD via manual threshold: {int(np.sum(np.isfinite(mld_vals)))} profiles")

    if mld_vals is None or mld_times is None:
        ds.close()
        print("  WARNING: could not compute MLD")
        return None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 4))

    valid = np.isfinite(mld_vals)
    ax.fill_between(mld_times[valid], mld_vals[valid], alpha=0.3,
                    color="steelblue", label="MLD")
    ax.plot(mld_times[valid], mld_vals[valid],
            color="steelblue", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_ylabel("Mixed Layer Depth (m)", fontsize=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(f"Glider {GLIDER_ID}  —  Mixed Layer Depth (ΔT=0.2°C threshold)",
                 fontsize=12, fontweight="bold")

    mean_mld = float(np.nanmean(mld_vals))
    ax.axhline(mean_mld, color="red", linestyle="--",
               linewidth=1, label=f"Mean MLD = {mean_mld:.0f} m")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  MLD plot saved: {plot_path}")
    plt.close(fig)
    ds.close()
    return plot_path


# ============================================================
# GliderTools QC comparison report
# ============================================================

def run_gt_comparison(l0_path, l1_path, report_path=None):
    """
    Compare our L1 QC output against GliderTools QC applied to the
    same L0 data. Writes a comparison table to reports/.
    """
    print("  Running GliderTools comparison...")

    try:
        import glidertools as gt
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        print("  GliderTools not installed — skipping comparison")
        return None

    if not l0_path or not os.path.exists(l0_path):
        print("  No L0 file — skipping GT comparison")
        return None
    if not l1_path or not os.path.exists(l1_path):
        print("  No L1 file — skipping GT comparison")
        return None

    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    if report_path is None:
        report_path = os.path.join(reports_dir,
                                   f"incois_glider_{GLIDER_ID}_gt_comparison.txt")

    l0 = xr.open_dataset(l0_path)
    l1 = xr.open_dataset(l1_path)
    n  = len(l0.time)

    lines = []
    def w(s=""): lines.append(s)

    w("=" * 65)
    w(f"  OUR PIPELINE vs GLIDERTOOLS  —  Glider {GLIDER_ID}")
    w(f"  GliderTools {gt.__version__}")
    w(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    w("=" * 65)
    w(f"  L0: {n:,} observations  |  L1: {len(l1.time):,} observations")
    w()

    hdr = f"  {'Metric':42s}  {'GliderTools':>14}  {'Our L1':>12}"
    sep = f"  {'-'*42}  {'-'*14}  {'-'*12}"

    def r(label, gv, ov):
        if isinstance(gv, (int, np.integer)):
            gs = f"{gv:>14,}"
        elif isinstance(gv, float):
            gs = f"{gv:>14.4f}"
        else:
            gs = f"{str(gv):>14}"
        if isinstance(ov, (int, np.integer)):
            os_ = f"{ov:>12,}"
        elif isinstance(ov, float):
            os_ = f"{ov:>12.4f}"
        else:
            os_ = f"{str(ov):>12}"
        w(f"  {label:42s}  {gs}  {os_}")

    for var_name, qc_name in [
        ("temperature",         "temperature_QC"),
        ("salinity",            "salinity_QC"),
        ("oxygen_concentration","oxygen_concentration_QC"),
    ]:
        if var_name not in l0 or var_name not in l1:
            continue
        w("-" * 65)
        w(f"  {var_name.upper()}")
        w("-" * 65)
        w(hdr); w(sep)

        raw = l0[var_name].values.copy().astype(float)

        # GT QC: IQR + despike
        try:
            gt_v = gt.cleaning.outlier_bounds_iqr(raw, multiplier=3.0)
            gt_v, _ = gt.cleaning.despike(gt_v, window_size=5,
                                           spike_method="median")
            gt_n = int(np.sum(np.isfinite(gt_v)))
        except Exception as e:
            gt_n = f"error: {str(e)[:25]}"
            gt_v = np.full_like(raw, np.nan)

        our_qc = l1[qc_name].values.astype(int) if qc_name in l1 else None
        our_n  = int(np.sum(our_qc == 1)) if our_qc is not None else 0
        our_v  = l1[var_name].values

        r("Valid points after QC",    gt_n if isinstance(gt_n,int) else gt_n,  our_n)
        r("% of L0 retained",
          f"{100*gt_n/n:.1f}%" if isinstance(gt_n,int) else "?",
          f"{100*our_n/n:.1f}%")

        if isinstance(gt_n, int) and np.any(np.isfinite(gt_v)):
            r("Min value", float(np.nanmin(gt_v)), float(np.nanmin(our_v[our_qc==1]) if our_n>0 else np.nan))
            r("Max value", float(np.nanmax(gt_v)), float(np.nanmax(our_v[our_qc==1]) if our_n>0 else np.nan))

            # Agreement at common good points
            n_min = min(len(gt_v), len(our_v))
            common = np.isfinite(gt_v[:n_min]) & ((our_qc[:n_min] == 1) if our_qc is not None else True)
            if np.sum(common) > 1000:
                d = gt_v[:n_min][common] - our_v[:n_min][common]
                r("Mean diff GT−Ours", float(np.mean(d)), 0.0)
                r("Std  diff GT−Ours", float(np.std(d)),  0.0)
                thresh = 0.01 if "temp" in var_name or "sal" in var_name else 1.0
                agree  = 100 * float(np.mean(np.abs(d) < thresh))
                w(f"  Agreement within {thresh}: {agree:.1f}% of co-located good points")
        w()

    # Feature comparison table
    w("=" * 65)
    w("  FEATURE COMPARISON")
    w("=" * 65)
    w(f"  {'Feature':46s}  GT   Ours")
    w(f"  {'-'*46}  ---  ----")
    features = [
        ("IQR outlier removal",                      True,  True),
        ("Median despike",                            True,  True),
        ("Savitzky-Golay smoothing",                  True,  True),
        ("Horizontal diff filter",                    True,  True),
        ("Optics dark-count + quenching correction",  True,  True),
        ("ARGO RTQC flag tests 5-16",                 False, True),
        ("QC flags (1/2/3/4/9) in output",            False, True),
        ("Oxygen lag correction (tau=30 s)",           False, True),
        ("Density inversion test (ARGO #14)",          False, True),
        ("Stuck value detection (ARGO #13)",           False, True),
        ("Variable corruption auto-detection",         False, True),
        ("Per-profile NetCDF (NGDAC format)",          False, True),
        ("GPS track map",                              False, True),
        ("Mixed layer depth",                          True,  True),
        ("Bottle calibration support",                 True,  False),
        ("Thermal lag correction",                     True,  False),
        ("Brunt-Väisälä frequency",                    True,  False),
    ]
    for feat, gt_has, our_has in features:
        g = "✓" if gt_has  else "·"
        o = "✓" if our_has else "·"
        w(f"  {feat:46s}  {g:>3}  {o:>4}")

    w()
    w("=" * 65)
    w("  NOTE: GliderTools global IQR can incorrectly remove warm surface")
    w("  water when deep cold water dominates the distribution.")
    w("  Our per-profile ARGO approach avoids this issue.")
    w("=" * 65)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # Print key lines to console
    print(f"  GT comparison saved: {report_path}")
    for line in lines[5:20]:
        print(line)

    l0.close()
    l1.close()
    return report_path


# ============================================================
# Main entry point
# ============================================================

def run_step6(l0_path=None, l1_path=None, grid_path=None):
    print("=" * 60)
    print("  STEP 6: Summary, Track Map & Diagnostic Plots")
    print("=" * 60)
    t0 = time.time()

    plots_dir   = os.path.join(OUTPUT_DIR, "plots")
    reports_dir = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(plots_dir,   exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if l1_path is None or not os.path.exists(l1_path or ""):
        print("  WARNING: L1 file not found — skipping step 6")
        return

    # 1. Track map
    track_out = plot_track(l1_path, l0_path)
    print()

    # 2. T-S diagram
    ts_out = plot_ts_diagram(l1_path)
    print()

    # 3. Data coverage matrix
    if l0_path and os.path.exists(l0_path):
        cov_out = plot_data_coverage(l0_path, l1_path)
        print()

    # 4. Mixed layer depth plot (uses GliderTools if available)
    mld_out = plot_mld(l1_path, grid_path)
    if mld_out:
        print()

    # 5. GliderTools QC comparison (if GT available)
    gt_out = run_gt_comparison(l0_path, l1_path)
    if gt_out:
        print()

    # 6. Deployment summary report
    report_out = write_summary_report(l0_path, l1_path, grid_path)

    elapsed = time.time() - t0
    print()
    print(f"  STEP 6 COMPLETE in {elapsed:.1f}s")
    print(f"  Plots:   {plots_dir}")
    print(f"  Reports: {reports_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--l0",         default=None)
    p.add_argument("--l1",         required=True)
    p.add_argument("--grid",       default=None)
    p.add_argument("--output-dir", default=None,
                   help="Override output directory")
    args = p.parse_args()

    import config as _cfg

    # Derive output dir from L1 path: output/l1/xxx.nc -> output/
    if args.output_dir:
        _cfg.OUTPUT_DIR = os.path.abspath(args.output_dir)
        OUTPUT_DIR = _cfg.OUTPUT_DIR
    else:
        # L1 is at <output_dir>/l1/<filename>.nc
        derived = os.path.dirname(os.path.dirname(os.path.abspath(args.l1)))
        _cfg.OUTPUT_DIR = derived
        OUTPUT_DIR = derived

    # Derive GLIDER_ID from L1 filename
    basename = os.path.basename(args.l1)
    if basename.startswith("incois_glider_") and basename.endswith("_L1.nc"):
        gid = basename[len("incois_glider_"):-len("_L1.nc")]
        _cfg.GLIDER_ID = gid
        GLIDER_ID = gid

    run_step6(l0_path=args.l0, l1_path=args.l1, grid_path=args.grid)
