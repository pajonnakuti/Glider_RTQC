#!/usr/bin/env python3
"""
step7.py - Comprehensive oceanographic plots from gridded L1 data.

Produces (in output/plots/):
  _contour_temp.png          — filled contour section: potential temperature
  _contour_salinity.png      — filled contour section: salinity
  _contour_oxygen.png        — filled contour section: oxygen + lag-corrected
  _contour_optics.png        — filled contour section: chlorophyll, cdom, bbp
  _contour_density.png       — potential density + isopycnal contours
  _profiles_envelope.png     — min/mean/max profile envelopes for T/S/O2
  _vertical_gradient.png     — dT/dz and dS/dz (thermocline/halocline depth)
  _hovmoller.png             — Hovmöller: temperature anomaly vs depth
  _surface_properties.png    — SST, SSS, surface O2, surface Chl timeseries
  _depth_timeseries.png      — depth of isotherms (20°C, 15°C, 10°C)
  _ts_density.png            — T-S diagram with density contours + time colour
"""
import os
import sys
import time
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import uniform_filter1d
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, GLIDER_ID

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import cmocean.cm as cmo
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

# ── colormaps ─────────────────────────────────────────────────
def _cmap(name):
    """Return cmocean colormap if available, else matplotlib fallback."""
    fallbacks = {
        "thermal":  "RdYlBu_r",
        "haline":   "viridis",
        "oxy":      "viridis",
        "algae":    "Greens",
        "matter":   "magma",
        "dense":    "Blues",
        "tempo":    "Purples",
        "delta":    "RdBu_r",
        "balance":  "RdBu_r",
    }
    if HAS_CMOCEAN:
        return getattr(cmo, name)
    return plt.get_cmap(fallbacks.get(name, "viridis"))


# ── shared helpers ────────────────────────────────────────────
GAP_H = 48  # mask time gaps longer than this (hours)

def _mask_gaps(V, t_vals, gap_h=GAP_H):
    """NaN the first column after each large time gap."""
    if len(t_vals) < 2:
        return V
    dt = np.diff(t_vals.astype("datetime64[h]").astype(float))
    for idx in np.where(dt > gap_h)[0] + 1:
        V[idx, :] = np.nan
    return V


def _pcolor_edges(centers):
    c = np.asarray(centers)
    if np.issubdtype(c.dtype, np.datetime64):
        c = mdates.date2num(c)
    e = np.empty(len(c) + 1)
    e[1:-1] = (c[1:] + c[:-1]) / 2
    e[0]    = c[0]  - (c[1]  - c[0])  / 2
    e[-1]   = c[-1] + (c[-1] - c[-2]) / 2
    return e


def _max_data_depth(V, depth_vals):
    col = np.any(np.isfinite(V), axis=0)
    return float(depth_vals[col].max()) if col.any() else float(depth_vals.max())


def _add_colorbar(fig, ax, mesh, label, fontsize=10):
    cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.015)
    cb.set_label(label, fontsize=fontsize)
    return cb


def _format_xaxis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)


def _save(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ── 1. Generic filled-contour section ────────────────────────
def _contour_section(grid, var, title, units, cmap_name,
                     depth_max=None, plot_path=None,
                     vmin=None, vmax=None, n_contours=12,
                     overlay_var=None, overlay_label=None):
    """
    One-panel filled contour section (pcolormesh + contour lines).
    overlay_var: optional second variable for overlaid black contours.
    """
    if var not in grid:
        print(f"  SKIP: {var} not in grid")
        return None

    V = grid[var].values.copy()
    T = grid.time.values
    D = grid.depth.values

    if depth_max is None:
        depth_max = _max_data_depth(V, D)
    dm = D <= depth_max
    V, D = V[:, dm], D[dm]
    V = _mask_gaps(V, T)

    valid = np.isfinite(V)
    if not valid.any():
        print(f"  SKIP: {var} all NaN")
        return None

    if vmin is None: vmin = np.nanpercentile(V[valid], 2)
    if vmax is None: vmax = np.nanpercentile(V[valid], 98)

    fig, ax = plt.subplots(figsize=(14, 5))
    te = _pcolor_edges(T)
    de = _pcolor_edges(D)

    mesh = ax.pcolormesh(te, de, V.T, cmap=_cmap(cmap_name),
                         vmin=vmin, vmax=vmax, shading="flat",
                         rasterized=True)
    _add_colorbar(fig, ax, mesh, f"{var} ({units})")

    # Contour lines — only where we have real data (no NaN fill)
    try:
        from scipy.ndimage import uniform_filter
        # Use a copy with NaN kept; matplotlib.contour handles masked arrays
        import numpy.ma as ma
        V_ma = ma.masked_invalid(V)
        X, Y = np.meshgrid(mdates.date2num(T), D)
        levels = np.linspace(vmin, vmax, n_contours)
        ax.contour(X.T, Y.T, V_ma, levels=levels,
                   colors="k", linewidths=0.4, alpha=0.4)
    except Exception:
        pass

    # Optional overlay (e.g. isopycnals on T or S section)
    if overlay_var and overlay_var in grid:
        OV = grid[overlay_var].values[:, dm].copy()
        OV = _mask_gaps(OV, T)
        import numpy.ma as ma
        OV_ma = ma.masked_invalid(OV)
        o_levels = np.linspace(
            np.nanpercentile(OV[np.isfinite(OV)], 5),
            np.nanpercentile(OV[np.isfinite(OV)], 95), 8)
        try:
            X2, Y2 = np.meshgrid(mdates.date2num(T), D)
            cs = ax.contour(X2.T, Y2.T, OV_ma, levels=o_levels,
                            colors="white", linewidths=0.8, alpha=0.7)
            ax.clabel(cs, fmt="%.1f", fontsize=7, colors="white")
        except Exception:
            pass

    ax.set_ylim(depth_max, 0)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title(f"Glider {GLIDER_ID}  —  {title}", fontsize=12, fontweight="bold")
    _format_xaxis(ax)

    _save(fig, plot_path)
    return plot_path


# ── 2. Dual-panel contour (e.g. O2 raw + lag-corrected) ──────
def _dual_contour_section(grid, var1, var2, titles, units,
                          cmap_name, depth_max=None, plot_path=None):
    vars_ok = [v for v in [var1, var2] if v in grid]
    if not vars_ok:
        print(f"  SKIP: neither {var1} nor {var2} in grid")
        return None

    T = grid.time.values
    D = grid.depth.values

    all_V = []
    for v in [var1, var2]:
        if v in grid:
            V = grid[v].values.copy()
            valid = V[np.isfinite(V)]
            vmin = np.nanpercentile(valid, 2) if len(valid) else 0
            vmax = np.nanpercentile(valid, 98) if len(valid) else 1
            all_V.append((v, V, vmin, vmax))

    dm = D <= (depth_max or _max_data_depth(all_V[0][1], D))
    D_trim = D[dm]

    fig, axes = plt.subplots(len(all_V), 1, figsize=(14, 5 * len(all_V)),
                             sharex=True, sharey=True)
    if len(all_V) == 1:
        axes = [axes]

    te = _pcolor_edges(T)
    de = _pcolor_edges(D_trim)

    for ax, (v, V, vmin, vmax), t_lbl in zip(axes, all_V, titles):
        V_trim = _mask_gaps(V[:, dm].copy(), T)
        if not np.any(np.isfinite(V_trim)):
            ax.set_title(f"{t_lbl} (NO DATA)")
            continue
        mesh = ax.pcolormesh(te, de, V_trim.T,
                             cmap=_cmap(cmap_name),
                             vmin=vmin, vmax=vmax,
                             shading="flat", rasterized=True)
        _add_colorbar(fig, ax, mesh, f"{v} ({units})")
        ax.set_ylim(D_trim.max(), 0)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title(f"Glider {GLIDER_ID}  —  {t_lbl}",
                     fontsize=11, fontweight="bold")

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


# ── 3. Profile envelopes ─────────────────────────────────────
def plot_profile_envelopes(grid, plot_path=None):
    """Min / mean / max profile envelopes for T, S, O2."""
    print("  Generating profile envelopes...")
    vars_cfg = [
        ("potential_temperature", "Pot. Temperature (°C)", "thermal"),
        ("salinity",              "Salinity (PSU)",         "haline"),
        ("oxygen_concentration",  "Oxygen (µmol/l)",        "oxy"),
    ]
    vars_present = [(v, l, c) for v, l, c in vars_cfg if v in grid]
    if not vars_present:
        print("  SKIP: no variables for envelopes")
        return None

    D = grid.depth.values
    fig, axes = plt.subplots(1, len(vars_present),
                             figsize=(5 * len(vars_present), 9), sharey=True)
    if len(vars_present) == 1:
        axes = [axes]

    fig.suptitle(f"Glider {GLIDER_ID}  —  Profile Envelopes",
                 fontsize=13, fontweight="bold")

    for ax, (var, label, _) in zip(axes, vars_present):
        V = grid[var].values          # (n_prof, n_depth)
        mn  = np.nanmin(V,  axis=0)
        mx  = np.nanmax(V,  axis=0)
        med = np.nanmedian(V, axis=0)
        p25 = np.nanpercentile(V, 25, axis=0)
        p75 = np.nanpercentile(V, 75, axis=0)

        valid = np.isfinite(med)
        ax.fill_betweenx(D[valid], mn[valid], mx[valid],
                         alpha=0.15, color="steelblue", label="Min–Max")
        ax.fill_betweenx(D[valid], p25[valid], p75[valid],
                         alpha=0.35, color="steelblue", label="IQR")
        ax.plot(med[valid], D[valid], color="steelblue",
                linewidth=2, label="Median")

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title(var.replace("_", " ").title(), fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="lower right")

    # invert once after all panels are drawn (sharey means one inversion is enough)
    axes[0].invert_yaxis()

    _save(fig, plot_path)
    return plot_path


# ── 4. Surface property timeseries ───────────────────────────
def plot_surface_properties(grid, plot_path=None):
    """SST, SSS, surface O2, surface Chl timeseries at 0–10 m."""
    print("  Generating surface properties timeseries...")
    T = grid.time.values
    D = grid.depth.values
    surf = D <= 15  # top 15 m = "surface"

    props = [
        ("potential_temperature", "SST (°C)",        "thermal"),
        ("salinity",              "SSS (PSU)",         "haline"),
        ("oxygen_concentration",  "O₂ (µmol/l)",      "oxy"),
        ("chlorophyll",           "Chl (mg m⁻³)",      "algae"),
    ]
    props_ok = [(v, l, c) for v, l, c in props if v in grid]
    if not props_ok:
        print("  SKIP: no surface vars")
        return None

    fig, axes = plt.subplots(len(props_ok), 1,
                             figsize=(14, 3 * len(props_ok)),
                             sharex=True)
    if len(props_ok) == 1:
        axes = [axes]

    fig.suptitle(f"Glider {GLIDER_ID}  —  Surface Properties (0–15 m)",
                 fontsize=13, fontweight="bold")

    for ax, (var, label, cmap_n) in zip(axes, props_ok):
        V_surf = np.nanmean(grid[var].values[:, surf], axis=1)
        valid  = np.isfinite(V_surf)
        if not valid.any():
            ax.set_title(f"{label} — NO DATA")
            continue
        # Smooth slightly
        sm = uniform_filter1d(V_surf, size=5, mode="nearest")
        sm[~valid] = np.nan
        ax.plot(T[valid], sm[valid], linewidth=1.5, color="steelblue")
        # fill between line and axis bottom (not y=0 which is wrong for temperatures)
        ax.fill_between(T[valid], sm[valid],
                        np.nanmin(sm[valid]),
                        alpha=0.15, color="steelblue")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(label, fontsize=10)

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


# ── 5. Isotherm / isopycnal depth timeseries ─────────────────
def plot_isotherm_depths(grid, plot_path=None):
    """Track depth of specific isotherms and isopycnals over time."""
    print("  Generating isotherm depth timeseries...")
    T_vals  = grid.time.values
    D_vals  = grid.depth.values
    n_t     = len(T_vals)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Glider {GLIDER_ID}  —  Isotherm & Isopycnal Depths",
                 fontsize=13, fontweight="bold")

    # --- Isotherms ---
    ax = axes[0]
    if "potential_temperature" in grid:
        V = grid["potential_temperature"].values
        for T_iso, col in [(28, "#d62728"), (25, "#ff7f0e"),
                           (20, "#2ca02c"), (15, "#1f77b4"), (10, "#9467bd")]:
            depths = np.full(n_t, np.nan)
            for i in range(n_t):
                prof = V[i, :]
                valid = np.isfinite(prof)
                if valid.sum() < 3:
                    continue
                # Only compute isotherm if surface temperature is ABOVE the threshold
                # (i.e. the isotherm actually exists in the water column)
                surf_T = prof[valid][0]  # shallowest valid point
                if surf_T <= T_iso:
                    continue  # water never warm enough — isotherm absent
                # Find where temperature drops below T_iso (linear interpolation)
                d_v = D_vals[valid]
                p_v = prof[valid]
                below = np.where(p_v < T_iso)[0]
                if below.size == 0:
                    continue  # temp stays above T_iso to max depth
                idx = below[0]
                if idx == 0:
                    depths[i] = float(d_v[0])
                else:
                    # Linear interpolation between idx-1 and idx
                    t1, t2 = p_v[idx-1], p_v[idx]
                    d1, d2 = d_v[idx-1], d_v[idx]
                    if t1 != t2:
                        depths[i] = d1 + (T_iso - t1) / (t2 - t1) * (d2 - d1)
            valid_mask = np.isfinite(depths)
            if valid_mask.any():
                ax.plot(T_vals[valid_mask], depths[valid_mask],
                        linewidth=1.5, color=col, label=f"{T_iso}°C")
        ax.invert_yaxis()
        ax.set_ylim(500, 0)   # isotherms are typically in top 500m
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title("Isotherm Depths", fontsize=11)
        ax.legend(fontsize=9, ncol=5, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    # --- Isopycnals ---
    ax = axes[1]
    pden_var = "potential_density" if "potential_density" in grid else None
    if pden_var:
        V = grid[pden_var].values
        for rho, col in [(1024.0, "#d62728"), (1025.0, "#ff7f0e"),
                         (1025.5, "#2ca02c"), (1026.0, "#1f77b4"),
                         (1026.5, "#9467bd")]:
            depths = np.full(n_t, np.nan)
            for i in range(n_t):
                prof = V[i, :]
                valid_mask = np.isfinite(prof)
                if valid_mask.sum() < 3:
                    continue
                d_v = D_vals[valid_mask]
                p_v = prof[valid_mask]
                # Only compute if surface density is BELOW the isopycnal
                if p_v[0] >= rho:
                    continue  # too dense at surface — isopycnal absent
                below = np.where(p_v > rho)[0]
                if below.size == 0:
                    continue
                idx = below[0]
                if idx == 0:
                    depths[i] = float(d_v[0])
                else:
                    r1, r2 = p_v[idx-1], p_v[idx]
                    d1, d2 = d_v[idx-1], d_v[idx]
                    if r1 != r2:
                        depths[i] = d1 + (rho - r1) / (r2 - r1) * (d2 - d1)
            valid_mask2 = np.isfinite(depths)
            if valid_mask2.any():
                ax.plot(T_vals[valid_mask2], depths[valid_mask2],
                        linewidth=1.5, color=col,
                        label=f"σ₀={rho-1000:.1f}")
        ax.invert_yaxis()
        ax.set_ylim(600, 0)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title("Isopycnal Depths", fontsize=11)
        ax.legend(fontsize=9, ncol=5, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
    else:
        axes[1].set_title("Isopycnal Depths — no density data")

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


# ── 6. Hovmöller anomaly ─────────────────────────────────────
def plot_hovmoller(grid, plot_path=None):
    """Temperature anomaly (T − time-mean T at each depth)."""
    print("  Generating Hovmöller anomaly plot...")
    if "potential_temperature" not in grid:
        print("  SKIP: no temperature")
        return None

    V = grid["potential_temperature"].values.copy()
    T = grid.time.values
    D = grid.depth.values
    dm = D <= _max_data_depth(V, D)
    V, D = V[:, dm], D[dm]
    V = _mask_gaps(V, T)

    # Anomaly = T(t,z) − time-mean at each depth (climatological profile removed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_at_depth = np.nanmean(V, axis=0)   # shape: (n_depth,)
    anom = V - mean_at_depth[np.newaxis, :]

    valid = np.isfinite(anom)
    if not valid.any():
        return None

    vlim = np.nanpercentile(np.abs(anom[valid]), 95)
    fig, ax = plt.subplots(figsize=(14, 5))
    te = _pcolor_edges(T)
    de = _pcolor_edges(D)

    mesh = ax.pcolormesh(te, de, anom.T,
                         cmap=_cmap("balance"),
                         vmin=-vlim, vmax=vlim,
                         shading="flat", rasterized=True)
    _add_colorbar(fig, ax, mesh, "T anomaly (°C)")
    ax.set_ylim(D.max(), 0)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.set_title(f"Glider {GLIDER_ID}  —  Temperature Anomaly "
                 f"(T − depth-mean)",
                 fontsize=12, fontweight="bold")
    _format_xaxis(ax)
    _save(fig, plot_path)
    return plot_path


# ── 7. Vertical gradients ────────────────────────────────────
def plot_vertical_gradients(grid, plot_path=None):
    """dT/dz and dS/dz sections — thermocline and halocline structure."""
    print("  Generating vertical gradient sections...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Glider {GLIDER_ID}  —  Vertical Gradients",
                 fontsize=13, fontweight="bold")

    cfgs = [
        ("potential_temperature", "dT/dz (°C m⁻¹)", "balance", axes[0]),
        ("salinity",              "dS/dz (PSU m⁻¹)", "delta",   axes[1]),
    ]

    T = grid.time.values
    D = grid.depth.values
    te = _pcolor_edges(T)

    for var, ylabel, cmap_n, ax in cfgs:
        if var not in grid:
            ax.set_title(f"{ylabel} — no data")
            continue
        V = grid[var].values.copy()
        dm = D <= min(600, _max_data_depth(V, D))
        V_trim, D_trim = V[:, dm], D[dm]
        V_trim = _mask_gaps(V_trim, T)

        # Gradient: dV/dz using actual depth spacing (correct for 1m bins)
        dVdz = np.gradient(V_trim, D_trim, axis=1)
        dVdz[~np.isfinite(V_trim)] = np.nan

        valid = np.isfinite(dVdz)
        if not valid.any():
            ax.set_title(f"{ylabel} — all NaN")
            continue

        vlim = np.nanpercentile(np.abs(dVdz[valid]), 97)
        de = _pcolor_edges(D_trim)
        mesh = ax.pcolormesh(te, de, dVdz.T,
                             cmap=_cmap(cmap_n),
                             vmin=-vlim, vmax=vlim,
                             shading="flat", rasterized=True)
        _add_colorbar(fig, ax, mesh, ylabel)
        ax.set_ylim(D_trim.max(), 0)
        ax.set_ylabel("Depth (m)", fontsize=11)
        ax.set_title(ylabel, fontsize=11)

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


# ── 8. T-S diagram with density contours ─────────────────────
def plot_ts_density(grid, plot_path=None):
    """
    T-S diagram from gridded data coloured by time, with
    sigma-0 density contours overlaid.
    """
    print("  Generating T-S density diagram...")
    if "potential_temperature" not in grid or "salinity" not in grid:
        print("  SKIP: no T or S")
        return None

    T_all = grid["potential_temperature"].values.ravel()
    S_all = grid["salinity"].values.ravel()
    D_all = np.tile(grid.depth.values, len(grid.time))
    t_num = np.repeat(
        mdates.date2num(grid.time.values), len(grid.depth))

    valid = np.isfinite(T_all) & np.isfinite(S_all) & np.isfinite(D_all)
    T_v, S_v = T_all[valid], S_all[valid]
    t_v, D_v = t_num[valid], D_all[valid]

    if len(T_v) < 100:
        print("  SKIP: too few T-S points")
        return None

    # Subsample for speed
    if len(T_v) > 100_000:
        idx = np.random.choice(len(T_v), 100_000, replace=False)
        T_v, S_v, t_v, D_v = T_v[idx], S_v[idx], t_v[idx], D_v[idx]

    t_norm = (t_v - t_v.min()) / max(t_v.max() - t_v.min(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Glider {GLIDER_ID}  —  T-S Diagrams",
                 fontsize=13, fontweight="bold")

    for ax, colour_arr, cb_label, cmap_n in [
        (axes[0], t_norm, "Time",        "plasma"),
        (axes[1], D_v,    "Depth (m)",   "viridis_r"),
    ]:
        sc = ax.scatter(S_v, T_v, c=colour_arr, cmap=cmap_n,
                        s=4, alpha=0.5, rasterized=True,
                        vmin=colour_arr.min(), vmax=colour_arr.max())
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.03)
        if cb_label == "Time":
            n_tick = 5
            tick_pos = np.linspace(0, 1, n_tick)
            tick_vals = t_v.min() + tick_pos * (t_v.max() - t_v.min())
            tick_lbls = [mdates.num2date(v).strftime("%b %Y")
                         for v in tick_vals]
            cb.set_ticks(tick_pos)
            cb.set_ticklabels(tick_lbls, fontsize=8)
        cb.set_label(cb_label, fontsize=10)

        # Density contours
        if HAS_GSW:
            s_r = np.linspace(S_v.min() - 0.1, S_v.max() + 0.1, 80)
            t_r = np.linspace(T_v.min() - 0.5, T_v.max() + 0.5, 80)
            SS, TT = np.meshgrid(s_r, t_r)
            lat_m = float(np.nanmean(grid["latitude"].values)) if "latitude" in grid else 12.0
            lon_m = float(np.nanmean(grid["longitude"].values)) if "longitude" in grid else 75.0
            try:
                SA = gsw.SA_from_SP(SS, 0, lon_m, lat_m)
                CT = gsw.CT_from_t(SA, TT, 0)
                sig = gsw.sigma0(SA, CT)
                lvls = np.arange(np.floor(sig.min()), np.ceil(sig.max()) + 0.5, 0.5)
                cs = ax.contour(SS, TT, sig, levels=lvls,
                                colors="gray", linewidths=0.6, alpha=0.6)
                ax.clabel(cs, fmt="%.1f", fontsize=8)
            except Exception:
                pass

        ax.set_xlabel("Salinity (PSU)", fontsize=11)
        ax.set_ylabel("Potential Temperature (°C)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"Coloured by {cb_label}", fontsize=11)

    _save(fig, plot_path)
    return plot_path


# ── 9. Five-panel overview section ───────────────────────────
def plot_overview_section(grid, plot_path=None):
    """
    Single figure with 5 stacked contour panels:
    T / S / O2 / Chl / Density — the 'summary' section plot.
    """
    print("  Generating overview section (5 panels)...")
    T = grid.time.values
    D = grid.depth.values

    panels = [
        ("potential_temperature", "Pot. Temp (°C)",   "thermal",   None,  None),
        ("salinity",              "Salinity (PSU)",    "haline",    None,  None),
        ("oxygen_concentration",  "Oxygen (µmol/l)",   "oxy",       None,  None),
        ("chlorophyll",           "Chl (mg m⁻³)",      "algae",     None,  0.1),
        ("potential_density",     "σ₀ (kg m⁻³)",       "dense",     None,  None),
    ]
    panels_ok = [(v, l, c, vn, vx) for v, l, c, vn, vx in panels if v in grid]
    if not panels_ok:
        print("  SKIP: no variables for overview")
        return None

    fig, axes = plt.subplots(len(panels_ok), 1,
                             figsize=(14, 4 * len(panels_ok)),
                             sharex=True)
    if len(panels_ok) == 1:
        axes = [axes]

    fig.suptitle(f"Glider {GLIDER_ID}  —  Oceanographic Sections",
                 fontsize=14, fontweight="bold", y=1.001)

    for ax, (var, label, cmap_n, vmin, vmax) in zip(axes, panels_ok):
        V = grid[var].values.copy()
        dm = D <= _max_data_depth(V, D)
        V_t, D_t = _mask_gaps(V[:, dm].copy(), T), D[dm]

        valid = np.isfinite(V_t)
        if not valid.any():
            ax.set_title(f"{label} — NO DATA"); continue

        if vmin is None: vmin = np.nanpercentile(V_t[valid], 2)
        if vmax is None: vmax = np.nanpercentile(V_t[valid], 98)
        # chlorophyll: force min=0
        if "chlor" in var: vmin = max(vmin, 0.0)

        te = _pcolor_edges(T)
        de = _pcolor_edges(D_t)
        mesh = ax.pcolormesh(te, de, V_t.T,
                             cmap=_cmap(cmap_n),
                             vmin=vmin, vmax=vmax,
                             shading="flat", rasterized=True)
        _add_colorbar(fig, ax, mesh, label)

        # Contour lines
        X, Y = np.meshgrid(mdates.date2num(T), D_t)
        V_sm = V_t.copy()
        V_sm[~np.isfinite(V_sm)] = np.nanmean(V_sm[np.isfinite(V_sm)])
        try:
            ax.contour(X.T, Y.T, V_sm,
                       levels=np.linspace(vmin, vmax, 10),
                       colors="k", linewidths=0.3, alpha=0.4)
        except Exception:
            pass

        ax.set_ylim(D_t.max(), 0)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


# ── main entry point ─────────────────────────────────────────
def run_step7(grid_path=None, l1_path=None):
    print("=" * 60)
    print("  STEP 7: Oceanographic Section Plots")
    print("=" * 60)
    t0 = time.time()

    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Resolve grid
    if grid_path is None:
        grid_path = os.path.join(OUTPUT_DIR, "gridfiles",
                                 f"incois_glider_{GLIDER_ID}_grid.nc")
    if not os.path.exists(grid_path):
        print(f"  ERROR: grid not found: {grid_path}")
        return

    grid = xr.open_dataset(grid_path)
    print(f"  Grid: {len(grid.time)} profiles × {len(grid.depth)} depth bins")
    print()

    def p(name):
        return os.path.join(plots_dir, f"incois_glider_{GLIDER_ID}_{name}.png")

    # 1. Individual variable contour sections
    _contour_section(grid, "potential_temperature",
                     "Potential Temperature", "°C", "thermal",
                     overlay_var="potential_density",
                     overlay_label="σ₀",
                     plot_path=p("contour_temp"))
    print()

    _contour_section(grid, "salinity",
                     "Salinity", "PSU", "haline",
                     overlay_var="potential_density",
                     overlay_label="σ₀",
                     plot_path=p("contour_salinity"))
    print()

    # Dual O2 panel: raw + lag-corrected
    _dual_contour_section(grid,
                          "oxygen_concentration",
                          "oxygen_concentration_lag_corrected",
                          ["Oxygen (raw)", "Oxygen (lag-corrected)"],
                          "µmol/l", "oxy",
                          plot_path=p("contour_oxygen"))
    print()

    # Optics 3-panel
    _contour_optics(grid, p("contour_optics"))
    print()

    # Density with isopycnal contours
    _contour_section(grid, "potential_density",
                     "Potential Density σ₀", "kg m⁻³", "dense",
                     n_contours=15,
                     plot_path=p("contour_density"))
    print()

    # 2. Profile envelopes
    plot_profile_envelopes(grid, p("profiles_envelope"))
    print()

    # 3. Surface properties
    plot_surface_properties(grid, p("surface_properties"))
    print()

    # 4. Isotherm / isopycnal depths
    plot_isotherm_depths(grid, p("depth_timeseries"))
    print()

    # 5. Hovmöller anomaly
    plot_hovmoller(grid, p("hovmoller"))
    print()

    # 6. Vertical gradients
    plot_vertical_gradients(grid, p("vertical_gradient"))
    print()

    # 7. T-S density diagrams
    plot_ts_density(grid, p("ts_density"))
    print()

    # 8. Five-panel overview
    plot_overview_section(grid, p("overview_section"))

    grid.close()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  STEP 7 COMPLETE in {elapsed:.1f}s")
    print(f"  Plots in: {plots_dir}")
    print("=" * 60)


def _contour_optics(grid, plot_path):
    """Three-panel optics section: Chl, CDOM, Backscatter."""
    print("  Generating optics contour sections...")
    vars_cfg = [
        ("chlorophyll",    "Chlorophyll (mg m⁻³)", "algae",  0.0, None),
        ("cdom",           "CDOM (ppb)",            "tempo",  0.0, None),
        ("backscatter_700","BBP700 (m⁻¹)",          "matter", 0.0, None),
    ]
    vars_ok = [(v, l, c, vn, vx) for v, l, c, vn, vx in vars_cfg
               if v in grid and np.any(np.isfinite(grid[v].values))]
    if not vars_ok:
        print("  SKIP: no optics data")
        return None

    T = grid.time.values
    D = grid.depth.values

    fig, axes = plt.subplots(len(vars_ok), 1,
                             figsize=(14, 4 * len(vars_ok)),
                             sharex=True)
    if len(vars_ok) == 1:
        axes = [axes]

    fig.suptitle(f"Glider {GLIDER_ID}  —  Optical Properties",
                 fontsize=13, fontweight="bold", y=1.001)

    for ax, (var, label, cmap_n, vmin_f, vmax_f) in zip(axes, vars_ok):
        V = grid[var].values.copy()
        dm = D <= _max_data_depth(V, D)
        V_t, D_t = _mask_gaps(V[:, dm].copy(), T), D[dm]
        valid = np.isfinite(V_t)
        if not valid.any():
            ax.set_title(f"{label} — NO DATA"); continue

        vmin = max(0.0, np.nanpercentile(V_t[valid], 2))
        vmax = np.nanpercentile(V_t[valid], 98)
        if vmin >= vmax:
            vmax = vmin + 1e-6

        te = _pcolor_edges(T)
        de = _pcolor_edges(D_t)
        mesh = ax.pcolormesh(te, de, V_t.T,
                             cmap=_cmap(cmap_n),
                             vmin=vmin, vmax=vmax,
                             shading="flat", rasterized=True)
        _add_colorbar(fig, ax, mesh, label)
        ax.set_ylim(D_t.max(), 0)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")

    _format_xaxis(axes[-1])
    _save(fig, plot_path)
    return plot_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid",       required=True)
    p.add_argument("--l1",         default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    import config as _cfg
    if args.output_dir:
        _cfg.OUTPUT_DIR = os.path.abspath(args.output_dir)
        OUTPUT_DIR = _cfg.OUTPUT_DIR
    else:
        OUTPUT_DIR = os.path.dirname(os.path.dirname(
            os.path.abspath(args.grid)))
        _cfg.OUTPUT_DIR = OUTPUT_DIR

    gid = os.path.basename(args.grid)
    if "_grid.nc" in gid:
        gid = gid.replace("incois_glider_", "").replace("_grid.nc", "")
        _cfg.GLIDER_ID = gid
        GLIDER_ID = gid

    run_step7(grid_path=args.grid, l1_path=args.l1)
