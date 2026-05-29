#!/usr/bin/env python3
"""
verify.py — L1 product diagnostics and verification.

  1. GPS track anomalies (gaps, jumps, wrong locations)
  2. T-S diagram problems (out-of-range, over-smoothed, spikes)
  3. QC flag inconsistencies (temp_QC vs salinity_QC, pressure cascade)

Usage:
    python pipeline/verify.py --input <L1_NetCDF> [--l0 <L0_NetCDF>]
"""
import argparse
import os
import sys
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_gps(ds):
    """1. GPS / MAP diagnostics."""
    print("\n" + "=" * 70)
    print("  1. GPS / TRACK DIAGNOSTICS")
    print("=" * 70)

    issues = []
    n = len(ds.time)
    lat = ds['latitude'].values if 'latitude' in ds else None
    lon = ds['longitude'].values if 'longitude' in ds else None

    if lat is None or lon is None:
        print("  SKIP: latitude/longitude not found")
        return issues

    lat_nan = int(np.sum(~np.isfinite(lat)))
    lon_nan = int(np.sum(~np.isfinite(lon)))
    print(f"\n  Latitude range:  [{np.nanmin(lat):.6f}, {np.nanmax(lat):.6f}]")
    print(f"  Longitude range: [{np.nanmin(lon):.6f}, {np.nanmax(lon):.6f}]")
    print(f"  NaN latitudes:   {lat_nan:,} / {n:,}")
    print(f"  NaN longitudes:  {lon_nan:,} / {n:,}")

    if lat_nan > n * 0.5:
        issues.append(f"CRITICAL: >50% of latitudes are NaN ({lat_nan:,}/{n:,})")

    # Impossible values
    lat_out = int(np.sum((np.abs(lat) > 90) & np.isfinite(lat)))
    lon_out = int(np.sum((np.abs(lon) > 180) & np.isfinite(lon)))
    if lat_out > 0:
        issues.append(f"WARNING: {lat_out} latitudes outside [-90, 90]")
    if lon_out > 0:
        issues.append(f"WARNING: {lon_out} longitudes outside [-180, 180]")

    # Arabian Sea check
    arabian = np.isfinite(lat) & np.isfinite(lon) & (lat >= 5) & (lat <= 25) & (lon >= 55) & (lon <= 80)
    n_arabian = int(np.sum(arabian))
    print(f"  In Arabian Sea:  {n_arabian:,} / {n:,} ({100*n_arabian/n:.1f}%)")

    # Sudden jumps — check per-profile to avoid false positives at profile boundaries
    profile_index = ds['profile_index'].values if 'profile_index' in ds else None
    valid = np.isfinite(lat) & np.isfinite(lon)
    if np.sum(valid) > 1:
        lat_v = lat[valid]
        lon_v = lon[valid]
        dlat = np.diff(lat_v) * 111.32  # km
        dlon = np.diff(lon_v) * 111.32 * np.cos(np.radians(np.mean(lat_v)))
        dist = np.sqrt(dlat**2 + dlon**2)

        # If we have profile info, only flag jumps within the same profile
        if profile_index is not None:
            valid_idx = np.where(valid)[0]
            prof_at_valid = profile_index[valid_idx]
            same_prof = np.diff(prof_at_valid) == 0
            within_prof_jumps = dist[same_prof] if len(same_prof) == len(dist) else dist
            big_jumps = int(np.sum(within_prof_jumps > 50))
            if big_jumps > 0:
                issues.append(f"WARNING: {big_jumps} GPS jumps >50km WITHIN same profile")
        else:
            big_jumps = int(np.sum(dist > 50))
            if big_jumps > 0:
                issues.append(f"WARNING: {big_jumps} GPS jumps >50km between consecutive points")

        print(f"  Sudden jumps >50km (within profile): {big_jumps:,}")
        if big_jumps > 0:
            jump_idx = np.where(dist > 50)[0][:3]
            for ji in jump_idx:
                print(f"    Jump: ({lat_v[ji]:.4f},{lon_v[ji]:.4f}) -> ({lat_v[ji+1]:.4f},{lon_v[ji+1]:.4f}) = {dist[ji]:.1f} km")

    # Stuck positions — gliders only get GPS at surface, so most positions are interpolated
    if np.sum(valid) > 10:
        lat_unique = len(np.unique(np.round(lat_v, 6)))
        lon_unique = len(np.unique(np.round(lon_v, 6)))
        print(f"  Unique GPS (6dp): {lat_unique}/{np.sum(valid)} lat, {lon_unique}/{np.sum(valid)} lon")
        if profile_index is not None:
            n_prof = len(np.unique(profile_index[np.isfinite(profile_index)]))
            print(f"  (Glider: GPS only acquired at surface ~{n_prof} times, rest interpolated)")

    return issues


def check_ts(ds):
    """2. T-S diagram diagnostics."""
    print("\n" + "=" * 70)
    print("  2. TEMPERATURE-SALINITY DIAGNOSTICS")
    print("=" * 70)

    issues = []
    n = len(ds.time)
    temp = ds['temperature'].values if 'temperature' in ds else None
    sal = ds['salinity'].values if 'salinity' in ds else None

    if temp is None or sal is None:
        print("  SKIP: temperature/salinity not found")
        return issues

    valid_ts = np.isfinite(temp) & np.isfinite(sal)
    n_ts = int(np.sum(valid_ts))
    print(f"\n  Valid T-S pairs: {n_ts:,} / {n:,}")

    t_min, t_max = np.nanmin(temp), np.nanmax(temp)
    s_min, s_max = np.nanmin(sal), np.nanmax(sal)
    print(f"  Temperature range: [{t_min:.4f}, {t_max:.4f}] C")
    print(f"  Salinity range:    [{s_min:.4f}, {s_max:.4f}] PSU")

    # Physical plausibility
    temp_neg = int(np.sum(temp < -2.5))
    temp_hot = int(np.sum(temp > 40))
    sal_low = int(np.sum(sal < 2))
    sal_high = int(np.sum(sal > 41))

    if temp_neg > 0:
        issues.append(f"WARNING: {temp_neg} temperature values < -2.5C")
    if temp_hot > 0:
        issues.append(f"WARNING: {temp_hot} temperature values > 40C")
    if sal_low > 0:
        issues.append(f"WARNING: {sal_low} salinity values < 2 PSU")
    if sal_high > 0:
        issues.append(f"WARNING: {sal_high} salinity values > 41 PSU")

    print(f"\n  Physical plausibility:")
    print(f"    Temp < -2.5C:  {temp_neg:,}")
    print(f"    Temp > 40C:    {temp_hot:,}")
    print(f"    Salinity < 2:  {sal_low:,}")
    print(f"    Salinity > 41: {sal_high:,}")

    # Consecutive jumps — check per-profile to avoid false positives at boundaries
    profile_index = ds['profile_index'].values if 'profile_index' in ds else None
    t_v = temp[valid_ts]
    s_v = sal[valid_ts]
    if len(t_v) > 2:
        if profile_index is not None:
            ts_prof = profile_index[valid_ts]
            same_prof = np.diff(ts_prof) == 0
            t_diff = np.abs(np.diff(t_v))
            s_diff = np.abs(np.diff(s_v))
            t_jumps = int(np.sum(t_diff[same_prof] > 10))
            s_jumps = int(np.sum(s_diff[same_prof] > 5))
        else:
            t_diff = np.abs(np.diff(t_v))
            s_diff = np.abs(np.diff(s_v))
            t_jumps = int(np.sum(t_diff > 10))
            s_jumps = int(np.sum(s_diff > 5))

        if t_jumps > 0:
            issues.append(f"WARNING: {t_jumps} temperature jumps >10C within same profile")
        if s_jumps > 0:
            issues.append(f"WARNING: {s_jumps} salinity jumps >5 PSU within same profile")

        # Check for over-smoothing — use 6dp for temperature, 5dp for salinity
        # (glider T/S sensors have ~0.001C / 0.001 PSU precision)
        t_unique = len(np.unique(np.round(t_v, 6)))
        s_unique = len(np.unique(np.round(s_v, 5)))
        t_stuck_frac = 1.0 - t_unique / len(t_v)
        s_stuck_frac = 1.0 - s_unique / len(s_v)
        print(f"\n  Smoothing check:")
        print(f"    Unique temp values (6dp): {t_unique}/{len(t_v)} (stuck: {t_stuck_frac*100:.1f}%)")
        print(f"    Unique sal values (5dp):  {s_unique}/{len(s_v)} (stuck: {s_stuck_frac*100:.1f}%)")
        if t_stuck_frac > 0.95:
            issues.append(f"WARNING: Temperature appears over-smoothed — {t_stuck_frac*100:.1f}% stuck values")
        if s_stuck_frac > 0.95:
            issues.append(f"WARNING: Salinity appears over-smoothed — {s_stuck_frac*100:.1f}% stuck values")

    # Arabian Sea T-S sanity
    warm = valid_ts & (temp > 25)
    if np.sum(warm) > 0:
        warm_sal = sal[warm]
        print(f"\n  Warm water (>25C) salinity: [{np.nanmin(warm_sal):.2f}, {np.nanmax(warm_sal):.2f}] PSU (mean: {np.nanmean(warm_sal):.2f})")

    return issues


def check_qc(ds):
    """3. QC flag consistency diagnostics."""
    print("\n" + "=" * 70)
    print("  3. QC FLAG DIAGNOSTICS")
    print("=" * 70)

    issues = []
    n = len(ds.time)
    qc_vars = [v for v in ds.data_vars if v.endswith('_QC')]
    print(f"\n  QC variables found: {len(qc_vars)}")

    for qv in sorted(qc_vars):
        qc = ds[qv].values.astype(int)
        good = int(np.sum(qc == 1))
        prob_good = int(np.sum(qc == 2))
        prob_bad = int(np.sum(qc == 3))
        bad = int(np.sum(qc == 4))
        missing = int(np.sum(qc == 9))
        other = int(np.sum((qc != 1) & (qc != 2) & (qc != 3) & (qc != 4) & (qc != 9)))
        pct = 100.0 * good / n if n > 0 else 0
        print(f"    {qv:40s} Good:{good:>8,} ({pct:5.1f}%)  PBad:{prob_bad:>6,}  Bad:{bad:>6,}  Miss:{missing:>6,}")

    # temp_QC vs salinity_QC consistency
    if 'temperature_QC' in ds and 'salinity_QC' in ds:
        t_qc = ds['temperature_QC'].values
        s_qc = ds['salinity_QC'].values

        print(f"\n  temperature_QC vs salinity_QC consistency:")

        # When temp is bad, salinity should also be bad (ARGO rule 2.1.4b)
        temp_bad = (t_qc == 4)
        sal_not_bad = (s_qc != 4) & (s_qc != 9)
        inconsistent = int(np.sum(temp_bad & sal_not_bad))
        if inconsistent > 0:
            issues.append(f"CRITICAL: {inconsistent:,} points where temp is BAD but salinity is NOT bad")
        print(f"    Temp=BAD but salinity!=BAD: {inconsistent:,}")

        # When temp is missing, salinity should also be missing
        temp_miss = (t_qc == 9)
        sal_not_miss = (s_qc != 9)
        inconsistent_miss = int(np.sum(temp_miss & sal_not_miss))
        if inconsistent_miss > 0:
            issues.append(f"WARNING: {inconsistent_miss:,} points where temp is MISSING but salinity is NOT missing")
        print(f"    Temp=MISS but salinity!=MISS: {inconsistent_miss:,}")

        # Distribution of QC combinations
        print(f"\n  QC flag combinations (temp_QC, sal_QC):")
        for tq in [1, 2, 3, 4, 9]:
            for sq in [1, 2, 3, 4, 9]:
                count = int(np.sum((t_qc == tq) & (s_qc == sq)))
                if count > 0:
                    print(f"    T={tq}, S={sq}: {count:>10,}")

    # Pressure cascade check
    if 'pressure_QC' in ds:
        p_qc = ds['pressure_QC'].values
        p_bad = (p_qc == 4) | (p_qc == 9)
        print(f"\n  Pressure QC cascade:")
        print(f"    Pressure bad/missing: {int(np.sum(p_bad)):,} / {n:,}")

        for var in ['temperature', 'salinity', 'oxygen_concentration', 'chlorophyll']:
            qv = f"{var}_QC"
            if qv in ds:
                v_qc = ds[qv].values
                v_not_bad = (v_qc != 4) & (v_qc != 9)
                cascade_fail = int(np.sum(p_bad & v_not_bad))
                if cascade_fail > 0:
                    issues.append(f"CRITICAL: {cascade_fail:,} points where {qv} not flagged when pressure is bad/missing")
                    print(f"    WARNING: {qv} not bad when pressure is bad: {cascade_fail:,} points")

    return issues


def compare_l0_l1(l1_path, l0_path):
    """4. L0 vs L1 comparison."""
    print("\n" + "=" * 70)
    print("  4. L0 vs L1 COMPARISON")
    print("=" * 70)

    issues = []
    if l0_path is None or not os.path.exists(l0_path):
        print(f"\n  L0 file not found: {l0_path}")
        print("  Skipping L0 vs L1 comparison")
        return issues

    ds0 = xr.open_dataset(l0_path)
    ds1 = xr.open_dataset(l1_path)
    n0 = len(ds0.time)
    n1 = len(ds1.time)

    print(f"\n  L0 time points: {n0:,}")
    print(f"  L1 time points: {n1:,}")
    if n0 != n1:
        issues.append(f"WARNING: L0 has {n0:,} points but L1 has {n1:,} points (difference: {n1-n0:+,})")
        print(f"  Difference: {n1-n0:+,}")
        ds0, ds1 = xr.align(ds0, ds1, join="inner")
        n_aligned = len(ds0.time)
        print(f"  Aligned on common time: {n_aligned:,} points")
        if n_aligned == 0:
            ds0.close()
            ds1.close()
            issues.append("CRITICAL: No common time points between L0 and L1")
            return issues

    for var in ['temperature', 'salinity', 'pressure', 'chlorophyll', 'cdom', 'backscatter_700', 'oxygen_concentration']:
        if var not in ds0 or var not in ds1:
            continue
        v0 = ds0[var].values
        v1 = ds1[var].values
        ok0 = int(np.sum(np.isfinite(v0)))
        ok1 = int(np.sum(np.isfinite(v1)))
        print(f"\n  {var}:")
        print(f"    L0: valid={ok0:,}  range=[{np.nanmin(v0):.6g}, {np.nanmax(v0):.6g}]")
        print(f"    L1: valid={ok1:,}  range=[{np.nanmin(v1):.6g}, {np.nanmax(v1):.6g}]")
        print(f"    Changed: {ok0-ok1:+,} valid points")

        if ok0 > 0 and ok1 > 0:
            both_valid = np.isfinite(v0) & np.isfinite(v1)
            if np.sum(both_valid) > 0:
                max_diff = float(np.nanmax(np.abs(v0[both_valid] - v1[both_valid])))
                mean_diff = float(np.nanmean(np.abs(v0[both_valid] - v1[both_valid])))
                print(f"    Max diff: {max_diff:.6g}  Mean diff: {mean_diff:.6g}")

    ds0.close()
    ds1.close()
    return issues


def main():
    parser = argparse.ArgumentParser(description='L1 product diagnostics')
    parser.add_argument('--input', '-i', required=True, help='Path to L1 NetCDF')
    parser.add_argument('--l0', default=None, help='Path to L0 NetCDF for comparison')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    ds = xr.open_dataset(args.input)
    n = len(ds.time)
    print("=" * 70)
    print(f"  L1 PRODUCT VERIFICATION")
    print("=" * 70)
    print(f"  File: {args.input}")
    print(f"  Size: {os.path.getsize(args.input)/1024/1024:.1f} MB")
    print(f"  Time points: {n:,}")
    t0 = str(ds.time.values[0])[:19]
    t1 = str(ds.time.values[-1])[:19]
    print(f"  Time range: {t0} -> {t1}")

    all_issues = []
    all_issues.extend(check_gps(ds))
    all_issues.extend(check_ts(ds))
    all_issues.extend(check_qc(ds))
    all_issues.extend(compare_l0_l1(args.input, args.l0))

    ds.close()

    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    if not all_issues:
        print("\n  NO ISSUES FOUND — L1 product looks good.")
    else:
        print(f"\n  {len(all_issues)} ISSUE(S) FOUND:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")

    print()


if __name__ == '__main__':
    main()
