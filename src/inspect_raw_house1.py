"""
Raw data quality inspection for House 1.

Run from project root:
    python -m src.inspect_raw_house1
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
import numpy as np
import json
from src.config import RAW_DIR, TABLE_DIR, MAX_POWER_WATTS, MIN_POWER_WATTS

def main():
    print("=" * 70)
    print("  RAW House 1 — Data Quality Inspection")
    print("=" * 70)

    # ── Load both parts ──────────────────────────────────────────────────
    print("\n[1] Loading raw Part1 + Part2 ...")
    p1 = pd.read_csv(RAW_DIR / "RAW_House1_Part1.csv")
    p2 = pd.read_csv(RAW_DIR / "RAW_House1_Part2.csv")
    print(f"    Part1: {len(p1):,} rows  |  Part2: {len(p2):,} rows")

    # Parse timestamps
    p1["Time"] = pd.to_datetime(p1["Time"])
    p2["Time"] = pd.to_datetime(p2["Time"])

    print(f"    Part1 range: {p1['Time'].min()} -> {p1['Time'].max()}")
    print(f"    Part2 range: {p2['Time'].min()} -> {p2['Time'].max()}")

    # Check overlap
    overlap_start = max(p1["Time"].min(), p2["Time"].min())
    overlap_end = min(p1["Time"].max(), p2["Time"].max())
    if overlap_start < overlap_end:
        p1_overlap = p1[(p1["Time"] >= overlap_start) & (p1["Time"] <= overlap_end)]
        p2_overlap = p2[(p2["Time"] >= overlap_start) & (p2["Time"] <= overlap_end)]
        print(f"    Overlap: {overlap_start} -> {overlap_end}")
        print(f"    Part1 rows in overlap: {len(p1_overlap):,}")
        print(f"    Part2 rows in overlap: {len(p2_overlap):,}")

    # ── Concatenate ──────────────────────────────────────────────────────
    print("\n[2] Concatenating and sorting ...")
    df = pd.concat([p1, p2], ignore_index=True)
    df = df.sort_values("Time").reset_index(drop=True)
    print(f"    Combined: {len(df):,} rows")

    # ── Duplicate timestamps ─────────────────────────────────────────────
    print("\n[3] Duplicate timestamp analysis ...")
    dup_mask = df["Time"].duplicated(keep=False)
    n_dup_total = dup_mask.sum()
    n_dup_unique = df.loc[dup_mask, "Time"].nunique()
    print(f"    Rows with duplicate timestamps: {n_dup_total:,}")
    print(f"    Unique duplicate timestamps: {n_dup_unique:,}")
    if n_dup_unique > 0:
        # Show a few examples
        dup_times = df.loc[dup_mask, "Time"].unique()[:5]
        print(f"    First few duplicate timestamps:")
        for t in dup_times:
            rows = df[df["Time"] == t]
            print(f"      {t}: {len(rows)} rows")
            print(rows.iloc[:, :6].to_string(index=False, max_rows=4))

    # ── Timestamp ordering ───────────────────────────────────────────────
    print("\n[4] Timestamp ordering ...")
    time_sorted = df["Time"].is_monotonic_increasing
    print(f"    Monotonically increasing after sort: {time_sorted}")

    # ── Interval analysis ────────────────────────────────────────────────
    print("\n[5] Sampling interval analysis ...")
    dt = df["Time"].diff().dt.total_seconds()
    dt_valid = dt.dropna()
    print(f"    Median interval: {dt_valid.median():.1f}s")
    print(f"    Mean interval: {dt_valid.mean():.1f}s")
    print(f"    Min interval: {dt_valid.min():.1f}s")
    print(f"    Max interval: {dt_valid.max():.1f}s")
    print(f"    Std dev: {dt_valid.std():.1f}s")

    # Distribution of intervals
    bins = [0, 1, 5, 8, 10, 15, 30, 60, 300, 3600, 86400, float("inf")]
    labels = ["<1s", "1-5s", "5-8s", "8-10s", "10-15s", "15-30s",
              "30s-1m", "1-5m", "5m-1h", "1h-1d", ">1d"]
    interval_cats = pd.cut(dt_valid, bins=bins, labels=labels, right=True)
    interval_dist = interval_cats.value_counts().sort_index()
    print(f"\n    Interval distribution:")
    for cat, count in interval_dist.items():
        pct = count / len(dt_valid) * 100
        print(f"      {cat:>8s}: {count:>10,} ({pct:5.2f}%)")

    # Gaps > 1 hour
    large_gaps = dt_valid[dt_valid > 3600]
    print(f"\n    Gaps > 1 hour: {len(large_gaps)}")
    if len(large_gaps) > 0:
        gap_idx = large_gaps.index
        print(f"    Largest 10 gaps:")
        for i in large_gaps.nlargest(10).index:
            gap_sec = dt_valid[i]
            gap_hours = gap_sec / 3600
            t_before = df["Time"].iloc[i - 1]
            t_after = df["Time"].iloc[i]
            print(f"      {t_before} -> {t_after}  ({gap_hours:.1f} hours)")

    # ── Missing / NaN values ─────────────────────────────────────────────
    print("\n[6] Missing (NaN) values per column ...")
    power_cols = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    for c in power_cols:
        n_nan = df[c].isna().sum()
        pct = n_nan / len(df) * 100
        print(f"    {c:>12s}: {n_nan:>10,} NaN ({pct:5.2f}%)")

    # ── Extreme / impossible values ──────────────────────────────────────
    print(f"\n[7] Extreme values (negative or > {MAX_POWER_WATTS} W) ...")
    for c in power_cols:
        col = df[c].dropna()
        n_neg = (col < MIN_POWER_WATTS).sum()
        n_high = (col > MAX_POWER_WATTS).sum()
        vmin, vmax = col.min(), col.max()
        print(f"    {c:>12s}: min={vmin:>8.0f}  max={vmax:>8.0f}  "
              f"negative={n_neg:>6,}  >{MAX_POWER_WATTS}W={n_high:>6,}")

    # ── Zero-reading analysis ────────────────────────────────────────────
    print("\n[8] Zero-reading prevalence (may indicate sensor-off in Part1) ...")
    for c in power_cols:
        n_zero = (df[c] == 0).sum()
        pct = n_zero / len(df) * 100
        print(f"    {c:>12s}: {n_zero:>10,} zeros ({pct:5.2f}%)")

    # ── Basic statistics ─────────────────────────────────────────────────
    print("\n[9] Descriptive statistics (Aggregate + Appliance5=Washing Machine) ...")
    for c in ["Aggregate", "Appliance5"]:
        col = df[c].dropna()
        print(f"\n    {c}:")
        print(f"      count  = {len(col):,}")
        print(f"      mean   = {col.mean():.1f} W")
        print(f"      std    = {col.std():.1f} W")
        print(f"      median = {col.median():.1f} W")
        print(f"      p95    = {col.quantile(0.95):.1f} W")
        print(f"      p99    = {col.quantile(0.99):.1f} W")
        print(f"      max    = {col.max():.1f} W")

    # ── Save summary to JSON ─────────────────────────────────────────────
    summary = {
        "total_rows_combined": len(df),
        "part1_rows": len(p1),
        "part2_rows": len(p2),
        "duplicate_timestamp_rows": int(n_dup_total),
        "unique_duplicate_timestamps": int(n_dup_unique),
        "date_range_start": str(df["Time"].min()),
        "date_range_end": str(df["Time"].max()),
        "median_interval_sec": float(dt_valid.median()),
        "mean_interval_sec": round(float(dt_valid.mean()), 2),
        "gaps_over_1hr": int(len(large_gaps)),
        "interval_distribution": {str(k): int(v) for k, v in interval_dist.items()},
    }
    out_path = TABLE_DIR / "raw_house1_inspection.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[*] Summary saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

