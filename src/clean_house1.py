"""
Raw-data cleaning pipeline for REFIT House 1.

Pipeline stages (in order):
  1. Load & concatenate Part1 + Part2
  2. Sort by timestamp
  3. Remove duplicate timestamps (keep Part2 row where available — NaN semantics)
  4. Clip physically impossible values (negative → 0, >10 kW → NaN)
  5. Handle missing values:
     - Short gaps (≤ 5 minutes): linear interpolation
     - Longer gaps: left as NaN (flagged, not hallucinated)
  6. Resample to 1-minute intervals (mean aggregation)
  7. Post-resample gap flagging

Run standalone:
    python -m src.clean_house1
"""
from __future__ import annotations

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    RAW_DIR, OUTPUT_DIR, TABLE_DIR, PLOT_DIR,
    MAX_POWER_WATTS, MIN_POWER_WATTS, RESAMPLE_INTERVAL,
)

logger = logging.getLogger(__name__)

POWER_COLS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
SHORT_GAP_LIMIT = pd.Timedelta("5min")   # interpolate gaps up to this
OUTAGE_THRESHOLD = pd.Timedelta("1h")    # flag gaps longer than this


# ────────────────────────────────────────────────────────────────────────────
# Stage functions
# ────────────────────────────────────────────────────────────────────────────

def _load_and_concat() -> pd.DataFrame:
    """Stage 1: Load Part1 + Part2 and concatenate."""
    p1 = pd.read_csv(RAW_DIR / "RAW_House1_Part1.csv")
    p2 = pd.read_csv(RAW_DIR / "RAW_House1_Part2.csv")

    # Mark provenance so we can prefer Part2 in dedup
    p1["_source"] = "part1"
    p2["_source"] = "part2"

    df = pd.concat([p1, p2], ignore_index=True)
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    return df


def _sort_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Sort by timestamp."""
    return df.sort_values("Time").reset_index(drop=True)


def _deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Stage 3: Remove duplicate timestamps.

    Rule: When the same timestamp appears in both Part1 and Part2, keep the
    Part2 row (it uses NaN instead of ambiguous 0 for missing sensors).
    If both come from the same part, keep the first occurrence.

    Returns (deduplicated_df, n_removed).
    """
    n_before = len(df)
    # Sort so part2 comes first (we'll keep='first' after sorting)
    df = df.sort_values(["Time", "_source"], ascending=[True, False])
    df = df.drop_duplicates(subset="Time", keep="first")
    df = df.sort_values("Time").reset_index(drop=True)
    n_removed = n_before - len(df)
    return df, n_removed


def _clip_impossible(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 4: Clip physically impossible values.

    - Negative readings → 0 (sensor noise)
    - Readings > MAX_POWER_WATTS → NaN (likely spike / sensor error)
    """
    clip_log: dict[str, dict[str, int]] = {}
    for c in POWER_COLS:
        neg_count = int((df[c] < MIN_POWER_WATTS).sum())
        high_count = int((df[c] > MAX_POWER_WATTS).sum())
        df.loc[df[c] < MIN_POWER_WATTS, c] = 0
        df.loc[df[c] > MAX_POWER_WATTS, c] = np.nan
        clip_log[c] = {"negative_clipped": neg_count, "spike_removed": high_count}
    return df, clip_log


def _handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 5: Handle missing values with gap-aware interpolation.

    Short gaps (<=5 min): linear interpolation.
    Long gaps: left as NaN — these are real outages, not to be fabricated.
    """
    df = df.set_index("Time")
    # Drop helper columns
    for col in ["Unix", "_source"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Identify gap regions
    dt = df.index.to_series().diff()
    long_gap_mask = dt > SHORT_GAP_LIMIT

    gap_stats = {
        "total_gaps_over_5min": int(long_gap_mask.sum()),
        "total_gaps_over_1hr": int((dt > OUTAGE_THRESHOLD).sum()),
    }

    # Interpolate only within short-gap regions
    # We create a mask of rows that are within a long gap
    # and protect them from interpolation
    for c in POWER_COLS:
        if c in df.columns:
            # Only interpolate where the gap before the NaN block is short
            df[c] = df[c].interpolate(method="time", limit_direction="forward",
                                       limit=37)  # 5min / 8s ~ 37 readings

    return df, gap_stats


def _resample_1min(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 6: Resample to 1-minute intervals using mean aggregation.

    Mean is appropriate here because we're aggregating ~7-8 readings per
    minute, each representing instantaneous power. The mean gives average
    power over the minute, which is proportional to energy consumed.
    """
    cols = [c for c in POWER_COLS if c in df.columns]
    df_1min = df[cols].resample(RESAMPLE_INTERVAL).mean()
    return df_1min


def _flag_outages(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 7: Add an 'Outage' column flagging minutes within long data gaps."""
    # After resampling, NaN minutes in Aggregate indicate no data
    df["Outage"] = df["Aggregate"].isna().astype(int)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def clean_house1(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full cleaning pipeline for House 1.

    Returns
    -------
    df_raw : pd.DataFrame
        The raw combined data (timestamp-indexed, after sort + dedup only)
        for before/after comparison.
    df_clean : pd.DataFrame
        The cleaned, 1-minute resampled data.
    log : dict
        Cleaning log with counts at each stage.
    """
    log: dict = {}

    # Stage 1: Load
    print("[Stage 1] Loading Part1 + Part2 ...")
    df = _load_and_concat()
    log["raw_rows"] = len(df)
    print(f"  Combined rows: {len(df):,}")

    # Stage 2: Sort
    print("[Stage 2] Sorting by timestamp ...")
    df = _sort_timestamps(df)

    # Stage 3: Deduplicate
    print("[Stage 3] Removing duplicate timestamps ...")
    df, n_removed = _deduplicate(df)
    log["duplicates_removed"] = n_removed
    log["rows_after_dedup"] = len(df)
    print(f"  Removed: {n_removed:,}  |  Remaining: {len(df):,}")

    # Stage 4: Clip
    print("[Stage 4] Clipping impossible values ...")
    df, clip_log = _clip_impossible(df)
    log["clip_log"] = clip_log
    total_clips = sum(v["negative_clipped"] + v["spike_removed"]
                      for v in clip_log.values())
    print(f"  Total values clipped/removed: {total_clips:,}")

    # Stage 5: Handle missing (also sets Time as index)
    print("[Stage 5] Gap-aware missing-value handling ...")
    df, gap_stats = _handle_missing(df)
    log["gap_stats"] = gap_stats
    print(f"  Gaps > 5 min: {gap_stats['total_gaps_over_5min']}")
    print(f"  Gaps > 1 hr:  {gap_stats['total_gaps_over_1hr']}")

    # Snapshot for before/after comparison
    df_before_resample = df.copy()

    # Count NaNs before resample
    nan_before = {c: int(df[c].isna().sum()) for c in POWER_COLS if c in df.columns}
    log["nan_after_cleaning"] = nan_before
    print(f"  NaN remaining (Aggregate): {nan_before.get('Aggregate', 0):,}")

    # Stage 6: Resample
    print("[Stage 6] Resampling to 1-minute intervals ...")
    df_1min = _resample_1min(df)
    log["rows_1min"] = len(df_1min)
    print(f"  1-minute rows: {len(df_1min):,}")

    # Stage 7: Outage flagging
    print("[Stage 7] Flagging outage minutes ...")
    df_1min = _flag_outages(df_1min)
    n_outage = df_1min["Outage"].sum()
    log["outage_minutes"] = int(n_outage)
    log["outage_pct"] = round(n_outage / len(df_1min) * 100, 2)
    print(f"  Outage minutes: {n_outage:,} ({log['outage_pct']:.2f}%)")

    # Save
    if save:
        out_path = OUTPUT_DIR / "house1_cleaned_1min.csv"
        df_1min.to_csv(out_path)
        print(f"\n[*] Saved cleaned data to {out_path}")
        print(f"    Shape: {df_1min.shape}")
        print(f"    Range: {df_1min.index.min()} -> {df_1min.index.max()}")

        import json
        log_path = TABLE_DIR / "house1_cleaning_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"    Log saved to {log_path}")

    return df_before_resample, df_1min, log


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def plot_before_after(df_raw_indexed: pd.DataFrame, df_clean: pd.DataFrame):
    """Generate before/after comparison plots.

    Plots:
      - Aggregate power (full timeline + zoomed 48h)
      - Washing Machine (Appliance5) — zoomed to active period
      - Dishwasher (Appliance6) — zoomed to active period
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    channels = {
        "Aggregate": "Aggregate Power",
        "Appliance5": "Washing Machine (Appliance 5)",
        "Appliance6": "Dishwasher (Appliance 6)",
    }

    # ── Full timeline comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(len(channels), 2, figsize=(18, 4 * len(channels)),
                             constrained_layout=True)
    fig.suptitle("House 1 — Raw vs Cleaned (1-min) Comparison", fontsize=14, y=1.01)

    for i, (col, title) in enumerate(channels.items()):
        # Raw (left)
        ax_raw = axes[i, 0]
        raw_series = df_raw_indexed[col].dropna()
        # Downsample for plotting (every 60th point ~ 8 min)
        raw_plot = raw_series.iloc[::60]
        ax_raw.plot(raw_plot.index, raw_plot.values, linewidth=0.3,
                    color="C0", alpha=0.7)
        ax_raw.set_title(f"{title} — RAW (~8s)")
        ax_raw.set_ylabel("Power (W)")
        ax_raw.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_raw.tick_params(axis="x", rotation=30)

        # Cleaned (right)
        ax_clean = axes[i, 1]
        clean_series = df_clean[col].dropna()
        clean_plot = clean_series.iloc[::10]
        ax_clean.plot(clean_plot.index, clean_plot.values, linewidth=0.3,
                      color="C1", alpha=0.7)
        ax_clean.set_title(f"{title} — CLEANED (1-min)")
        ax_clean.set_ylabel("Power (W)")
        ax_clean.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_clean.tick_params(axis="x", rotation=30)

    path1 = PLOT_DIR / "house1_raw_vs_clean_full.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Zoomed 48-hour comparison ────────────────────────────────────────
    # Pick a 48h window with interesting activity
    # Find a day where the washing machine is active
    wm_daily = df_clean["Appliance5"].resample("D").max()
    active_days = wm_daily[wm_daily > 50].index
    if len(active_days) > 0:
        zoom_start = active_days[len(active_days) // 3]  # pick one from first third
    else:
        zoom_start = df_clean.index[len(df_clean) // 4]
    zoom_end = zoom_start + pd.Timedelta("48h")

    fig2, axes2 = plt.subplots(len(channels), 2, figsize=(18, 4 * len(channels)),
                               constrained_layout=True)
    fig2.suptitle(f"House 1 — Zoomed 48h: {zoom_start.date()} to {zoom_end.date()}",
                  fontsize=14, y=1.01)

    for i, (col, title) in enumerate(channels.items()):
        # Raw
        ax_raw = axes2[i, 0]
        raw_zoom = df_raw_indexed.loc[zoom_start:zoom_end, col].dropna()
        ax_raw.plot(raw_zoom.index, raw_zoom.values, linewidth=0.5,
                    color="C0", alpha=0.8)
        ax_raw.set_title(f"{title} — RAW (~8s)")
        ax_raw.set_ylabel("Power (W)")
        ax_raw.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax_raw.tick_params(axis="x", rotation=30)

        # Cleaned
        ax_clean = axes2[i, 1]
        clean_zoom = df_clean.loc[zoom_start:zoom_end, col].dropna()
        ax_clean.plot(clean_zoom.index, clean_zoom.values, linewidth=0.5,
                      color="C1", alpha=0.8)
        ax_clean.set_title(f"{title} — CLEANED (1-min)")
        ax_clean.set_ylabel("Power (W)")
        ax_clean.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        ax_clean.tick_params(axis="x", rotation=30)

    path2 = PLOT_DIR / "house1_raw_vs_clean_48h.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Data quality overview plot ───────────────────────────────────────
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)
    fig3.suptitle("House 1 — Data Quality Overview", fontsize=14)

    # Interval histogram (raw)
    dt = df_raw_indexed.index.to_series().diff().dt.total_seconds().dropna()
    dt_normal = dt[dt <= 30]  # focus on normal range
    ax1.hist(dt_normal, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax1.set_xlabel("Sampling Interval (seconds)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Raw Sampling Intervals (<=30s)")
    ax1.axvline(8, color="red", linestyle="--", linewidth=1, label="Expected 8s")
    ax1.legend()

    # Outage timeline
    outage_daily = df_clean["Outage"].resample("D").mean() * 100
    ax2.bar(outage_daily.index, outage_daily.values, width=1,
            color="salmon", edgecolor="none", alpha=0.8)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("% Minutes with Missing Data")
    ax2.set_title("Daily Data Availability Gaps (Cleaned)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.tick_params(axis="x", rotation=30)

    path3 = PLOT_DIR / "house1_data_quality.png"
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw, df_clean, log = clean_house1(save=True)

    print("\n[Plots] Generating before/after comparison plots ...")
    from src.plot_house1_cleaning import main as plot_main
    plot_main()

    print("\n" + "=" * 70)
    print("  House 1 cleaning pipeline complete!")
    print("=" * 70)

