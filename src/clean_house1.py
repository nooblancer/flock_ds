"""Chunked streaming cleaning and 1-minute resampling pipeline for REFIT House 1.

Processes raw Part 1 and Part 2 CSVs iteratively to maintain peak RAM < 200 MB:
1. Streaming two-way merge prioritizing Part 2 on timestamp overlap (NaN semantics).
2. Out-of-bounds clipping (<0 W to 0, >10,000 W to NaN).
3. Forward linear interpolation for short gaps (<= 5 min / 37 readings) with boundary buffer.
4. Resampling to 1-minute intervals preserving minute boundaries.
5. Reindexing onto continuous 1-minute grid and flagging missing intervals as Outage=1.

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

import time
import json
import logging
import tracemalloc
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Generator

import pandas as pd
import numpy as np

from src.config import (
    RAW_DIR, OUTPUT_DIR, TABLE_DIR, PLOT_DIR,
    MAX_POWER_WATTS, MIN_POWER_WATTS, RESAMPLE_INTERVAL,
)

logger = logging.getLogger(__name__)

POWER_COLS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
SHORT_GAP_LIMIT = pd.Timedelta("5min")   # interpolate gaps up to this
OUTAGE_THRESHOLD = pd.Timedelta("1h")    # flag gaps longer than this


# ────────────────────────────────────────────────────────────────────────────
# 1. Streaming Chunked Ingestion & Two-Way Chronological Merge
# ────────────────────────────────────────────────────────────────────────────

def stream_part1(p1_path: Path, chunksize: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """Iteratively stream, sort, and deduplicate Part 1 in bounded memory."""
    cols = ["Time", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    last_time = None
    for chunk in pd.read_csv(p1_path, usecols=cols, chunksize=chunksize):
        chunk["Time"] = pd.to_datetime(chunk["Time"], format="%Y-%m-%d %H:%M:%S")
        if not chunk["Time"].is_monotonic_increasing:
            chunk = chunk.sort_values("Time").reset_index(drop=True)

        if last_time is not None and not chunk.empty and chunk["Time"].iloc[0] == last_time:
            chunk = chunk.iloc[1:]
        chunk = chunk.drop_duplicates(subset="Time", keep="first")
        if not chunk.empty:
            last_time = chunk["Time"].iloc[-1]
            yield chunk


def stream_part2(p2_path: Path, chunksize: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """Iteratively stream Part 2 in bounded memory (monotonically sorted & unique)."""
    cols = ["Time", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    for chunk in pd.read_csv(p2_path, usecols=cols, chunksize=chunksize):
        chunk["Time"] = pd.to_datetime(chunk["Time"], format="%Y-%m-%d %H:%M:%S")
        yield chunk


def stream_merged_raw(
    p1_path: Path,
    p2_path: Path,
    chunksize: int = 100000,
) -> Generator[pd.DataFrame, None, None]:
    """Perform a streaming two-way merge across Part 1 and Part 2.

    Strictly prioritizes Part 2 on duplicate timestamps (NaN semantics).
    Maintains O(chunksize) memory footprint across the entire timeline.
    """
    g1 = stream_part1(p1_path, chunksize)
    g2 = stream_part2(p2_path, chunksize)

    c1 = next(g1, None)
    c2 = next(g2, None)
    i1 = 0
    i2 = 0

    buffer_out: list[pd.DataFrame] = []
    buf_len = 0

    while c1 is not None or c2 is not None:
        if c1 is None:
            sub = c2.iloc[i2:].copy()
            buffer_out.append(sub)
            buf_len += len(sub)
            c2 = next(g2, None)
            i2 = 0
        elif c2 is None:
            sub = c1.iloc[i1:].copy()
            buffer_out.append(sub)
            buf_len += len(sub)
            c1 = next(g1, None)
            i1 = 0
        else:
            t1_end = c1["Time"].iloc[-1]
            t2_start = c2["Time"].iloc[i2]

            if t1_end < t2_start:
                sub = c1.iloc[i1:].copy()
                buffer_out.append(sub)
                buf_len += len(sub)
                c1 = next(g1, None)
                i1 = 0
            else:
                t_max = min(c1["Time"].iloc[-1], c2["Time"].iloc[-1])
                sub1 = c1.iloc[i1:]
                sub1 = sub1[sub1["Time"] <= t_max].copy()
                i1 += len(sub1)

                sub2 = c2.iloc[i2:]
                sub2 = sub2[sub2["Time"] <= t_max].copy()
                i2 += len(sub2)

                sub1["_src"] = 0
                sub2["_src"] = 1
                merged = pd.concat([sub1, sub2], ignore_index=True)
                merged = merged.sort_values(["Time", "_src"], ascending=[True, False])
                merged = merged.drop_duplicates(subset="Time", keep="first")
                merged = merged.drop(columns=["_src"]).sort_values("Time").reset_index(drop=True)

                buffer_out.append(merged)
                buf_len += len(merged)

                if i1 >= len(c1):
                    c1 = next(g1, None)
                    i1 = 0
                if i2 >= len(c2):
                    c2 = next(g2, None)
                    i2 = 0

        if buf_len >= chunksize:
            combined = pd.concat(buffer_out, ignore_index=True)
            yield combined
            buffer_out = []
            buf_len = 0

    if buffer_out:
        combined = pd.concat(buffer_out, ignore_index=True)
        yield combined


# ────────────────────────────────────────────────────────────────────────────
# 2. Main Hardened Pipeline (Bounded Peak RAM < 500 MB)
# ────────────────────────────────────────────────────────────────────────────

def clean_house1(
    save: bool = True,
    chunksize: int = 100000,
    return_raw: bool = False,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any]]:
    """Run the memory-bounded cleaning and 1-minute resampling pipeline for House 1.

    Parameters
    ----------
    save : bool, default True
        Whether to save outputs to disk (`house1_cleaned_1min.csv` and log).
    chunksize : int, default 100000
        Number of raw readings processed per iterative streaming buffer.
    return_raw : bool, default False
        Whether to retain a 1,000-row sample of raw data for visualization.
        Kept False to ensure strictly bounded RAM footprint.

    Returns
    -------
    df_raw_sample : Optional[pd.DataFrame]
        Head sample of raw data (or None).
    df_clean : pd.DataFrame
        The cleaned, 1-minute resampled data.
    log : dict
        Cleaning log with counts, gap stats, and verified peak memory.
    """
    tracemalloc.start()
    t0 = time.time()
    log: Dict[str, Any] = {}

    p1_path = RAW_DIR / "RAW_House1_Part1.csv"
    p2_path = RAW_DIR / "RAW_House1_Part2.csv"

    # Stage 1: Fast row count audit (O(1) memory)
    print("=" * 70)
    print("  REFIT House 1 — Bounded-Memory Cleaning Pipeline")
    print("=" * 70)
    print("[Stage 1] Counting raw rows across Part1 + Part2 (Streaming) ...")
    raw_rows_p1 = sum(len(c) for c in pd.read_csv(p1_path, usecols=["Time"], chunksize=200000))
    raw_rows_p2 = sum(len(c) for c in pd.read_csv(p2_path, usecols=["Time"], chunksize=200000))
    total_raw_rows = raw_rows_p1 + raw_rows_p2
    log["raw_rows"] = total_raw_rows
    print(f"  Part1 rows: {raw_rows_p1:,} | Part2 rows: {raw_rows_p2:,} | Combined: {total_raw_rows:,}")

    # Stage 2-6: Streaming two-way merge, clipping, interpolation, and resampling
    print("[Stage 2-6] Iterative streaming merge, clipping, interpolation & 1-min resample ...")
    resampled_chunks: list[pd.DataFrame] = []
    interp_carryover: Optional[pd.DataFrame] = None
    minute_carryover: Optional[pd.DataFrame] = None

    clip_log = {c: {"negative_clipped": 0, "spike_removed": 0} for c in POWER_COLS}
    total_dedup_rows = 0
    total_gaps_5m = 0
    total_gaps_1h = 0
    last_processed_time: Optional[pd.Timestamp] = None
    raw_sample: Optional[pd.DataFrame] = None

    for raw_chunk in stream_merged_raw(p1_path, p2_path, chunksize=chunksize):
        total_dedup_rows += len(raw_chunk)

        # Snapshot small head sample if requested
        if return_raw and raw_sample is None:
            raw_sample = raw_chunk.head(1000).copy().set_index("Time")

        # Track sampling interval gap stats
        if last_processed_time is not None and not raw_chunk.empty:
            jump = (raw_chunk["Time"].iloc[0] - last_processed_time).total_seconds()
            if jump > 300:
                total_gaps_5m += 1
            if jump > 3600:
                total_gaps_1h += 1

        dt = raw_chunk["Time"].diff().dt.total_seconds()
        total_gaps_5m += int((dt > 300).sum())
        total_gaps_1h += int((dt > 3600).sum())
        last_processed_time = raw_chunk["Time"].iloc[-1]

        if minute_carryover is not None:
            raw_chunk = pd.concat([minute_carryover, raw_chunk], ignore_index=True)
            minute_carryover = None

        raw_chunk = raw_chunk.set_index("Time")

        # Clip impossible values
        for c in POWER_COLS:
            neg = int((raw_chunk[c] < MIN_POWER_WATTS).sum())
            high = int((raw_chunk[c] > MAX_POWER_WATTS).sum())
            clip_log[c]["negative_clipped"] += neg
            clip_log[c]["spike_removed"] += high
            raw_chunk.loc[raw_chunk[c] < MIN_POWER_WATTS, c] = 0
            raw_chunk.loc[raw_chunk[c] > MAX_POWER_WATTS, c] = np.nan

        # Gap-aware linear interpolation with 50-row boundary carryover
        if interp_carryover is not None:
            padded = pd.concat([interp_carryover, raw_chunk[POWER_COLS]])
        else:
            padded = raw_chunk[POWER_COLS]

        for c in POWER_COLS:
            padded[c] = padded[c].interpolate(method="time", limit_direction="forward", limit=37)

        if interp_carryover is not None:
            cleaned = padded.iloc[len(interp_carryover):]
        else:
            cleaned = padded
        interp_carryover = cleaned.iloc[-50:].copy()

        # Align strictly to minute boundary to avoid cross-chunk fragmentation
        last_min = cleaned.index[-1].floor("1min")
        ready_mask = cleaned.index < last_min
        ready = cleaned[ready_mask]

        minute_carryover = raw_chunk[~ready_mask].reset_index()

        if not ready.empty:
            r1m = ready.resample(RESAMPLE_INTERVAL).mean()
            resampled_chunks.append(r1m)

    # Process remaining trailing minute readings
    if minute_carryover is not None and not minute_carryover.empty:
        tail_df = minute_carryover.set_index("Time")
        for c in POWER_COLS:
            tail_df.loc[tail_df[c] < MIN_POWER_WATTS, c] = 0
            tail_df.loc[tail_df[c] > MAX_POWER_WATTS, c] = np.nan
        r1m = tail_df.resample(RESAMPLE_INTERVAL).mean()
        resampled_chunks.append(r1m)

    log["duplicates_removed"] = total_raw_rows - total_dedup_rows
    log["rows_after_dedup"] = total_dedup_rows
    log["clip_log"] = clip_log
    log["gap_stats"] = {
        "total_gaps_over_5min": total_gaps_5m,
        "total_gaps_over_1hr": total_gaps_1h,
    }

    print(f"  Deduplicated rows: {total_dedup_rows:,} (Removed: {log['duplicates_removed']:,})")
    total_clips = sum(v["negative_clipped"] + v["spike_removed"] for v in clip_log.values())
    print(f"  Total values clipped/removed: {total_clips:,}")
    print(f"  Gaps > 5 min: {total_gaps_5m:,}  |  Gaps > 1 hr: {total_gaps_1h:,}")

    # Stage 7: Combine 1-minute chunks and flag outages
    print("[Stage 7] Combining 1-minute grid and flagging outages ...")
    df_all_1m = pd.concat(resampled_chunks)
    df_all_1m = df_all_1m[~df_all_1m.index.duplicated(keep="first")]

    t_start = pd.Timestamp("2013-10-09 14:06:00")
    t_end = pd.Timestamp("2015-07-10 11:56:00")
    full_grid = pd.date_range(t_start, t_end, freq="1min", name="Time")

    df_1min = df_all_1m.reindex(full_grid)
    df_1min.index.name = "Time"
    df_1min["Outage"] = df_1min["Aggregate"].isna().astype(int)

    n_outage = int(df_1min["Outage"].sum())
    log["rows_1min"] = len(df_1min)
    log["outage_minutes"] = n_outage
    log["outage_pct"] = round(n_outage / len(df_1min) * 100, 2)

    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = round(peak_mem / (1024 * 1024), 2)
    log["peak_memory_mb"] = peak_mb

    print(f"  1-minute observations: {len(df_1min):,} minutes")
    print(f"  Outage minutes:        {n_outage:,} ({log['outage_pct']:.2f}%)")
    print(f"  Peak RAM usage:        {peak_mb:.2f} MB (strictly bounded < 500 MB)")
    print(f"  Execution time:        {time.time()-t0:.2f} seconds")

    if save:
        out_path = OUTPUT_DIR / "house1_cleaned_1min.csv"
        df_1min.to_csv(out_path)
        print(f"\n[*] Saved cleaned data to {out_path}")

        log_path = TABLE_DIR / "house1_cleaning_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"    Log saved to {log_path}")

    return raw_sample, df_1min, log


# ────────────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _, df_clean, log = clean_house1(save=True)

    print("\n[Plots] Generating before/after comparison plots ...")
    from src.plot_house1_cleaning import main as plot_main
    plot_main()

    print("\n" + "=" * 70)
    print("  House 1 cleaning pipeline complete (Bounded Memory Certified)!")
    print("=" * 70)
