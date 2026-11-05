"""
Data ingestion utilities for the REFIT Smart-Home dataset.

This module provides functions to:
  1. Load raw CSV data for a given house (concatenating Part1 + Part2).
  2. Load the official cleaned CSV data for a given house.
  3. Identify the washing-machine channel for any house.
  4. Return a quick summary (shape, date range, missing %) for any loaded frame.

The REFIT dataset is described in:
  Murray, Liao, Stankovic et al. (2015), "A data management platform for
  personalised real-time energy feedback", EEDAL 2015.

Data characteristics
--------------------
* **Unit**: Active power in Watts.
* **Raw sampling**: ~8-second polling interval (6-8 s, sensors not synchronised).
* **Cleaned dataset**: Same resolution, but daylight-saving corrected, IAM columns
  re-aligned after sensor swaps, NaN forward-filled, spikes > 4 kW zeroed, and
  an ``Issues`` column flagging rows where ΣIAMs > Aggregate.
* **Coverage**: 20 UK households, Oct 2013 – Jul 2015 (House 14 skipped).
* **Known gaps**: In particular Feb 2014; causes include internet/hardware failures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    RAW_DIR,
    CLEAN_DIR,
    APPLIANCE_MAP,
    WASHING_MACHINE_CHANNEL,
    MAX_POWER_WATTS,
    MIN_POWER_WATTS,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'Time' column to datetime and set as index."""
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=False)
    df = df.set_index("Time")
    # Drop the redundant Unix column
    if "Unix" in df.columns:
        df = df.drop(columns=["Unix"])
    return df


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def load_raw_house(
    house_id: int,
    parts: str = "both",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw REFIT data for a single house.

    Parameters
    ----------
    house_id : int
        House number (1-21, skipping 14).
    parts : {"both", "part1", "part2"}
        Which file(s) to load.  *Part1* uses 0 for unavailable sensors;
        *Part2* uses NaN.  ``"both"`` concatenates them, preferring NaN
        semantics where they overlap.
    nrows : int, optional
        Limit rows per file (useful for quick inspection).

    Returns
    -------
    pd.DataFrame
        DatetimeIndex='Time'; columns = Aggregate, Appliance1-9.
    """
    frames: list[pd.DataFrame] = []

    # Raw filenames are inconsistent — houses 1-12 use "House1", 13+ use "House_13"
    def _raw_path(part: int) -> Path:
        # Try without underscore first (House1_Part1), then with (House_13_Part1)
        p = RAW_DIR / f"RAW_House{house_id}_Part{part}.csv"
        if p.exists():
            return p
        p = RAW_DIR / f"RAW_House_{house_id}_Part{part}.csv"
        if p.exists():
            return p
        raise FileNotFoundError(
            f"Cannot find raw data for House {house_id} Part {part} in {RAW_DIR}"
        )

    if parts in ("both", "part1"):
        p1 = pd.read_csv(_raw_path(1), nrows=nrows)
        frames.append(p1)
        logger.info("Loaded RAW Part1 for House %d: %s rows", house_id, len(p1))

    if parts in ("both", "part2"):
        p2 = pd.read_csv(_raw_path(2), nrows=nrows)
        frames.append(p2)
        logger.info("Loaded RAW Part2 for House %d: %s rows", house_id, len(p2))

    df = pd.concat(frames, ignore_index=True)
    df = _parse_timestamps(df)
    df = df.sort_index()

    # Drop exact duplicate timestamps (keep first occurrence)
    df = df[~df.index.duplicated(keep="first")]

    return df


def load_clean_house(
    house_id: int,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the official cleaned REFIT CSV for one house.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex='Time'; columns = Aggregate, Appliance1-9, Issues.
    """
    path = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")

    df = pd.read_csv(path, nrows=nrows)
    logger.info("Loaded CLEAN House %d: %s rows", house_id, len(df))
    df = _parse_timestamps(df)
    df = df.sort_index()
    return df


def get_washing_machine_col(house_id: int) -> str:
    """Return the column name for the washing-machine IAM in a given house.

    Raises
    ------
    KeyError
        If the house has no known washing-machine channel (e.g., House 12).
    """
    if house_id not in WASHING_MACHINE_CHANNEL:
        raise KeyError(
            f"House {house_id} has no mapped washing-machine channel. "
            f"Available houses: {sorted(WASHING_MACHINE_CHANNEL)}"
        )
    return WASHING_MACHINE_CHANNEL[house_id]


def rename_appliances(df: pd.DataFrame, house_id: int) -> pd.DataFrame:
    """Rename generic Appliance* columns to human-readable names."""
    mapping = APPLIANCE_MAP.get(house_id, {})
    return df.rename(columns=mapping)


def data_summary(df: pd.DataFrame, house_id: int) -> dict:
    """Return a quick-look summary dictionary for a loaded house DataFrame.

    Keys: house_id, n_rows, start, end, duration_days, sampling_median_sec,
    missing_pct (per column), negative_count, above_max_count.
    """
    dt_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    power_cols = [c for c in df.columns if c not in ("Issues",)]

    return {
        "house_id": house_id,
        "n_rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "duration_days": (df.index.max() - df.index.min()).days,
        "median_interval_sec": round(dt_diffs.median(), 2),
        "mean_interval_sec": round(dt_diffs.mean(), 2),
        "missing_pct": {
            c: round(df[c].isna().mean() * 100, 2) for c in power_cols
        },
        "negative_count": {
            c: int((df[c] < MIN_POWER_WATTS).sum()) for c in power_cols
        },
        "above_max_count": {
            c: int((df[c] > MAX_POWER_WATTS).sum()) for c in power_cols
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Convenience: load multiple cleaned houses for cross-house EDA
# ────────────────────────────────────────────────────────────────────────────

def load_clean_houses(
    house_ids: list[int],
    nrows: Optional[int] = None,
) -> dict[int, pd.DataFrame]:
    """Load cleaned data for multiple houses, returned as {house_id: df}."""
    return {h: load_clean_house(h, nrows=nrows) for h in house_ids}

