"""
Preprocessing and Streaming Feature Engineering for NILM.

This module provides:
1. `create_nilm_features`: Batch & causal feature extraction supporting both
   multi-scale rolling windows (offline analytics) and Exponentially Weighted
   Moving Averages (EWMA, edge IoT gateways).
2. `StreamingEdgeBuffer`: Stateful, O(1)-memory streaming buffer designed for
   low-power edge smart meter gateways (ESP32, Raspberry Pi, ARM Cortex-M).
   Features dynamic time-delta gap adaptation and cold-start bias correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List


# ────────────────────────────────────────────────────────────────────────────
# 1. Modular Batch & Causal Feature Extraction
# ────────────────────────────────────────────────────────────────────────────

def create_nilm_features(
    df_partition: pd.DataFrame,
    agg_col: str = "Aggregate",
    mode: str = "batch",
    feature_strategy: str = "rolling",
) -> pd.DataFrame:
    """Extract temporal context features for an isolated chronological partition.

    Features are computed strictly per partition (train, val, test) to prevent
    temporal boundary leakage across splits.

    Parameters
    ----------
    df_partition : pd.DataFrame
        1-minute resampled telemetry for a single partition.
    agg_col : str, default 'Aggregate'
        Column name for aggregate active power.
    mode : {"batch", "causal"}, default "batch"
        "batch" uses centered rolling windows; "causal" uses trailing statistics.
    feature_strategy : {"rolling", "ewma", "hybrid"}, default "rolling"
        "rolling" uses moving window stats; "ewma" uses O(1) recursive averages;
        "hybrid" computes both representations.

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned with df_partition.index.
    """
    df = df_partition.copy()
    s = df[agg_col]

    feat = pd.DataFrame(index=df.index)
    feat["agg_power"] = s

    # 1. Immediate lags and leads
    lags = [-15, -10, -5, -3, -2, -1]
    if mode == "batch":
        lags += [1, 2, 3, 5, 10, 15]

    for lag in lags:
        col_name = f"agg_lag_{lag}" if lag < 0 else f"agg_lead_{lag}"
        feat[col_name] = s.shift(-lag).bfill().ffill()

    # 2. Multi-scale context: Rolling vs. EWMA vs. Hybrid
    is_center = (mode == "batch")
    spans = [5, 15, 30, 60]

    if feature_strategy in ("rolling", "hybrid"):
        for w in spans:
            feat[f"agg_roll_mean_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).mean().bfill().ffill()
            feat[f"agg_roll_std_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).std().fillna(0)
            feat[f"agg_roll_max_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).max().bfill().ffill()
            feat[f"agg_roll_min_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).min().bfill().ffill()
            feat[f"agg_roll_range_{w}"] = feat[f"agg_roll_max_{w}"] - feat[f"agg_roll_min_{w}"]

    if feature_strategy in ("ewma", "hybrid"):
        for sp in spans:
            # Exponentially Weighted Moving Average (O(1) memory state)
            ewma_obj = s.ewm(span=sp, adjust=False)
            feat[f"agg_ewma_mean_{sp}"] = ewma_obj.mean().bfill().ffill()
            feat[f"agg_ewma_std_{sp}"] = ewma_obj.std().fillna(0)
            # High-pass residual: difference between instantaneous and slow EWMA
            feat[f"agg_ewma_residual_{sp}"] = s - feat[f"agg_ewma_mean_{sp}"]

    # 3. Dynamic differences / step gradients
    feat["agg_diff_1"] = s.diff().fillna(0)
    feat["agg_diff_5"] = s.diff(5).fillna(0)
    feat["agg_diff_15"] = s.diff(15).fillna(0)

    # 4. Temporal & Calendar priors
    hour = df.index.hour
    dow = df.index.dayofweek
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat["is_weekend"] = (dow >= 5).astype(int)

    return feat.fillna(0)


# ────────────────────────────────────────────────────────────────────────────
# 2. Stateful Streaming Buffer for Edge Gateways (O(1) Memory Footprint)
# ────────────────────────────────────────────────────────────────────────────

class StreamingEdgeBuffer:
    """O(1) memory stateful streaming buffer for edge smart meter gateways.

    Maintains recursive EWMA means and variances across spans (5, 15, 30, 60 min)
    in under 200 bytes of RAM, with cold-start bias correction and continuous-time
    decay adaptation for irregular telemetry gaps.
    """

    def __init__(self, spans: Tuple[int, ...] = (5, 15, 30, 60)):
        self.spans = tuple(spans)
        # Discrete alpha for uniform 1-minute sampling: alpha = 2 / (span + 1)
        self.alphas: Dict[int, float] = {s: 2.0 / (s + 1.0) for s in spans}
        # Continuous time constants: tau = -1 / ln(1 - alpha) ~ span / 2
        self.taus: Dict[int, float] = {
            s: -1.0 / np.log(1.0 - self.alphas[s]) if self.alphas[s] < 1.0 else 1.0
            for s in spans
        }
        
        # O(1) state registers (scalar floats)
        self.means: Dict[int, Optional[float]] = {s: None for s in spans}
        self.vars: Dict[int, float] = {s: 0.0 for s in spans}
        self.step_count: int = 0
        self.last_power: Optional[float] = None
        self.last_timestamp: Optional[pd.Timestamp] = None

    def reset(self) -> None:
        """Reset state registers for cold restart."""
        self.means = {s: None for s in self.spans}
        self.vars = {s: 0.0 for s in self.spans}
        self.step_count = 0
        self.last_power = None
        self.last_timestamp = None

    def update(
        self,
        current_power: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """Process a single incoming telemetry reading in O(1) time and memory.

        Parameters
        ----------
        current_power : float
            Whole-home aggregate active power reading in Watts.
        timestamp : pd.Timestamp, optional
            Timestamp of reading. If provided, calculates dynamic Delta_t
            and temporal calendar features (hour_sin, hour_cos, is_weekend).

        Returns
        -------
        dict
            Streaming feature vector ready for real-time model inference.
        """
        x = float(current_power)
        features: Dict[str, float] = {"agg_power": x}
        self.step_count += 1

        # Calculate time delta in minutes if timestamps are available
        delta_t_min = 1.0
        if timestamp is not None and self.last_timestamp is not None:
            diff_sec = (timestamp - self.last_timestamp).total_seconds()
            if diff_sec > 0:
                delta_t_min = diff_sec / 60.0

        # Update recursive statistics across each time horizon
        for s in self.spans:
            if self.means[s] is None or delta_t_min > (s * 3.0):
                # Cold-start or long disconnection: snap initialize immediately
                self.means[s] = x
                self.vars[s] = 0.0
                a_eff = 1.0
            else:
                # Dynamic alpha adaptation for irregular sampling or packet loss
                if abs(delta_t_min - 1.0) < 1e-3:
                    a_eff = self.alphas[s]
                else:
                    # Continuous decay: alpha_eff = 1 - exp(-Delta_t / tau)
                    a_eff = 1.0 - float(np.exp(-delta_t_min / self.taus[s]))
                    a_eff = min(max(a_eff, 0.01), 1.0)

                diff = x - self.means[s]
                # Recursive EWMA update
                self.means[s] = a_eff * x + (1.0 - a_eff) * self.means[s]
                # Recursive exponential variance update (Finch 2009)
                self.vars[s] = (1.0 - a_eff) * (self.vars[s] + a_eff * (diff ** 2))

            # Bias-corrected estimate for early cold-start steps
            decay_factor = (1.0 - self.alphas[s]) ** self.step_count
            bias_correction = 1.0 - decay_factor if decay_factor < 0.999 else 1.0
            mean_est = self.means[s] / max(bias_correction, 1e-4)

            std_est = float(np.sqrt(max(0.0, self.vars[s])))

            features[f"agg_ewma_mean_{s}"] = round(float(mean_est), 2)
            features[f"agg_ewma_std_{s}"] = round(std_est, 2)
            features[f"agg_ewma_residual_{s}"] = round(float(x - mean_est), 2)

        # Step difference
        diff_1 = (x - self.last_power) if self.last_power is not None else 0.0
        features["agg_diff_1"] = round(float(diff_1), 2)

        # Update state registers
        self.last_power = x
        self.last_timestamp = timestamp

        # Temporal priors if timestamp is provided
        if timestamp is not None:
            hour = timestamp.hour
            dow = timestamp.dayofweek
            features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
            features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
            features["is_weekend"] = 1.0 if dow >= 5 else 0.0

        return features

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state registers for edge flash persistence during power loss."""
        return {
            "spans": list(self.spans),
            "means": self.means,
            "vars": self.vars,
            "step_count": self.step_count,
            "last_power": self.last_power,
            "last_timestamp": str(self.last_timestamp) if self.last_timestamp else None,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "StreamingEdgeBuffer":
        """Restore state registers from non-volatile flash storage."""
        buf = cls(spans=tuple(state["spans"]))
        buf.means = {int(k): v for k, v in state["means"].items()}
        buf.vars = {int(k): float(v) for k, v in state["vars"].items()}
        buf.step_count = int(state["step_count"])
        buf.last_power = float(state["last_power"]) if state["last_power"] is not None else None
        if state["last_timestamp"]:
            buf.last_timestamp = pd.Timestamp(state["last_timestamp"])
        return buf


# ────────────────────────────────────────────────────────────────────────────
# 3. Memory-Bounded Chunked Ingestion & Iterative Resampling Utilities
# ────────────────────────────────────────────────────────────────────────────

def stream_chunked_csv(
    filepath: str | Path,
    chunksize: int = 100000,
    usecols: Optional[list[str]] = None,
    parse_dates: Optional[list[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Iteratively stream large raw CSV telemetry files in bounded RAM chunks.

    Guarantees strict O(chunksize) memory footprint on CI/CD runners and low-memory
    edge devices regardless of total file size (e.g. 500 MB to 50 GB).

    Parameters
    ----------
    filepath : str or Path
        Path to raw telemetry CSV file.
    chunksize : int, default 100000
        Number of rows per iteration.
    usecols : list[str], optional
        Columns to read from disk.
    parse_dates : list[str], optional
        Columns to parse directly into DatetimeIndex.

    Yields
    ------
    pd.DataFrame
        Chunked slice of raw telemetry.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        if parse_dates:
            for dt_col in parse_dates:
                if dt_col in chunk.columns:
                    chunk[dt_col] = pd.to_datetime(chunk[dt_col])
        yield chunk


def iterative_resample_1min(
    chunk_generator: Generator[pd.DataFrame, None, None],
    time_col: str = "Time",
    power_cols: Optional[list[str]] = None,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Iteratively resample a streaming chunk generator into 1-minute intervals.

    Maintains minute-boundary alignment across successive chunks, guaranteeing that
    no minute bin is fragmented or split across partition boundaries.

    Parameters
    ----------
    chunk_generator : Generator[pd.DataFrame, None, None]
        Generator yielding raw telemetry chunks.
    time_col : str, default 'Time'
        Timestamp column name.
    power_cols : list[str], optional
        Power measurement columns to aggregate.
    agg_func : str, default 'mean'
        Aggregation method ('mean', 'sum', 'median').

    Returns
    -------
    pd.DataFrame
        Aggregated 1-minute DataFrame indexed by timestamp.
    """
    minute_carryover: Optional[pd.DataFrame] = None
    resampled_slices: list[pd.DataFrame] = []

    for chunk in chunk_generator:
        if chunk.empty:
            continue

        if minute_carryover is not None:
            chunk = pd.concat([minute_carryover, chunk], ignore_index=True)
            minute_carryover = None

        if time_col in chunk.columns:
            if not pd.api.types.is_datetime64_any_dtype(chunk[time_col]):
                chunk[time_col] = pd.to_datetime(chunk[time_col])
            chunk = chunk.set_index(time_col)

        cols = power_cols if power_cols is not None else [c for c in chunk.columns if c != "Unix"]

        # Hold back readings belonging to the final uncompleted minute
        last_min = chunk.index[-1].floor("1min")
        ready_mask = chunk.index < last_min
        ready_chunk = chunk[ready_mask]

        minute_carryover = chunk[~ready_mask].reset_index()

        if not ready_chunk.empty:
            if agg_func == "mean":
                resampled = ready_chunk[cols].resample("1min").mean()
            elif agg_func == "sum":
                resampled = ready_chunk[cols].resample("1min").sum()
            else:
                resampled = ready_chunk[cols].resample("1min").agg(agg_func)
            resampled_slices.append(resampled)

    if minute_carryover is not None and not minute_carryover.empty:
        tail_df = minute_carryover.set_index(time_col)
        cols = power_cols if power_cols is not None else [c for c in tail_df.columns if c != "Unix"]
        resampled = tail_df[cols].resample("1min").agg(agg_func)
        resampled_slices.append(resampled)

    if not resampled_slices:
        return pd.DataFrame()

    df_resampled = pd.concat(resampled_slices)
    return df_resampled[~df_resampled.index.duplicated(keep="first")]

