"""
Appliance-Disaggregation (NILM) Pipeline for Washing Machine Power Estimation.

This module implements a rigorous, leak-free machine learning workflow:
1. Strict pre-split isolation: Splits chronological 1-minute data into Train (70%),
   Validation (10%), and Test (20%) BEFORE computing any rolling or lag features,
   guaranteeing zero temporal boundary leakage across partitions.
2. Feature engineering: Temporal lags, multi-scale rolling statistics, and calendar priors.
3. Dual-tier evaluation:
     - Sample-Level Point-wise Metrics: MAE, RMSE, RETE (Net Energy Error), Sample F1/Precision/Recall (P >= 15W).
     - Event-Based Macro Cycle Metrics: 1-to-1 bipartite cycle matching with 5-minute tolerance window.
4. Benchmark models: Always-Off (0W) baseline, Diurnal Usage Profile baseline, and XGBoost Regressor.
5. Diagnostic visual plots & failure mode analysis.
"""
from __future__ import annotations

import sys, os
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
from src.config import OUTPUT_DIR, PLOT_DIR, TABLE_DIR, CLEAN_DIR, WASHING_MACHINE_CHANNEL

cache_dir = OUTPUT_DIR / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)
os.environ["PYTHONIOENCODING"] = "utf-8"

import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score
import xgboost as xgb

from src.washing_machine_eda import detect_washing_cycles
from src.preprocessing import create_nilm_features, StreamingEdgeBuffer

logger = logging.getLogger(__name__)

ACTIVE_THRESHOLD_WATTS = 15.0


# ────────────────────────────────────────────────────────────────────────────
# 1. Leak-Free Feature Engineering (Imported from src.preprocessing)
# ────────────────────────────────────────────────────────────────────────────
# Note: create_nilm_features and StreamingEdgeBuffer are imported from src.preprocessing
# supporting multi-scale rolling windows (batch), EWMA (O(1) IoT edge), and hybrid modes.


# ────────────────────────────────────────────────────────────────────────────
# 2. Event-Based Macro Cycle Evaluation (Dynamic Matching & Tolerance)
# ────────────────────────────────────────────────────────────────────────────

def match_and_evaluate_events(
    true_cycles: list[dict],
    pred_cycles: list[dict],
    tolerance_min: int = 5,
) -> dict:
    """Evaluate event detection by matching predicted cycles to ground-truth cycles.

    Matches cycles using greedy assignment based on interval overlap and start-time
    proximity within `tolerance_min`.

    Parameters
    ----------
    true_cycles : list[dict]
        Ground-truth appliance cycles.
    pred_cycles : list[dict]
        Predicted appliance cycles.
    tolerance_min : int, default 5
        Start-time tolerance in minutes.

    Returns
    -------
    dict
        Precision, recall, F1, and count metrics for cycle events.
    """
    matched_true = set()
    matched_pred = set()

    for p_idx, p in enumerate(pred_cycles):
        best_t_idx = None
        best_score = -1

        for t_idx, t in enumerate(true_cycles):
            if t_idx in matched_true:
                continue

            # Calculate temporal overlap in minutes
            overlap_start = max(p["start_dt"], t["start_dt"])
            overlap_end = min(p["end_dt"], t["end_dt"])
            overlap_min = (overlap_end - overlap_start).total_seconds() / 60.0

            # Start time deviation in minutes
            start_diff_min = abs((p["start_dt"] - t["start_dt"]).total_seconds() / 60.0)

            if overlap_min > 0 or start_diff_min <= tolerance_min:
                score = max(overlap_min, 0) + (100.0 if start_diff_min <= tolerance_min else 0.0)
                if score > best_score:
                    best_score = score
                    best_t_idx = t_idx

        if best_t_idx is not None:
            matched_pred.add(p_idx)
            matched_true.add(best_t_idx)

    tp = len(matched_true)
    fp = len(pred_cycles) - len(matched_pred)
    fn = len(true_cycles) - len(matched_true)

    prec = tp / len(pred_cycles) if len(pred_cycles) > 0 else 0.0
    rec = tp / len(true_cycles) if len(true_cycles) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "true_events": len(true_cycles),
        "pred_events": len(pred_cycles),
        "tp_events": tp,
        "fp_events": fp,
        "fn_events": fn,
        "event_precision": round(float(prec), 4),
        "event_recall": round(float(rec), 4),
        "event_f1": round(float(f1), 4),
    }


# ────────────────────────────────────────────────────────────────────────────
# 3. Asymmetric Loss Objective (Resolving the Thresholding Paradox)
# ────────────────────────────────────────────────────────────────────────────

def asymmetric_mse_objective(
    under_penalty: float = 2.0,
    active_threshold: float = 500.0,
):
    """Custom asymmetric weighted MSE objective for XGBoost.

    Penalizes under-predictions during active peak cycles (>= active_threshold)
    more heavily than over-predictions during idle baseline periods:
        L(y, y_hat) = 1/2 * w(y, y_hat) * (y_hat - y)^2

    Parameters
    ----------
    under_penalty : float, default 2.0
        Loss multiplier applied when y >= active_threshold and y_hat < y.
    active_threshold : float, default 500.0
        Threshold in Watts separating high-power peaks from baseline load.
    """
    def custom_obj(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(y_true, "get_label"):
            y_t = y_true.get_label()
        else:
            y_t = y_true
        y_p = y_pred

        diff = y_p - y_t
        is_under_peak = (y_t >= active_threshold) & (diff < 0)
        is_under_active = (y_t >= ACTIVE_THRESHOLD_WATTS) & (y_t < active_threshold) & (diff < 0)

        weight = np.ones_like(diff, dtype=np.float32)
        weight = np.where(is_under_peak, float(under_penalty), weight)
        weight = np.where(is_under_active, 1.5, weight)

        grad = weight * diff
        hess = weight
        return grad, hess

    return custom_obj


# ────────────────────────────────────────────────────────────────────────────
# 4. Dual-Tier Evaluation Function with Residual Cancellation Diagnostics
# ────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_index: pd.DatetimeIndex,
    model_name: str,
    true_cycles: list[dict],
    active_threshold_watts: float = ACTIVE_THRESHOLD_WATTS,
    max_pause_min: int = 5,
    min_duration_min: int = 15,
    min_peak_watts: float = 500.0,
    tolerance_min: int = 5,
    cycle_stitch_method: str = "dynamic_bridge",
) -> dict:
    """Compute comprehensive sample-level, event-level, and energy cancellation metrics.

    Dynamic Cycle Stitching Parameters:
    ----------------------------------
    active_threshold_watts : float
        Minimum active power separating drum motion from standby.
    max_pause_min : int
        Maximum allowable pause between active readings (e.g. soaking phase).
    min_duration_min : int
        Minimum length in minutes to qualify as a valid wash cycle.
    min_peak_watts : float
        Minimum peak power required across the cycle (water heating confirmation).
    tolerance_min : int
        Temporal tolerance window for matching cycle start/end bounds.
    cycle_stitch_method : {"dynamic_bridge", "dbscan"}
        Algorithm for temporal grouping: vectorized bridge or 1D density clustering.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)

    # 1. Continuous Regression Metrics
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = root_mean_squared_error(y_true, y_pred_clipped)

    total_true_energy = np.sum(y_true)
    total_pred_energy = np.sum(y_pred_clipped)
    # Net Energy Error (Relative Error in Total Energy / RETE)
    rete = abs(total_pred_energy - total_true_energy) / (total_true_energy + 1e-6)

    # 2. Residual Cancellation Diagnostics (The Thresholding Paradox)
    # Measures the balance between idle noise bleed and active peak deficit
    idle_mask = (y_true < 5.0)
    peak_mask = (y_true >= 500.0)
    idle_bleed_kwh = float(np.sum(y_pred_clipped[idle_mask]) / 60.0 / 1000.0)
    peak_deficit_kwh = float(np.sum(np.maximum(0, y_true[peak_mask] - y_pred_clipped[peak_mask])) / 60.0 / 1000.0)
    energy_bias_kwh = float((total_pred_energy - total_true_energy) / 60.0 / 1000.0)

    # 3. Sample-Level Classification Metrics (P >= active_threshold_watts)
    y_true_bin = (y_true >= active_threshold_watts).astype(int)
    y_pred_bin = (y_pred_clipped >= active_threshold_watts).astype(int)

    sample_precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    sample_recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    sample_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # 4. Event-Level Macro Cycle Metrics (Dynamic Stitching & Tolerance)
    pred_s = pd.Series(y_pred_clipped, index=test_index)
    pred_cycles = detect_washing_cycles(
        pred_s,
        on_thresh=active_threshold_watts,
        min_duration=min_duration_min,
        max_pause=max_pause_min,
        min_peak=min_peak_watts,
        method=cycle_stitch_method,
    )
    event_metrics = match_and_evaluate_events(true_cycles, pred_cycles, tolerance_min=tolerance_min)

    return {
        "model": model_name,
        "mae_watts": round(float(mae), 2),
        "rmse_watts": round(float(rmse), 2),
        "rete_energy_pct": round(float(rete * 100), 2),
        "total_true_kwh": round(float(total_true_energy / 60.0 / 1000.0), 2),
        "total_pred_kwh": round(float(total_pred_energy / 60.0 / 1000.0), 2),
        "energy_bias_kwh": round(energy_bias_kwh, 2),
        "idle_bleed_kwh": round(idle_bleed_kwh, 2),
        "peak_deficit_kwh": round(peak_deficit_kwh, 2),
        "sample_precision": round(float(sample_precision), 4),
        "sample_recall": round(float(sample_recall), 4),
        "sample_f1": round(float(sample_f1), 4),
        "event_precision": event_metrics["event_precision"],
        "event_recall": event_metrics["event_recall"],
        "event_f1": event_metrics["event_f1"],
        "true_events": event_metrics["true_events"],
        "pred_events": event_metrics["pred_events"],
        "matched_tp_events": event_metrics["tp_events"],
    }


# ────────────────────────────────────────────────────────────────────────────
# 5. Modular Data Ingestion (Strict Per-Household Telemetry Loading)
# ────────────────────────────────────────────────────────────────────────────

def load_house_clean(house_id: int) -> tuple[pd.DataFrame, str]:
    """Load and resample 1-minute cleaned data for a specific REFIT household.

    Parameters
    ----------
    house_id : int
        REFIT household identifier (e.g., 1, 2, 5, 7, 20).

    Returns
    -------
    tuple[pd.DataFrame, str]
        (Cleaned 1-minute DataFrame indexed by timestamp, Washing machine column name)
    """
    fpath = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Cleaned dataset for House {house_id} not found at {fpath}")

    wm_col = WASHING_MACHINE_CHANNEL.get(house_id)
    if wm_col is None:
        raise KeyError(f"No washing machine channel configured for House {house_id} in WASHING_MACHINE_CHANNEL")

    df_raw = pd.read_csv(fpath, usecols=["Time", "Aggregate", wm_col])
    df_raw["Time"] = pd.to_datetime(df_raw["Time"], format="%Y-%m-%d %H:%M:%S")
    df_raw = df_raw.set_index("Time").sort_index()

    df_1min = df_raw.resample("1min").mean()
    valid_mask = df_1min["Aggregate"].notna() & df_1min[wm_col].notna()
    df_clean = df_1min[valid_mask].copy()
    return df_clean, wm_col


# ────────────────────────────────────────────────────────────────────────────
# 6. Single-Household Chronological Disaggregation Pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_nilm_pipeline(
    house_id: int = 1,
    active_threshold_watts: float = ACTIVE_THRESHOLD_WATTS,
    max_pause_min: int = 5,
    min_duration_min: int = 15,
    min_peak_watts: float = 500.0,
    tolerance_min: int = 5,
    cycle_stitch_method: str = "dynamic_bridge",
    under_penalty: float = 2.0,
    feature_strategy: str = "rolling",
):
    print("=" * 70, flush=True)
    print(f"  Appliance Disaggregation (NILM) Pipeline — House {house_id}", flush=True)
    print(f"  Stitching: {cycle_stitch_method} (pause={max_pause_min}m) | Feature Strategy: {feature_strategy}", flush=True)
    print("=" * 70, flush=True)

    # Load cleaned 1-min data
    print("[1] Loading 1-minute cleaned data ...", flush=True)
    df_clean, wm_col = load_house_clean(house_id)
    print(f"  Valid 1-minute observations: {len(df_clean):,} minutes "
          f"({df_clean.index.min().date()} to {df_clean.index.max().date()})", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # CRITICAL DATA LEAKAGE AUDIT:
    # Split raw series into Train (70%), Val (10%), Test (20%) BEFORE feature engineering!
    # ────────────────────────────────────────────────────────────────────────
    print("[2] Strict Chronological Splitting (Zero Boundary Leakage) ...", flush=True)
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    df_train = df_clean.iloc[:train_end].copy()
    df_val = df_clean.iloc[train_end:val_end].copy()
    df_test = df_clean.iloc[val_end:].copy()

    print(f"  Train partition:      {len(df_train):,} mins ({df_train.index[0].date()} -> {df_train.index[-1].date()})", flush=True)
    print(f"  Validation partition: {len(df_val):,} mins ({df_val.index[0].date()} -> {df_val.index[-1].date()})", flush=True)
    print(f"  Test partition:       {len(df_test):,} mins ({df_test.index[0].date()} -> {df_test.index[-1].date()})", flush=True)

    # Engineer features independently on each partition
    print(f"[3] Engineering within-partition context features ({feature_strategy}) ...", flush=True)
    X_train = create_nilm_features(df_train, "Aggregate", mode="batch", feature_strategy=feature_strategy)
    y_train = df_train[wm_col].values

    X_val = create_nilm_features(df_val, "Aggregate", mode="batch", feature_strategy=feature_strategy)
    y_val = df_val[wm_col].values

    X_test = create_nilm_features(df_test, "Aggregate", mode="batch", feature_strategy=feature_strategy)
    y_test = df_test[wm_col].values
    test_index = df_test.index

    # Pre-extract ground truth test cycles for macro event evaluation (dynamic parameters)
    true_test_s = pd.Series(y_test, index=test_index)
    true_cycles = detect_washing_cycles(
        true_test_s,
        on_thresh=active_threshold_watts,
        min_duration=min_duration_min,
        max_pause=max_pause_min,
        min_peak=min_peak_watts,
        method=cycle_stitch_method,
    )
    print(f"  Ground-truth washing cycles in test set: {len(true_cycles)} cycles "
          f"(method={cycle_stitch_method}, pause={max_pause_min}m, thresh={active_threshold_watts}W)", flush=True)
    print(f"  Active test minutes (>={active_threshold_watts}W): {(y_test >= active_threshold_watts).sum():,} "
          f"({(y_test >= active_threshold_watts).mean()*100:.2f}%)", flush=True)

    eval_kwargs = dict(
        active_threshold_watts=active_threshold_watts,
        max_pause_min=max_pause_min,
        min_duration_min=min_duration_min,
        min_peak_watts=min_peak_watts,
        tolerance_min=tolerance_min,
        cycle_stitch_method=cycle_stitch_method,
    )

    metrics_list = []

    # ── Baseline 1: Always-Off (0 W) ────────────────────────────────────
    print("\n[4] Evaluating Baseline 1: Always-Off (0 W) ...", flush=True)
    y_pred_base1 = np.zeros_like(y_test)
    m1 = evaluate_predictions(y_test, y_pred_base1, test_index, "Baseline 1: Always-Off (0W)", true_cycles, **eval_kwargs)
    metrics_list.append(m1)

    # ── Baseline 2: Diurnal Usage Profile ───────────────────────────────
    print("[5] Evaluating Baseline 2: Diurnal Usage Profile ...", flush=True)
    train_df = pd.DataFrame({"wm": y_train, "hour": df_train.index.hour, 
                             "is_weekend": (df_train.index.dayofweek >= 5).astype(int)})
    profile = train_df.groupby(["hour", "is_weekend"])["wm"].mean()

    test_keys = list(zip(test_index.hour, (test_index.dayofweek >= 5).astype(int)))
    y_pred_base2 = np.array([profile.get(k, 0.0) for k in test_keys])
    m2 = evaluate_predictions(y_test, y_pred_base2, test_index, "Baseline 2: Diurnal Profile", true_cycles, **eval_kwargs)
    metrics_list.append(m2)

    # ── Model 3: XGBoost Regressor (Symmetric MSE) ──────────────────────
    print("\n[6] Training Leak-Free XGBoost Regressor (Symmetric MSE) ...", flush=True)
    model_sym = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    model_sym.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    y_pred_sym_raw = np.clip(model_sym.predict(X_test), 0, None)
    y_pred_sym_thresh = y_pred_sym_raw.copy()
    y_pred_sym_thresh[y_pred_sym_thresh < 10.0] = 0.0

    m3 = evaluate_predictions(y_test, y_pred_sym_raw, test_index, "XGBoost (Symmetric MSE - Raw)", true_cycles, **eval_kwargs)
    m4 = evaluate_predictions(y_test, y_pred_sym_thresh, test_index, "XGBoost (Symmetric MSE - Thresholded)", true_cycles, **eval_kwargs)
    metrics_list.extend([m3, m4])

    # ── Model 4: XGBoost Regressor (Asymmetric Peak-Weighted Loss) ───────
    print("\n[7] Training Leak-Free XGBoost Regressor (Asymmetric Peak-Weighted Loss) ...", flush=True)
    asym_obj = asymmetric_mse_objective(under_penalty=under_penalty, active_threshold=min_peak_watts)
    model_asym = xgb.XGBRegressor(
        objective=asym_obj,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    model_asym.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    y_pred_asym_raw = np.clip(model_asym.predict(X_test), 0, None)
    y_pred_asym_thresh = y_pred_asym_raw.copy()
    y_pred_asym_thresh[y_pred_asym_thresh < 10.0] = 0.0

    m5 = evaluate_predictions(y_test, y_pred_asym_raw, test_index, "XGBoost (Asymmetric Loss - Raw)", true_cycles, **eval_kwargs)
    m6 = evaluate_predictions(y_test, y_pred_asym_thresh, test_index, "XGBoost (Asymmetric Loss - Thresholded)", true_cycles, **eval_kwargs)
    metrics_list.extend([m5, m6])

    # Default model for subsequent feature importance and failure mode plots
    model = model_asym
    y_pred_xgb_thresh = y_pred_asym_thresh

    # Save comprehensive metrics table
    metrics_df = pd.DataFrame(metrics_list)
    out_table = TABLE_DIR / "nilm_model_metrics.csv"
    if out_table.exists():
        existing_df = pd.read_csv(out_table)
        existing_df = existing_df[~existing_df["model"].isin(metrics_df["model"])]
        full_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    else:
        full_df = metrics_df
    full_df.to_csv(out_table, index=False)
    print(f"\n[*] Metrics table saved to {out_table}", flush=True)
    print("\n" + "=" * 115, flush=True)
    print("  COMPREHENSIVE MODEL EVALUATION BENCHMARK (DUAL-TIER METRICS & RESIDUAL CANCELLATION DIAGNOSTICS)", flush=True)
    print("=" * 115, flush=True)
    cols_display = ["model", "mae_watts", "rmse_watts", "rete_energy_pct", "idle_bleed_kwh", "peak_deficit_kwh",
                    "sample_f1", "event_precision", "event_recall", "event_f1", "matched_tp_events"]
    print(full_df[cols_display].to_string(index=False), flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1: Actual vs. Predicted on Evaluation Data (Zoomed 48h Window)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating Plot 1: Actual vs. Predicted Evaluation Plot ...", flush=True)
    test_wm_s = pd.Series(y_test, index=test_index)
    test_active_days = test_wm_s.resample("D").max()
    active_days = test_active_days[test_active_days > 1500].index

    if len(active_days) > 0:
        zoom_start = pd.Timestamp(active_days[len(active_days) // 2]).floor("D")
    else:
        zoom_start = test_index[len(test_index) // 2].floor("D")
    zoom_end = zoom_start + pd.Timedelta(hours=48)

    sub_mask = (test_index >= zoom_start) & (test_index <= zoom_end)
    t_sub = test_index[sub_mask]
    actual_sub = y_test[sub_mask]
    pred_asym_sub = y_pred_asym_thresh[sub_mask]
    pred_sym_sub = y_pred_sym_thresh[sub_mask]
    pred_base2_sub = y_pred_base2[sub_mask]
    agg_sub = X_test.loc[sub_mask, "agg_power"].values

    fig1, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig1.suptitle(f"REFIT House 1: Disaggregation Evaluation on Unseen Test Data (Leak-Free Split)\n"
                  f"48-Hour Trace ({zoom_start.strftime('%Y-%m-%d')} to {zoom_end.strftime('%Y-%m-%d')})",
                  fontsize=13, fontweight='bold', y=0.98)

    # Top: Aggregate mains power
    ax_top.plot(t_sub, agg_sub, color="#1f77b4", linewidth=1.1, alpha=0.85, label="Whole-House Aggregate")
    ax_top.set_ylabel("Aggregate (W)", fontsize=10)
    ax_top.set_title("Input Signal: Household Aggregate Active Power", fontsize=11, fontweight='semibold')
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="upper right", fontsize=9)

    # Middle: Actual vs Predicted Washing Machine Power
    ax_mid.plot(t_sub, actual_sub, color="#000000", linewidth=1.8, label="Actual Washing Machine (Ground Truth)")
    ax_mid.plot(t_sub, pred_asym_sub, color="#d62728", linewidth=1.6, alpha=0.95, 
                label="XGBoost (Asymmetric Peak Loss - Thresholded)")
    ax_mid.plot(t_sub, pred_sym_sub, color="#ff7f0e", linewidth=1.3, alpha=0.85, linestyle="--", 
                label="XGBoost (Symmetric MSE - Thresholded)")
    ax_mid.plot(t_sub, pred_base2_sub, color="#7f7f7f", linewidth=1.0, alpha=0.6, linestyle=":", 
                label="Baseline 2: Diurnal Profile")
    ax_mid.set_ylabel("Power (Watts)", fontsize=10)
    ax_mid.set_title("Target Appliance Disaggregation: Actual vs. Symmetric & Asymmetric Model Predictions", fontsize=11, fontweight='semibold')
    ax_mid.grid(True, linestyle="--", alpha=0.4)
    ax_mid.legend(loc="upper right", fontsize=9)

    # Bottom: Absolute Disaggregation Error
    error_asym = np.abs(actual_sub - pred_asym_sub)
    error_sym = np.abs(actual_sub - pred_sym_sub)
    ax_bot.plot(t_sub, error_sym, color="#ff7f0e", linewidth=1.0, alpha=0.7, linestyle="--", label="Symmetric MSE Absolute Error")
    ax_bot.fill_between(t_sub, 0, error_asym, color="#d62728", alpha=0.35, label="Asymmetric Loss Absolute Error")
    ax_bot.set_ylabel("Absolute Error (W)", fontsize=10)
    ax_bot.set_xlabel("Time of Day", fontsize=11)
    ax_bot.set_title("Disaggregation Error Profile Comparison", fontsize=11, fontweight='semibold')
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    ax_bot.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p1 = PLOT_DIR / "nilm_actual_vs_predicted_test.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p1}", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 2: Feature Importance
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating Plot 2: Feature Importance ...", flush=True)
    importances = model.feature_importances_
    feat_names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1][:15]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feat_names[i] for i in sorted_idx[::-1]], fontsize=10)
    ax2.set_xlabel("Feature Importance (Gini Gain)", fontsize=11)
    ax2.set_title("Top 15 Most Informative Features for Washing Machine Disaggregation (Leak-Free)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

    plt.tight_layout()
    p2 = PLOT_DIR / "nilm_feature_importance.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2}", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 3: Failure Cases Analysis
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating Plot 3: Model Failure Modes Analysis ...", flush=True)
    fp_mask = (y_test < 5.0) & (y_pred_xgb_thresh > 200.0)
    fp_indices = np.where(fp_mask)[0]

    fn_mask = (y_test > 500.0) & (y_pred_xgb_thresh < 50.0)
    fn_indices = np.where(fn_mask)[0]

    fig3, (ax_fp, ax_fn) = plt.subplots(1, 2, figsize=(16, 5))
    fig3.suptitle("NILM Disaggregation Failure Mode Analysis (House 1 Test Data)", 
                  fontsize=13, fontweight='bold', y=1.03)

    if len(fp_indices) > 0:
        fp_center = test_index[fp_indices[len(fp_indices) // 2]]
        fp_start = fp_center - pd.Timedelta(hours=2)
        fp_end = fp_center + pd.Timedelta(hours=2)
        sub_fp = (test_index >= fp_start) & (test_index <= fp_end)

        ax_fp.plot(test_index[sub_fp], X_test.loc[sub_fp, "agg_power"], color="#1f77b4", linewidth=1.1, label="Aggregate")
        ax_fp.plot(test_index[sub_fp], y_test[sub_fp], color="#000000", linewidth=1.8, label="Actual WM")
        ax_fp.plot(test_index[sub_fp], y_pred_xgb_thresh[sub_fp], color="#d62728", linewidth=1.5, linestyle="--", label="Predicted WM")
        ax_fp.set_title(f"Failure Mode A: Resistive Cross-Appliance Confusion\n({fp_center.strftime('%Y-%m-%d %H:%M')})", 
                        fontsize=10, fontweight='bold')
        ax_fp.set_xlabel("Time", fontsize=10)
        ax_fp.set_ylabel("Power (W)", fontsize=10)
        ax_fp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_fp.grid(True, linestyle="--", alpha=0.4)
        ax_fp.legend(fontsize=9)
    else:
        ax_fp.text(0.5, 0.5, "No severe False Positives (>200W on 0W) found", ha="center")

    if len(fn_indices) > 0:
        fn_center = test_index[fn_indices[len(fn_indices) // 2]]
        fn_start = fn_center - pd.Timedelta(hours=2)
        fn_end = fn_center + pd.Timedelta(hours=2)
        sub_fn = (test_index >= fn_start) & (test_index <= fn_end)

        ax_fn.plot(test_index[sub_fn], X_test.loc[sub_fn, "agg_power"], color="#1f77b4", linewidth=1.1, label="Aggregate")
        ax_fn.plot(test_index[sub_fn], y_test[sub_fn], color="#000000", linewidth=1.8, label="Actual WM")
        ax_fn.plot(test_index[sub_fn], y_pred_xgb_thresh[sub_fn], color="#d62728", linewidth=1.5, linestyle="--", label="Predicted WM")
        ax_fn.set_title(f"Failure Mode B: Missed Extraction / Under-Prediction\n({fn_center.strftime('%Y-%m-%d %H:%M')})", 
                        fontsize=10, fontweight='bold')
        ax_fn.set_xlabel("Time", fontsize=10)
        ax_fn.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_fn.grid(True, linestyle="--", alpha=0.4)
        ax_fn.legend(fontsize=9)
    else:
        ax_fn.text(0.5, 0.5, "No severe False Negatives (>500W actual, <50W pred) found", ha="center")

    plt.tight_layout()
    p3 = PLOT_DIR / "nilm_failure_cases.png"
    fig3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {p3}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Disaggregation Modeling & Evaluation Complete!", flush=True)
    print("=" * 70, flush=True)


# ────────────────────────────────────────────────────────────────────────────
# 7. Leave-One-House-Out (LOHO) Cross-Validation Benchmark
# ────────────────────────────────────────────────────────────────────────────

def run_loho_experiment(
    train_houses: list[int] = [1, 2, 5, 7],
    test_house: int = 20,
    active_threshold_watts: float = ACTIVE_THRESHOLD_WATTS,
    max_pause_min: int = 5,
    min_duration_min: int = 15,
    min_peak_watts: float = 500.0,
    tolerance_min: int = 5,
    cycle_stitch_method: str = "dynamic_bridge",
    under_penalty: float = 2.0,
    feature_strategy: str = "rolling",
) -> pd.DataFrame:
    """Leave-One-House-Out (LOHO) Cross-Validation Benchmark.

    Rigorous Zero-Leakage Cross-Household Architecture:
    --------------------------------------------------
    1. Independent Per-Household Ingestion & Feature Engineering:
       Iterates through each training household in `train_houses`.
       Extracts temporal lags, rolling windows, and EWMA statistics strictly
       on that household's isolated 1-minute timeline.
       Guarantees ZERO cross-house boundary bleeding: House 1's trailing samples
       never bleed into House 2's starting lag features or moving averages.
    2. Isolated Internal Validation Split:
       Constructs validation partitions exclusively from the chronological holdout
       (final 15%) of each training house.
    3. Pure Out-of-Sample Evaluation:
       The target test household (`test_house`, e.g. House 20) remains 100% unseen
       during feature scaling, objective tuning, and tree growth.
    4. Dual-Tier Logging:
       Records MAE, RMSE, RETE, Idle Bleed, Peak Deficit, Sample F1, and Event F1,
       appending the results directly to `outputs/tables/nilm_model_metrics.csv`.

    Parameters
    ----------
    train_houses : list[int], default [1, 2, 5, 7]
        List of household IDs used for model training.
    test_house : int, default 20
        Target unseen household ID for zero-shot evaluation.
    active_threshold_watts : float, default 15.0
        Standby-to-active power threshold (W).
    max_pause_min : int, default 5
        Maximum bridgeable pause duration (minutes).
    min_duration_min : int, default 15
        Minimum valid cycle duration (minutes).
    min_peak_watts : float, default 500.0
        Minimum heating peak required for cycle validation (W).
    tolerance_min : int, default 5
        Event matching tolerance window (minutes).
    cycle_stitch_method : {"dynamic_bridge", "dbscan"}
        Cycle grouping algorithm.
    under_penalty : float, default 2.0
        Asymmetric loss multiplier for peak under-predictions.
    feature_strategy : {"rolling", "ewma", "hybrid"}
        Feature engineering method.

    Returns
    -------
    pd.DataFrame
        Table of evaluated LOHO model performance metrics.
    """
    print("=" * 80, flush=True)
    print(f"  LEAVE-ONE-HOUSE-OUT (LOHO) CROSS-HOUSEHOLD BENCHMARK", flush=True)
    print(f"  Training Houses: {train_houses} -> Target Unseen Test House: {test_house}", flush=True)
    print(f"  Stitching: {cycle_stitch_method} (pause={max_pause_min}m) | Feature Strategy: {feature_strategy}", flush=True)
    print("=" * 80, flush=True)

    # 1. Multi-House Ingestion & Isolated Feature Extraction
    print(f"\n[1] Ingesting {len(train_houses)} Training Households (Zero Cross-House Leakage Guarantee) ...", flush=True)
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for h in train_houses:
        df_h, wm_col = load_house_clean(h)
        n_h = len(df_h)
        # Chronological holdout strictly within this household for early stopping
        train_end = int(n_h * 0.85)
        df_h_tr = df_h.iloc[:train_end].copy()
        df_h_v = df_h.iloc[train_end:].copy()

        # Extract features independently per household to ensure sliding windows and lags NEVER bridge houses!
        X_h_tr = create_nilm_features(df_h_tr, "Aggregate", mode="batch", feature_strategy=feature_strategy)
        y_h_tr = df_h_tr[wm_col].values

        X_h_v = create_nilm_features(df_h_v, "Aggregate", mode="batch", feature_strategy=feature_strategy)
        y_h_v = df_h_v[wm_col].values

        X_train_list.append(X_h_tr)
        y_train_list.append(y_h_tr)
        X_val_list.append(X_h_v)
        y_val_list.append(y_h_v)
        print(f"  House {h}: {len(df_h_tr):,} train samples, {len(df_h_v):,} val samples ({df_h.index[0].date()} to {df_h.index[-1].date()})", flush=True)

    # Concatenate isolated matrices
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)
    X_val = pd.concat(X_val_list, ignore_index=True)
    y_val = np.concatenate(y_val_list)

    print(f"\n[2] Aggregated Multi-House Training Matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features", flush=True)
    print(f"    Multi-House Validation Matrix:      {X_val.shape[0]:,} rows x {X_val.shape[1]} features", flush=True)

    # 2. Ingest Unseen Target Household
    print(f"\n[3] Ingesting Target Unseen Household (House {test_house}) ...", flush=True)
    df_test, wm_test = load_house_clean(test_house)
    X_test = create_nilm_features(df_test, "Aggregate", mode="batch", feature_strategy=feature_strategy)
    y_test = df_test[wm_test].values
    test_index = df_test.index

    print(f"  Test House {test_house}: {len(df_test):,} minutes ({test_index.min().date()} to {test_index.max().date()})", flush=True)

    true_test_s = pd.Series(y_test, index=test_index)
    true_cycles = detect_washing_cycles(
        true_test_s,
        on_thresh=active_threshold_watts,
        min_duration=min_duration_min,
        max_pause=max_pause_min,
        min_peak=min_peak_watts,
        method=cycle_stitch_method,
    )
    print(f"  Ground-truth washing cycles in unseen House {test_house}: {len(true_cycles)} cycles", flush=True)

    eval_kwargs = dict(
        active_threshold_watts=active_threshold_watts,
        max_pause_min=max_pause_min,
        min_duration_min=min_duration_min,
        min_peak_watts=min_peak_watts,
        tolerance_min=tolerance_min,
        cycle_stitch_method=cycle_stitch_method,
    )

    metrics_list = []

    # Baseline 1: Always-Off (0W) on Test House
    print(f"\n[4] Evaluating Always-Off Baseline on House {test_house} ...", flush=True)
    y_pred_base1 = np.zeros_like(y_test)
    m1 = evaluate_predictions(y_test, y_pred_base1, test_index, f"LOHO Baseline: Always-Off (0W) [House {test_house}]", true_cycles, **eval_kwargs)
    metrics_list.append(m1)

    # Model 1: LOHO XGBoost (Symmetric MSE)
    print(f"\n[5] Training LOHO XGBoost Regressor (Symmetric MSE) ...", flush=True)
    model_sym = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model_sym.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    y_pred_sym_raw = np.clip(model_sym.predict(X_test), 0, None)
    y_pred_sym_thresh = y_pred_sym_raw.copy()
    y_pred_sym_thresh[y_pred_sym_thresh < 10.0] = 0.0

    m2 = evaluate_predictions(y_test, y_pred_sym_raw, test_index, f"LOHO XGBoost (Symmetric MSE - Raw) [Test: H{test_house}]", true_cycles, **eval_kwargs)
    m3 = evaluate_predictions(y_test, y_pred_sym_thresh, test_index, f"LOHO XGBoost (Symmetric MSE - Thresholded) [Test: H{test_house}]", true_cycles, **eval_kwargs)
    metrics_list.extend([m2, m3])

    # Model 2: LOHO XGBoost (Asymmetric Peak-Weighted Loss)
    print(f"\n[6] Training LOHO XGBoost Regressor (Asymmetric Peak-Weighted Loss) ...", flush=True)
    asym_obj = asymmetric_mse_objective(under_penalty=under_penalty, active_threshold=min_peak_watts)
    model_asym = xgb.XGBRegressor(
        objective=asym_obj,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )
    model_asym.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    y_pred_asym_raw = np.clip(model_asym.predict(X_test), 0, None)
    y_pred_asym_thresh = y_pred_asym_raw.copy()
    y_pred_asym_thresh[y_pred_asym_thresh < 10.0] = 0.0

    m4 = evaluate_predictions(y_test, y_pred_asym_raw, test_index, f"LOHO XGBoost (Asymmetric Loss - Raw) [Test: H{test_house}]", true_cycles, **eval_kwargs)
    m5 = evaluate_predictions(y_test, y_pred_asym_thresh, test_index, f"LOHO XGBoost (Asymmetric Loss - Thresholded) [Test: H{test_house}]", true_cycles, **eval_kwargs)
    metrics_list.extend([m4, m5])

    # Save to metrics table
    loho_df = pd.DataFrame(metrics_list)
    out_table = TABLE_DIR / "nilm_model_metrics.csv"
    if out_table.exists():
        existing_df = pd.read_csv(out_table)
        existing_df = existing_df[~existing_df["model"].isin(loho_df["model"])]
        combined_df = pd.concat([existing_df, loho_df], ignore_index=True)
    else:
        combined_df = loho_df
    combined_df.to_csv(out_table, index=False)
    print(f"\n[*] Updated metrics table saved to {out_table}", flush=True)

    print("\n" + "=" * 115, flush=True)
    print(f"  LEAVE-ONE-HOUSE-OUT (LOHO) CROSS-HOUSEHOLD BENCHMARK RESULTS (TEST: HOUSE {test_house})", flush=True)
    print("=" * 115, flush=True)
    cols_display = ["model", "mae_watts", "rmse_watts", "rete_energy_pct", "idle_bleed_kwh", "peak_deficit_kwh",
                    "sample_f1", "event_precision", "event_recall", "event_f1", "matched_tp_events"]
    print(loho_df[cols_display].to_string(index=False), flush=True)

    # Diagnostic Plot: 48-Hour Trace on Unseen House
    print(f"\n[Plots] Generating LOHO Actual vs Predicted Plot for House {test_house} ...", flush=True)
    test_wm_s = pd.Series(y_test, index=test_index)
    test_active_days = test_wm_s.resample("D").max()
    active_days = test_active_days[test_active_days > 1500].index

    if len(active_days) > 0:
        zoom_start = pd.Timestamp(active_days[len(active_days) // 2]).floor("D")
    else:
        zoom_start = test_index[len(test_index) // 2].floor("D")
    zoom_end = zoom_start + pd.Timedelta(hours=48)

    sub_mask = (test_index >= zoom_start) & (test_index <= zoom_end)
    t_sub = test_index[sub_mask]
    actual_sub = y_test[sub_mask]
    pred_asym_sub = y_pred_asym_thresh[sub_mask]
    pred_sym_sub = y_pred_sym_thresh[sub_mask]
    agg_sub = X_test.loc[sub_mask, "agg_power"].values

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle(f"REFIT Leave-One-House-Out (LOHO) Disaggregation on Unseen House {test_house}\n"
                 f"Trained on Houses {train_houses} | 48-Hour Trace ({zoom_start.strftime('%Y-%m-%d')} to {zoom_end.strftime('%Y-%m-%d')})",
                 fontsize=13, fontweight='bold', y=0.98)

    # Top: Whole-House Aggregate
    ax_top.plot(t_sub, agg_sub, color="#1f77b4", linewidth=1.1, alpha=0.85, label=f"Whole-House Aggregate (House {test_house})")
    ax_top.set_ylabel("Aggregate (W)", fontsize=10)
    ax_top.set_title(f"Input Signal: Unseen Household Aggregate Active Power (House {test_house})", fontsize=11, fontweight='semibold')
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="upper right", fontsize=9)

    # Mid: Actual vs Predicted
    ax_mid.plot(t_sub, actual_sub, color="#000000", linewidth=1.8, label="Actual Washing Machine (Ground Truth)")
    ax_mid.plot(t_sub, pred_asym_sub, color="#d62728", linewidth=1.6, alpha=0.95, label="LOHO XGBoost (Asymmetric Peak Loss - Thresholded)")
    ax_mid.plot(t_sub, pred_sym_sub, color="#ff7f0e", linewidth=1.3, alpha=0.85, linestyle="--", label="LOHO XGBoost (Symmetric MSE - Thresholded)")
    ax_mid.set_ylabel("Power (Watts)", fontsize=10)
    ax_mid.set_title("Zero-Shot Generalization: Ground Truth vs. Cross-Household Disaggregation", fontsize=11, fontweight='semibold')
    ax_mid.grid(True, linestyle="--", alpha=0.4)
    ax_mid.legend(loc="upper right", fontsize=9)

    # Bot: Error profile
    error_asym = np.abs(actual_sub - pred_asym_sub)
    error_sym = np.abs(actual_sub - pred_sym_sub)
    ax_bot.plot(t_sub, error_sym, color="#ff7f0e", linewidth=1.0, alpha=0.7, linestyle="--", label="LOHO Symmetric MSE Absolute Error")
    ax_bot.fill_between(t_sub, 0, error_asym, color="#d62728", alpha=0.35, label="LOHO Asymmetric Loss Absolute Error")
    ax_bot.set_ylabel("Absolute Error (W)", fontsize=10)
    ax_bot.set_xlabel("Time of Day", fontsize=11)
    ax_bot.set_title("Cross-Household Disaggregation Error Profile Comparison", fontsize=11, fontweight='semibold')
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    ax_bot.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p_out = PLOT_DIR / f"nilm_loho_house{test_house}_actual_vs_predicted.png"
    fig.savefig(p_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p_out}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  LOHO Cross-Validation Benchmark Complete!", flush=True)
    print("=" * 70, flush=True)
    return loho_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="REFIT NILM Disaggregation Model")
    parser.add_argument("--house-id", type=int, default=1, help="House ID")
    parser.add_argument("--loho", action="store_true", help="Run Leave-One-House-Out cross-validation")
    parser.add_argument("--train-houses", nargs="+", type=int, default=[1, 2, 5, 7], help="Training house IDs for LOHO")
    parser.add_argument("--test-house", type=int, default=20, help="Target unseen test house ID for LOHO")
    parser.add_argument("--active-thresh", type=float, default=15.0, help="Active power threshold (W)")
    parser.add_argument("--max-pause", type=int, default=5, help="Bridgeable pause duration in minutes (e.g. 5, 8, 10)")
    parser.add_argument("--min-duration", type=int, default=15, help="Minimum cycle duration in minutes")
    parser.add_argument("--tolerance-min", type=int, default=5, help="Event matching tolerance window (minutes)")
    parser.add_argument("--stitch-method", choices=["dynamic_bridge", "dbscan"], default="dynamic_bridge",
                        help="Cycle stitching algorithm: dynamic_bridge (fast 1D) or dbscan (1D density clustering)")
    parser.add_argument("--under-penalty", type=float, default=2.0, help="Asymmetric under-prediction penalty")
    parser.add_argument("--feature-strategy", choices=["rolling", "ewma", "hybrid"], default="rolling",
                        help="Feature engineering strategy: rolling (O(W) window buffer), ewma (O(1) memory state), or hybrid")
    args = parser.parse_args()

    if args.loho:
        run_loho_experiment(
            train_houses=args.train_houses,
            test_house=args.test_house,
            active_threshold_watts=args.active_thresh,
            max_pause_min=args.max_pause,
            min_duration_min=args.min_duration,
            min_peak_watts=500.0,
            tolerance_min=args.tolerance_min,
            cycle_stitch_method=args.stitch_method,
            under_penalty=args.under_penalty,
            feature_strategy=args.feature_strategy,
        )
    else:
        run_nilm_pipeline(
            house_id=args.house_id,
            active_threshold_watts=args.active_thresh,
            max_pause_min=args.max_pause,
            min_duration_min=args.min_duration,
            min_peak_watts=500.0,
            tolerance_min=args.tolerance_min,
            cycle_stitch_method=args.stitch_method,
            under_penalty=args.under_penalty,
            feature_strategy=args.feature_strategy,
        )
