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
os.environ["MPLCONFIGDIR"] = os.path.expanduser("~/.matplotlib_tmp")

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

from src.config import OUTPUT_DIR, PLOT_DIR, TABLE_DIR, CLEAN_DIR, WASHING_MACHINE_CHANNEL
from src.washing_machine_eda import detect_washing_cycles

logger = logging.getLogger(__name__)

ACTIVE_THRESHOLD_WATTS = 15.0


# ────────────────────────────────────────────────────────────────────────────
# 1. Leak-Free Feature Engineering (Strictly Within-Partition)
# ────────────────────────────────────────────────────────────────────────────

def create_nilm_features(
    df_partition: pd.DataFrame,
    agg_col: str = "Aggregate",
    mode: str = "batch",
) -> pd.DataFrame:
    """Engineer temporal context features strictly within an isolated partition.

    DATA LEAKAGE AUDIT GUARANTEE:
    -----------------------------
    To eliminate temporal boundary leakage, this function is executed STRICTLY
    AFTER the chronological train/val/test split. Because df_partition is an
    isolated slice, no future samples from validation or test partitions can
    inform the training features, and vice versa.

    Parameters
    ----------
    df_partition : pd.DataFrame
        1-minute resampled DataFrame for a single isolated partition (Train, Val, or Test).
    agg_col : str
        Column name of whole-home aggregate active power.
    mode : {"batch", "causal"}
        - "batch": Symmetrical rolling windows & centered statistics suitable for
          offline next-day utility disaggregation (Seq2Point paradigm).
        - "causal": Strictly trailing historical statistics suitable for real-time streaming.
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

    # 2. Multi-scale rolling statistics
    is_center = (mode == "batch")
    for w in [5, 15, 30, 60]:
        feat[f"agg_roll_mean_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).mean().bfill().ffill()
        feat[f"agg_roll_std_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).std().fillna(0)
        feat[f"agg_roll_max_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).max().bfill().ffill()
        feat[f"agg_roll_min_{w}"] = s.rolling(window=w, min_periods=1, center=is_center).min().bfill().ffill()
        feat[f"agg_roll_range_{w}"] = feat[f"agg_roll_max_{w}"] - feat[f"agg_roll_min_{w}"]

    # 3. Dynamic differences / gradients
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
# 2. Event-Based Macro Cycle Evaluation (Matching with Tolerance)
# ────────────────────────────────────────────────────────────────────────────

def match_and_evaluate_events(
    true_cycles: list[dict],
    pred_cycles: list[dict],
    tolerance_min: int = 5,
) -> dict:
    """Evaluate event detection by matching predicted cycles to ground-truth cycles.

    Matching Criteria:
    ------------------
    A predicted cycle matches a ground-truth cycle if:
      1. Their operating intervals overlap in time: max(p_start, t_start) <= min(p_end, t_end)
         OR
      2. Their cycle start times agree within `tolerance_min` (e.g. +/- 5 minutes).

    Each predicted cycle matches at most one ground-truth cycle (1-to-1 greedy matching).

    Returns
    -------
    dict: true_events, pred_events, tp_events, fp_events, fn_events,
          event_precision, event_recall, event_f1
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
# 3. Dual-Tier Evaluation Function
# ────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_index: pd.DatetimeIndex,
    model_name: str,
    true_cycles: list[dict],
) -> dict:
    """Compute comprehensive sample-level and event-level evaluation metrics."""
    y_pred_clipped = np.clip(y_pred, 0, None)

    # 1. Continuous Regression Metrics
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = root_mean_squared_error(y_true, y_pred_clipped)

    total_true_energy = np.sum(y_true)
    total_pred_energy = np.sum(y_pred_clipped)
    # Net Energy Error (Relative Error in Total Energy / RETE)
    rete = abs(total_pred_energy - total_true_energy) / (total_true_energy + 1e-6)

    # 2. Sample-Level Classification Metrics (P >= 15W)
    y_true_bin = (y_true >= ACTIVE_THRESHOLD_WATTS).astype(int)
    y_pred_bin = (y_pred_clipped >= ACTIVE_THRESHOLD_WATTS).astype(int)

    sample_precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    sample_recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    sample_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    # 3. Event-Level Macro Cycle Metrics
    pred_s = pd.Series(y_pred_clipped, index=test_index)
    pred_cycles = detect_washing_cycles(pred_s)
    event_metrics = match_and_evaluate_events(true_cycles, pred_cycles, tolerance_min=5)

    return {
        "model": model_name,
        "mae_watts": round(float(mae), 2),
        "rmse_watts": round(float(rmse), 2),
        "rete_energy_pct": round(float(rete * 100), 2),
        "total_true_kwh": round(float(total_true_energy / 60.0 / 1000.0), 2),
        "total_pred_kwh": round(float(total_pred_energy / 60.0 / 1000.0), 2),
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
# 4. Main Model Pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_nilm_pipeline(house_id: int = 1):
    print("=" * 70, flush=True)
    print(f"  Appliance Disaggregation (NILM) Pipeline — House {house_id}", flush=True)
    print("=" * 70, flush=True)

    # Load cleaned 1-min data
    print("[1] Loading 1-minute cleaned data ...", flush=True)
    fpath = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    wm_col = WASHING_MACHINE_CHANNEL[house_id]
    df_raw = pd.read_csv(fpath, usecols=["Time", "Aggregate", wm_col])
    df_raw["Time"] = pd.to_datetime(df_raw["Time"], format="%Y-%m-%d %H:%M:%S")
    df_raw = df_raw.set_index("Time").sort_index()

    # Resample to 1-minute
    df_1min = df_raw.resample("1min").mean()

    # Filter out outage periods (where data was missing/unmonitored)
    valid_mask = df_1min["Aggregate"].notna() & df_1min[wm_col].notna()
    df_clean = df_1min[valid_mask].copy()
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
    print("[3] Engineering within-partition context features ...", flush=True)
    X_train = create_nilm_features(df_train, "Aggregate", mode="batch")
    y_train = df_train[wm_col].values

    X_val = create_nilm_features(df_val, "Aggregate", mode="batch")
    y_val = df_val[wm_col].values

    X_test = create_nilm_features(df_test, "Aggregate", mode="batch")
    y_test = df_test[wm_col].values
    test_index = df_test.index

    # Pre-extract ground truth test cycles for macro event evaluation
    true_test_s = pd.Series(y_test, index=test_index)
    true_cycles = detect_washing_cycles(true_test_s)
    print(f"  Ground-truth washing cycles in test set: {len(true_cycles)} cycles", flush=True)
    print(f"  Active test minutes (>=15W): {(y_test >= ACTIVE_THRESHOLD_WATTS).sum():,} "
          f"({(y_test >= ACTIVE_THRESHOLD_WATTS).mean()*100:.2f}%)", flush=True)

    metrics_list = []

    # ── Baseline 1: Always-Off (0 W) ────────────────────────────────────
    print("\n[4] Evaluating Baseline 1: Always-Off (0 W) ...", flush=True)
    y_pred_base1 = np.zeros_like(y_test)
    m1 = evaluate_predictions(y_test, y_pred_base1, test_index, "Baseline 1: Always-Off (0W)", true_cycles)
    metrics_list.append(m1)

    # ── Baseline 2: Diurnal Usage Profile ───────────────────────────────
    print("[5] Evaluating Baseline 2: Diurnal Usage Profile ...", flush=True)
    train_df = pd.DataFrame({"wm": y_train, "hour": df_train.index.hour, 
                             "is_weekend": (df_train.index.dayofweek >= 5).astype(int)})
    profile = train_df.groupby(["hour", "is_weekend"])["wm"].mean()

    test_keys = list(zip(test_index.hour, (test_index.dayofweek >= 5).astype(int)))
    y_pred_base2 = np.array([profile.get(k, 0.0) for k in test_keys])
    m2 = evaluate_predictions(y_test, y_pred_base2, test_index, "Baseline 2: Diurnal Profile", true_cycles)
    metrics_list.append(m2)

    # ── Model 3: XGBoost Regressor ──────────────────────────────────────
    print("\n[6] Training Leak-Free XGBoost Regressor ...", flush=True)
    model = xgb.XGBRegressor(
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    y_pred_xgb_raw = model.predict(X_test)
    y_pred_xgb = np.clip(y_pred_xgb_raw, 0, None)

    # Threshold filtering (power < 10 W is zeroed out to eliminate standby bleed)
    y_pred_xgb_thresh = y_pred_xgb.copy()
    y_pred_xgb_thresh[y_pred_xgb_thresh < 10.0] = 0.0

    m3 = evaluate_predictions(y_test, y_pred_xgb, test_index, "XGBoost Regressor (Raw)", true_cycles)
    m4 = evaluate_predictions(y_test, y_pred_xgb_thresh, test_index, "XGBoost Regressor (Thresholded)", true_cycles)
    metrics_list.extend([m3, m4])

    # Save comprehensive metrics table
    metrics_df = pd.DataFrame(metrics_list)
    out_table = TABLE_DIR / "nilm_model_metrics.csv"
    metrics_df.to_csv(out_table, index=False)
    print(f"\n[*] Metrics table saved to {out_table}", flush=True)
    print("\n" + "=" * 95, flush=True)
    print("  COMPREHENSIVE MODEL EVALUATION BENCHMARK (DUAL-TIER METRICS)", flush=True)
    print("=" * 95, flush=True)
    cols_display = ["model", "mae_watts", "rmse_watts", "rete_energy_pct", 
                    "sample_f1", "event_precision", "event_recall", "event_f1", "matched_tp_events"]
    print(metrics_df[cols_display].to_string(index=False), flush=True)

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
    pred_xgb_sub = y_pred_xgb_thresh[sub_mask]
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
    ax_mid.plot(t_sub, pred_xgb_sub, color="#d62728", linewidth=1.5, alpha=0.9, linestyle="--", 
                label="XGBoost Disaggregation (Predicted)")
    ax_mid.plot(t_sub, pred_base2_sub, color="#ff7f0e", linewidth=1.0, alpha=0.6, linestyle=":", 
                label="Baseline 2: Diurnal Profile")
    ax_mid.set_ylabel("Power (Watts)", fontsize=10)
    ax_mid.set_title("Target Appliance Disaggregation: Actual vs. Predicted Power", fontsize=11, fontweight='semibold')
    ax_mid.grid(True, linestyle="--", alpha=0.4)
    ax_mid.legend(loc="upper right", fontsize=9)

    # Bottom: Absolute Disaggregation Error
    error = np.abs(actual_sub - pred_xgb_sub)
    ax_bot.fill_between(t_sub, 0, error, color="#d62728", alpha=0.35, label="Absolute Error |Actual - Predicted|")
    ax_bot.set_ylabel("Absolute Error (W)", fontsize=10)
    ax_bot.set_xlabel("Time of Day", fontsize=11)
    ax_bot.set_title("Disaggregation Error Profile", fontsize=11, fontweight='semibold')
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


if __name__ == "__main__":
    run_nilm_pipeline(house_id=1)
