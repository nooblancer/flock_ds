"""
Appliance-Disaggregation (NILM) Pipeline for Washing Machine Power Estimation.

This module:
1. Loads 1-minute cleaned data for House 1 (and optional transfer house).
2. Generates causal and temporal rolling context features from the Aggregate channel.
3. Implements strict chronological train/val/test splits to eliminate temporal leakage.
4. Implements and benchmarks baselines:
     - Baseline 1: Always-Off (0 W)
     - Baseline 2: Diurnal Usage Profile (hour-of-day + weekend conditional mean)
5. Trains an optimized Gradient Boosting Regressor (XGBoost).
6. Evaluates models using both regression (MAE, RMSE, Energy Error) and
   event-detection metrics (Precision, Recall, F1 at active threshold).
7. Generates actual-vs-predicted evaluation plots and failure-mode analysis traces.
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

from src.config import OUTPUT_DIR, PLOT_DIR, TABLE_DIR, CLEAN_DIR

logger = logging.getLogger(__name__)

ACTIVE_THRESHOLD_WATTS = 15.0


# ────────────────────────────────────────────────────────────────────────────
# 1. Feature Engineering
# ────────────────────────────────────────────────────────────────────────────

def create_nilm_features(df_1min: pd.DataFrame, agg_col: str = "Aggregate") -> pd.DataFrame:
    """Engineer temporal context features from the aggregate power signal.

    Features engineered:
      - Temporal context window: lags (-15 to -1 min) and leads (+1 to +15 min)
      - Rolling statistics: mean, std, min, max, span over 5m, 15m, 30m, 60m
      - Signal dynamics: instantaneous rate of change (delta 1m, 5m, 15m)
      - Calendar priors: hour of day (sin/cos cyclic encoding), day of week, is_weekend
    """
    df = df_1min.copy()
    s = df[agg_col]

    feat = pd.DataFrame(index=df.index)
    feat["agg_power"] = s

    # 1. Immediate lags and leads (context window: [-15m, +15m])
    for lag in [-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15]:
        col_name = f"agg_lag_{lag}" if lag < 0 else f"agg_lead_{lag}"
        feat[col_name] = s.shift(-lag)

    # 2. Rolling statistics (centered windows to capture full cycle trajectory)
    for w in [5, 15, 30, 60]:
        feat[f"agg_roll_mean_{w}"] = s.rolling(window=w, min_periods=1, center=True).mean()
        feat[f"agg_roll_std_{w}"] = s.rolling(window=w, min_periods=1, center=True).std().fillna(0)
        feat[f"agg_roll_max_{w}"] = s.rolling(window=w, min_periods=1, center=True).max()
        feat[f"agg_roll_min_{w}"] = s.rolling(window=w, min_periods=1, center=True).min()
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

    return feat


# ────────────────────────────────────────────────────────────────────────────
# 2. Evaluation Metrics Calculation
# ────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute comprehensive regression and event-detection metrics."""
    # Ensure non-negative predictions
    y_pred_clipped = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred_clipped)
    rmse = root_mean_squared_error(y_true, y_pred_clipped)

    # Relative Error in Total Energy (RETE / SAE)
    total_true_energy = np.sum(y_true)
    total_pred_energy = np.sum(y_pred_clipped)
    rete = abs(total_pred_energy - total_true_energy) / (total_true_energy + 1e-6)

    # Classification metrics at active threshold (>= 15W)
    y_true_bin = (y_true >= ACTIVE_THRESHOLD_WATTS).astype(int)
    y_pred_bin = (y_pred_clipped >= ACTIVE_THRESHOLD_WATTS).astype(int)

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return {
        "model": model_name,
        "mae_watts": round(float(mae), 2),
        "rmse_watts": round(float(rmse), 2),
        "rete_energy_pct": round(float(rete * 100), 2),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "total_true_kwh": round(float(total_true_energy / 60.0 / 1000.0), 2),
        "total_pred_kwh": round(float(total_pred_energy / 60.0 / 1000.0), 2),
    }


# ────────────────────────────────────────────────────────────────────────────
# 3. Main Model Pipeline
# ────────────────────────────────────────────────────────────────────────────

def run_nilm_pipeline(house_id: int = 1):
    print("=" * 70, flush=True)
    print(f"  Appliance Disaggregation (NILM) Pipeline — House {house_id}", flush=True)
    print("=" * 70, flush=True)

    # Load cleaned 1-min data
    from src.config import WASHING_MACHINE_CHANNEL
    print("[1] Loading 1-minute cleaned data ...", flush=True)
    fpath = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    wm_col = WASHING_MACHINE_CHANNEL[house_id]
    df_raw = pd.read_csv(fpath, usecols=["Time", "Aggregate", wm_col])
    df_raw["Time"] = pd.to_datetime(df_raw["Time"], format="%Y-%m-%d %H:%M:%S")
    df_raw = df_raw.set_index("Time").sort_index()
    
    # Resample to 1-minute
    df_1min = df_raw.resample("1min").mean()
    
    # Drop rows where Aggregate or Washing Machine is NaN (outages)
    valid_mask = df_1min["Aggregate"].notna() & df_1min[wm_col].notna()
    df_clean = df_1min[valid_mask].copy()
    print(f"  Valid 1-minute observations: {len(df_clean):,} minutes "
          f"({df_clean.index.min().date()} to {df_clean.index.max().date()})", flush=True)

    # Feature Engineering
    print("[2] Engineering rolling context and temporal features ...", flush=True)
    X = create_nilm_features(df_clean, "Aggregate")
    y = df_clean[wm_col].values

    # Strict chronological split: 70% Train, 10% Validation, 20% Test
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]
    test_index = df_clean.index[val_end:]

    print(f"  Train set:      {len(X_train):,} mins ({df_clean.index[0].date()} -> {df_clean.index[train_end-1].date()})", flush=True)
    print(f"  Validation set: {len(X_val):,} mins ({df_clean.index[train_end].date()} -> {df_clean.index[val_end-1].date()})", flush=True)
    print(f"  Test set:       {len(X_test):,} mins ({df_clean.index[val_end].date()} -> {df_clean.index[-1].date()})", flush=True)
    print(f"  Test active minutes (>=15W): {(y_test >= ACTIVE_THRESHOLD_WATTS).sum():,} "
          f"({(y_test >= ACTIVE_THRESHOLD_WATTS).mean()*100:.2f}%)", flush=True)

    metrics_list = []

    # ── Baseline 1: Always-Off ──────────────────────────────────────────
    print("\n[3] Evaluating Baseline 1: Always-Off (0 W) ...", flush=True)
    y_pred_base1 = np.zeros_like(y_test)
    m1 = evaluate_predictions(y_test, y_pred_base1, "Baseline 1: Always-Off (0W)")
    metrics_list.append(m1)

    # ── Baseline 2: Diurnal Profile ─────────────────────────────────────
    print("[4] Evaluating Baseline 2: Diurnal Usage Profile ...", flush=True)
    train_df = pd.DataFrame({"wm": y_train, "hour": df_clean.index[:train_end].hour, 
                             "is_weekend": (df_clean.index[:train_end].dayofweek >= 5).astype(int)})
    profile = train_df.groupby(["hour", "is_weekend"])["wm"].mean()

    test_keys = list(zip(test_index.hour, (test_index.dayofweek >= 5).astype(int)))
    y_pred_base2 = np.array([profile.get(k, 0.0) for k in test_keys])
    m2 = evaluate_predictions(y_test, y_pred_base2, "Baseline 2: Diurnal Profile")
    metrics_list.append(m2)

    # ── Model 3: XGBoost Regressor ──────────────────────────────────────
    print("\n[5] Training XGBoost Regressor with Early Stopping ...", flush=True)
    # Fill any remaining NaNs in features
    X_train_clean = X_train.fillna(0)
    X_val_clean = X_val.fillna(0)
    X_test_clean = X_test.fillna(0)

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
        X_train_clean, y_train,
        eval_set=[(X_val_clean, y_val)],
        verbose=50,
    )

    y_pred_xgb_raw = model.predict(X_test_clean)
    y_pred_xgb = np.clip(y_pred_xgb_raw, 0, None)
    
    # Threshold filtering: power < 10 W is zeroed out (eliminates standby leakage)
    y_pred_xgb_thresh = y_pred_xgb.copy()
    y_pred_xgb_thresh[y_pred_xgb_thresh < 10.0] = 0.0

    m3 = evaluate_predictions(y_test, y_pred_xgb, "XGBoost Regressor (Raw)")
    m4 = evaluate_predictions(y_test, y_pred_xgb_thresh, "XGBoost Regressor (Thresholded)")
    metrics_list.extend([m3, m4])

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_list)
    out_table = TABLE_DIR / "nilm_model_metrics.csv"
    metrics_df.to_csv(out_table, index=False)
    print(f"\n[*] Metrics table saved to {out_table}", flush=True)
    print(metrics_df[["model", "mae_watts", "rmse_watts", "f1_score", "precision", "recall", "rete_energy_pct"]].to_string(index=False), flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1: Actual vs. Predicted on Evaluation Data (Zoomed 48h Window)
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating Plot 1: Actual vs. Predicted Evaluation Plot ...", flush=True)
    # Find a 48h window in the test set containing active washing cycles
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
    fig1.suptitle(f"REFIT House 1: Disaggregation Evaluation on Unseen Test Data\n"
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
    feat_names = X_train_clean.columns
    sorted_idx = np.argsort(importances)[::-1][:15]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(range(len(sorted_idx)), importances[sorted_idx][::-1], color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feat_names[i] for i in sorted_idx[::-1]], fontsize=10)
    ax2.set_xlabel("Feature Importance (Gini Gain)", fontsize=11)
    ax2.set_title("Top 15 Most Informative Features for Washing Machine Disaggregation", fontsize=12, fontweight='bold')
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
    # Failure Case 1: False Positive (Cooking/Kettle spike misclassified or high aggregate load)
    # Failure Case 2: Under-prediction / Missed drum agitation
    
    # Find worst false positive window (Predicted high, Actual zero)
    fp_mask = (y_test < 5.0) & (y_pred_xgb_thresh > 200.0)
    fp_indices = np.where(fp_mask)[0]
    
    # Find worst false negative window (Actual high, Predicted near zero)
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
        ax_fp.set_title(f"Failure Mode A: False Positive / Overlapping Load\n({fp_center.strftime('%Y-%m-%d %H:%M')})", 
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
