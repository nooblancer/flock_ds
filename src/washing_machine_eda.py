"""
Washing Machine Exploratory Data Analysis & Cycle Detection across multiple REFIT households.

This module:
1. Loads cleaned REFIT data for selected households (at 1-minute resolution).
2. Detects discrete washing-machine operating cycles using physical thresholding.
3. Extracts cycle metrics (duration, energy, peak power).
4. Analyzes diurnal and day-of-week usage distributions.
5. Produces required visual figures and summary tables.
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

from src.config import (
    CLEAN_DIR, OUTPUT_DIR, PLOT_DIR, TABLE_DIR,
    WASHING_MACHINE_CHANNEL, APPLIANCE_MAP
)

cache_dir = OUTPUT_DIR / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Cycle Detection Parameters (Configurable Defaults)
# ────────────────────────────────────────────────────────────────────────────
ON_POWER_THRESHOLD_WATTS = 15.0       # Above standby power (~1-5W)
MIN_CYCLE_DURATION_MINUTES = 15       # Minimum duration for valid wash cycle
MAX_PAUSE_MINUTES = 5                 # Bridgeable pause between cycle phases (soaking/rest)
MIN_PEAK_POWER_WATTS = 500.0          # Must engage heating element or high-torque motor


def detect_washing_cycles(
    series_1min: pd.Series,
    on_thresh: float = ON_POWER_THRESHOLD_WATTS,
    min_duration: int = MIN_CYCLE_DURATION_MINUTES,
    max_pause: int = MAX_PAUSE_MINUTES,
    min_peak: float = MIN_PEAK_POWER_WATTS,
    method: str = "dynamic_bridge",
) -> list[dict]:
    """Detect discrete operating cycles from a 1-minute washing machine power series.

    A washing cycle is defined as an active window where power exceeds `on_thresh`,
    intermittent pauses up to `max_pause` minutes are bridged, total duration is
    at least `min_duration`, and peak power exceeds `min_peak`.

    Parameters
    ----------
    series_1min : pd.Series
        1-minute power readings (Watts) with DatetimeIndex.
    on_thresh : float, default 15.0
        Active power threshold in Watts.
    min_duration : int, default 15
        Minimum valid cycle duration in minutes.
    max_pause : int, default 5
        Maximum inactive pause in minutes bridged within an ongoing cycle.
    min_peak : float, default 500.0
        Minimum peak power required to confirm thermal or motor engagement.
    method : {"dynamic_bridge", "dbscan"}, default "dynamic_bridge"
        Temporal grouping method: vectorized index bridging or 1D DBSCAN.

    Returns
    -------
    list of dict
        Cycle records with start, end, duration_min, mean_power_w, peak_power_w, energy_kwh.
    """
    clean_s = series_1min.dropna()
    if clean_s.empty:
        return []

    values = clean_s.values
    index = clean_s.index
    is_on = (values >= on_thresh).astype(np.int32)
    on_indices = np.where(is_on == 1)[0]
    if len(on_indices) == 0:
        return []

    if method == "dbscan":
        from sklearn.cluster import DBSCAN
        # 1D DBSCAN on discrete temporal minute indices
        # eps is the maximum distance between points in the same cluster: (max_pause + 1)
        clustering = DBSCAN(eps=float(max_pause + 1), min_samples=1).fit(on_indices.reshape(-1, 1))
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        starts = []
        ends = []
        for lab in unique_labels:
            if lab == -1:
                continue
            c_indices = on_indices[labels == lab]
            starts.append(c_indices[0])
            ends.append(c_indices[-1] + 1)
        starts = np.array(starts, dtype=int)
        ends = np.array(ends, dtype=int)
    else:
        # Vectorized numpy gap-bridging for pauses <= max_pause
        bridged = is_on.copy()
        gaps = np.diff(on_indices) - 1
        for i, g in enumerate(gaps):
            if 0 < g <= max_pause:
                bridged[on_indices[i] + 1 : on_indices[i + 1]] = 1

        # Extract continuous segments via difference on padded array
        diff = np.diff(np.pad(bridged, (1, 1), 'constant'))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

    cycles = []
    for s_idx, e_idx in zip(starts, ends):
        dur_min = int(e_idx - s_idx)
        if dur_min >= min_duration:
            cycle_power = values[s_idx:e_idx]
            peak_p = float(np.max(cycle_power))
            if peak_p >= min_peak:
                mean_p = float(np.mean(cycle_power))
                energy_kwh = float(np.sum(cycle_power) / 60.0 / 1000.0)
                start_dt = index[s_idx]
                end_dt = index[e_idx - 1]

                cycles.append({
                    "start": str(start_dt),
                    "end": str(end_dt),
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "duration_min": dur_min,
                    "mean_power_w": round(mean_p, 1),
                    "peak_power_w": round(peak_p, 1),
                    "energy_kwh": round(energy_kwh, 3),
                    "hour": start_dt.hour,
                    "dayofweek": start_dt.day_name(),
                    "dayofweek_idx": start_dt.dayofweek,
                })

    return cycles


def load_and_resample_household(house_id: int) -> tuple[pd.DataFrame, str]:
    """Load official cleaned data for a house and resample to 1-minute."""
    fpath = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")

    wm_col = WASHING_MACHINE_CHANNEL[house_id]
    cols_to_load = ["Time", "Aggregate", wm_col]
    
    # Check if Issues exists
    sample_df = pd.read_csv(fpath, nrows=1)
    if "Issues" in sample_df.columns:
        cols_to_load.append("Issues")

    print(f"  Loading House {house_id} ({wm_col}) ...", flush=True)
    df = pd.read_csv(fpath, usecols=cols_to_load)
    
    # Fast format parsing
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("Time").sort_index()
    
    if "Issues" in df.columns:
        df.loc[df["Issues"] == 1, ["Aggregate", wm_col]] = np.nan
        df = df.drop(columns=["Issues"])

    # 1-minute mean resampling
    df_1min = df.resample("1min").mean()
    return df_1min, wm_col


def run_eda(target_houses: list[int] = [1, 2, 5, 7, 20]):
    """Execute complete cross-house EDA on washing machines."""
    print("=" * 70, flush=True)
    print("  REFIT Washing-Machine Exploratory Data Analysis (>=4 Households)", flush=True)
    print("=" * 70, flush=True)

    house_dfs: dict[int, pd.DataFrame] = {}
    house_wm_cols: dict[int, str] = {}
    house_cycles: dict[int, list[dict]] = {}
    house_summaries: list[dict] = []

    for hid in target_houses:
        df_1min, wm_col = load_and_resample_household(hid)
        house_dfs[hid] = df_1min
        house_wm_cols[hid] = wm_col

        wm_series = df_1min[wm_col]
        agg_series = df_1min["Aggregate"]
        
        # Summary statistics
        total_minutes = len(df_1min)
        valid_minutes = int(wm_series.notna().sum())
        missing_pct = round((total_minutes - valid_minutes) / total_minutes * 100, 2)
        zero_minutes = int((wm_series < 1.0).sum())
        zero_pct = round(zero_minutes / valid_minutes * 100, 2) if valid_minutes > 0 else 0
        standby_minutes = int(((wm_series >= 1.0) & (wm_series < ON_POWER_THRESHOLD_WATTS)).sum())
        standby_pct = round(standby_minutes / valid_minutes * 100, 2) if valid_minutes > 0 else 0
        active_minutes = int((wm_series >= ON_POWER_THRESHOLD_WATTS).sum())
        active_pct = round(active_minutes / valid_minutes * 100, 2) if valid_minutes > 0 else 0

        # Cycle detection
        print(f"  Detecting cycles for House {hid} ...", flush=True)
        cycles = detect_washing_cycles(wm_series)
        house_cycles[hid] = cycles
        print(f"    House {hid}: Detected {len(cycles)} active operating cycles.", flush=True)

        # Cycle stats
        durations = [c["duration_min"] for c in cycles]
        energies = [c["energy_kwh"] for c in cycles]
        peaks = [c["peak_power_w"] for c in cycles]

        # Days of coverage
        n_days = (df_1min.index.max() - df_1min.index.min()).days
        cycles_per_week = round(len(cycles) / (n_days / 7.0), 2) if n_days > 0 else 0

        summary = {
            "house_id": hid,
            "channel": wm_col,
            "appliance_meta": APPLIANCE_MAP[hid].get(wm_col, "Washing_Machine"),
            "start_date": str(df_1min.index.min().date()),
            "end_date": str(df_1min.index.max().date()),
            "duration_days": n_days,
            "total_minutes": total_minutes,
            "missing_pct": missing_pct,
            "zero_off_pct": zero_pct,
            "standby_pct": standby_pct,
            "active_pct": active_pct,
            "total_cycles_detected": len(cycles),
            "cycles_per_week": cycles_per_week,
            "median_cycle_duration_min": round(float(np.median(durations)), 1) if durations else 0,
            "mean_cycle_duration_min": round(float(np.mean(durations)), 1) if durations else 0,
            "median_cycle_energy_kwh": round(float(np.median(energies)), 3) if energies else 0,
            "mean_cycle_energy_kwh": round(float(np.mean(energies)), 3) if energies else 0,
            "mean_peak_power_w": round(float(np.mean(peaks)), 1) if peaks else 0,
        }
        house_summaries.append(summary)

    # Save summary table
    summary_df = pd.DataFrame(house_summaries)
    summary_csv = TABLE_DIR / "wm_cross_house_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[*] Summary table saved to {summary_csv}", flush=True)
    print(summary_df[["house_id", "appliance_meta", "total_cycles_detected", "cycles_per_week", 
                      "median_cycle_duration_min", "mean_cycle_energy_kwh"]].to_string(index=False), flush=True)

    # Save cycle JSON details
    cycles_json_path = TABLE_DIR / "wm_detected_cycles_detail.json"
    serializable_cycles = {
        h: [{k: v for k, v in c.items() if not k.endswith("_dt")} for c in clist]
        for h, clist in house_cycles.items()
    }
    with open(cycles_json_path, "w") as f:
        json.dump(serializable_cycles, f, indent=2)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1: Representative 24-Hour Aggregate + Washing Machine Trace
    # ────────────────────────────────────────────────────────────────────────
    print("\n[Plots] Generating Plot 1: Representative 24-hour trace ...", flush=True)
    h1_df = house_dfs[1]
    h1_wm = house_wm_cols[1]
    h1_cycles = house_cycles[1]
    
    # Pick a representative cycle in House 1
    chosen_cycle = h1_cycles[len(h1_cycles) // 3]
    cycle_dt = pd.Timestamp(chosen_cycle["start_dt"])
    day_start = cycle_dt.floor("D")
    day_end = day_start + pd.Timedelta(hours=24)

    day_slice = h1_df.loc[day_start:day_end]

    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(day_slice.index, day_slice["Aggregate"], label="Whole-House Aggregate (W)", 
             color="#1f77b4", linewidth=1.2, alpha=0.85)
    ax1.plot(day_slice.index, day_slice[h1_wm], label="Washing Machine (Appliance 5 - Beko)", 
             color="#d62728", linewidth=1.8, alpha=0.95)
    
    # Highlight detected cycle window
    ax1.axvspan(chosen_cycle["start_dt"], chosen_cycle["end_dt"], color="#ff7f0e", alpha=0.25,
                label=f"Detected Cycle: {chosen_cycle['duration_min']} min, {chosen_cycle['energy_kwh']} kWh (Peak {chosen_cycle['peak_power_w']} W)")

    ax1.set_title(f"REFIT House 1: Representative 24-Hour Trace ({day_start.strftime('%A, %d %B %Y')})\n"
                  f"Aggregate vs. Washing Machine Sub-metered Load with Cycle Detection", 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel("Time of Day", fontsize=11)
    ax1.set_ylabel("Active Power (Watts)", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax1.set_ylim(0, max(day_slice["Aggregate"].max() * 1.05, 3000))

    p1 = PLOT_DIR / "wm_representative_24h_trace.png"
    fig1.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {p1}", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 2: Detailed Signature of Detected Washing Machine Cycles
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating Plot 2: Detected cycle signatures zoom ...", flush=True)
    sample_cycles = [h1_cycles[10], h1_cycles[25], h1_cycles[40]]
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig2.suptitle("REFIT House 1: High-Resolution Profile of Detected Washing-Machine Cycles\n"
                  "(Illustrating Heating Element Spike, Intermittent Motor Agitation, and Final Spin Extraction)",
                  fontsize=13, fontweight='bold', y=1.03)

    for i, c in enumerate(sample_cycles):
        ax = axes2[i]
        c_start = pd.Timestamp(c["start_dt"]) - pd.Timedelta(minutes=10)
        c_end = pd.Timestamp(c["end_dt"]) + pd.Timedelta(minutes=10)
        c_sub = h1_df.loc[c_start:c_end]

        ax.plot(c_sub.index, c_sub["Aggregate"], color="#7f7f7f", linestyle=":", linewidth=1.0, alpha=0.7, label="Aggregate")
        ax.plot(c_sub.index, c_sub[h1_wm], color="#d62728", linewidth=1.8, label="Washing Machine")
        ax.axvspan(c["start_dt"], c["end_dt"], color="#2ca02c", alpha=0.15, label="Detected Event")

        ax.set_title(f"Cycle #{i+1}: {c['start_dt'].strftime('%Y-%m-%d %H:%M')}\n"
                     f"Duration: {c['duration_min']}m | Energy: {c['energy_kwh']} kWh | Peak: {c['peak_power_w']:.0f}W",
                     fontsize=10, fontweight='semibold')
        ax.set_xlabel("Time", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.set_ylabel("Power (Watts)", fontsize=11)
            ax.legend(loc="upper left", fontsize=9)

    p2 = PLOT_DIR / "wm_detected_cycles_detail.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2}", flush=True)

    # ────────────────────────────────────────────────────────────────────────
    # Plot 3: Cross-House Comparison (Diurnal, Day-of-Week, Cycle Metrics)
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating Plot 3: Cross-house behavioral comparisons ...", flush=True)
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle("Cross-Household Washing Machine Behavioral Patterns & Electrical Characteristics\n"
                  f"(Analysed across Houses {target_houses} at 1-minute resolution)",
                  fontsize=14, fontweight='bold', y=0.99)

    # 3A: Diurnal Usage Profile (Hour of Day)
    ax_hour = axes3[0, 0]
    hours = np.arange(24)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, hid in enumerate(target_houses):
        c_hours = [c["hour"] for c in house_cycles[hid]]
        counts, _ = np.histogram(c_hours, bins=np.arange(25))
        prob = counts / len(c_hours) * 100 if len(c_hours) > 0 else np.zeros(24)
        ax_hour.plot(hours, prob, marker='o', markersize=4, label=f"House {hid} ({len(c_hours)} cycles)",
                     color=colors[idx % len(colors)], linewidth=1.8)

    ax_hour.set_title("Diurnal Profile: Washing Machine Starts by Hour of Day", fontsize=11, fontweight='bold')
    ax_hour.set_xlabel("Hour of Day (Local Time)", fontsize=10)
    ax_hour.set_ylabel("% of House Cycles Started", fontsize=10)
    ax_hour.set_xticks(range(0, 24, 2))
    ax_hour.grid(True, linestyle="--", alpha=0.4)
    ax_hour.legend(fontsize=9)

    # 3B: Day of Week Profile
    ax_dow = axes3[0, 1]
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    x = np.arange(len(days_order))
    width = 0.15

    for idx, hid in enumerate(target_houses):
        c_dow = [c["dayofweek"] for c in house_cycles[hid]]
        counts = [c_dow.count(d) / len(c_dow) * 100 if len(c_dow) > 0 else 0 for d in days_order]
        ax_dow.bar(x + (idx - 2) * width, counts, width, label=f"House {hid}", 
                   color=colors[idx % len(colors)], alpha=0.85)

    ax_dow.set_title("Weekly Profile: Washing Machine Use by Day of Week", fontsize=11, fontweight='bold')
    ax_dow.set_xlabel("Day of Week", fontsize=10)
    ax_dow.set_ylabel("% of Total Cycles", fontsize=10)
    ax_dow.set_xticks(x)
    ax_dow.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=10)
    ax_dow.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_dow.legend(fontsize=9)

    # 3C: Cycle Duration Boxplot
    ax_dur = axes3[1, 0]
    duration_data = [[c["duration_min"] for c in house_cycles[hid]] for hid in target_houses]
    bp_dur = ax_dur.boxplot(duration_data, tick_labels=[f"House {h}" for h in target_houses],
                            patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp_dur['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_dur.set_title("Cycle Duration Distributions Across Households", fontsize=11, fontweight='bold')
    ax_dur.set_ylabel("Duration (Minutes)", fontsize=10)
    ax_dur.grid(True, linestyle="--", alpha=0.4, axis="y")

    # 3D: Energy per Cycle Boxplot
    ax_ene = axes3[1, 1]
    energy_data = [[c["energy_kwh"] for c in house_cycles[hid]] for hid in target_houses]
    bp_ene = ax_ene.boxplot(energy_data, tick_labels=[f"House {h}" for h in target_houses],
                            patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp_ene['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax_ene.set_title("Energy Consumption per Cycle (kWh)", fontsize=11, fontweight='bold')
    ax_ene.set_ylabel("Electrical Energy (kWh)", fontsize=10)
    ax_ene.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    p3 = PLOT_DIR / "wm_cross_house_comparison.png"
    fig3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {p3}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  Washing Machine EDA & Cross-House Analysis Complete!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_eda([1, 2, 5, 7, 20])
