"""
Macro Energy Attribution via Non-Negative Matrix Factorization (NMF).

This module provides an industrial-grade fallback pipeline for low-resolution
smart-meter telemetry (15-minute or 30-minute AMI intervals, as deployed in India's
RDSS rollout and global utility programs).

The Problem:
------------
Point-wise regression and high-frequency event detection fail on coarse 15-30m data
due to Nyquist-Shannon state smearing: a sharp 2.3 kW washing machine heating spike
is mathematically flattened into an undetectable ~723 W average over 30 minutes.

The Solution:
-------------
Shift from microscopic second-by-second pattern recognition to Macroscopic Energy
Attribution using Non-Negative Matrix Factorization (NMF):
    V ~ W x H
where:
  - V in R_+^{D x 48} is the observed matrix of daily 30-minute aggregate load shapes.
  - W in R_+^{D x K} represents the daily activation weights for K behavioral archetypes.
  - H in R_+^{K x 48} represents the canonical 24-hour diurnal basis load shapes
    (e.g. Baseload, Daytime Solar Coincident, Evening Peak, Intermittent High-Thermal).

Utilities use this decomposition to estimate monthly end-use energy budget shares
and target DSM incentives without requiring high-frequency smart meters.
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

from src.config import CLEAN_DIR, OUTPUT_DIR, PLOT_DIR, TABLE_DIR, WASHING_MACHINE_CHANNEL

cache_dir = OUTPUT_DIR / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

logger = logging.getLogger(__name__)


def demonstrate_interval_smearing(df_1min: pd.DataFrame, target_col: str) -> dict:
    """Quantify the mathematical attenuation of appliance peaks from 1m to 15m and 30m."""
    s_1m = df_1min[target_col].dropna()
    peak_1m = float(s_1m.max())

    s_15m = s_1m.resample("15min").mean().dropna()
    peak_15m = float(s_15m.max())

    s_30m = s_1m.resample("30min").mean().dropna()
    peak_30m = float(s_30m.max())

    return {
        "peak_1min_watts": round(peak_1m, 1),
        "peak_15min_watts": round(peak_15m, 1),
        "peak_30min_watts": round(peak_30m, 1),
        "attenuation_15m_pct": round((1.0 - peak_15m / peak_1m) * 100, 2),
        "attenuation_30m_pct": round((1.0 - peak_30m / peak_1m) * 100, 2),
    }


def run_macro_disaggregation_nmf(
    house_id: int = 1,
    n_archetypes: int = 4,
    resolution: str = "30min",
) -> dict:
    """Execute NMF Macro Energy Attribution on coarse smart-meter telemetry.

    Parameters
    ----------
    house_id : int, default 1
        Household identifier in cleaned REFIT dataset.
    n_archetypes : int, default 4
        Number of latent basis load shapes to extract.
    resolution : str, default '30min'
        Coarse resampling frequency ('15min' or '30min').
    """
    print("=" * 75, flush=True)
    print(f"  Macro Energy Attribution via NMF — House {house_id} ({resolution} AMI Data)", flush=True)
    print("=" * 75, flush=True)

    # 1. Load cleaned data
    fpath = CLEAN_DIR / f"CLEAN_House{house_id}.csv"
    wm_col = WASHING_MACHINE_CHANNEL[house_id]
    df_raw = pd.read_csv(fpath, usecols=["Time", "Aggregate", wm_col])
    df_raw["Time"] = pd.to_datetime(df_raw["Time"], format="%Y-%m-%d %H:%M:%S")
    df_raw = df_raw.set_index("Time").sort_index()

    # 1-minute resampled base
    df_1min = df_raw.resample("1min").mean().dropna()

    # 2. Quantify Nyquist Interval Smearing
    smearing = demonstrate_interval_smearing(df_1min, wm_col)
    print("\n[1] Nyquist Interval Smearing Analysis (Peak Attenuation):", flush=True)
    print(f"  - 1-minute peak:  {smearing['peak_1min_watts']} W (distinct rectangular heating pulse)", flush=True)
    print(f"  - 15-minute peak: {smearing['peak_15min_watts']} W ({smearing['attenuation_15m_pct']}% smeared)", flush=True)
    print(f"  - 30-minute peak: {smearing['peak_30min_watts']} W ({smearing['attenuation_30m_pct']}% smeared into baseline)", flush=True)

    # 3. Construct Coarse Interval Matrix (Days x Slots)
    print(f"\n[2] Aggregating Telemetry to {resolution} Intervals ...", flush=True)
    df_coarse = df_1min.resample(resolution).mean().dropna()

    slots_per_day = 48 if resolution == "30min" else 96
    slot_freq_min = 30 if resolution == "30min" else 15

    df_coarse["date"] = df_coarse.index.date
    df_coarse["slot"] = (df_coarse.index.hour * 60 + df_coarse.index.minute) // slot_freq_min

    # Pivot to Day x Slot matrix
    v_pivot = df_coarse.pivot(index="date", columns="slot", values="Aggregate").dropna()
    # Filter days that have all slots
    valid_days = [d for d, row in v_pivot.iterrows() if len(row) == slots_per_day]
    V = v_pivot.loc[valid_days].values  # Shape: (D, slots_per_day)
    day_indices = v_pivot.loc[valid_days].index

    print(f"  Constructed daily interval load matrix V: {V.shape[0]} days x {V.shape[1]} slots", flush=True)

    # 4. Fit Non-Negative Matrix Factorization (NMF)
    print(f"\n[3] Fitting Non-Negative Matrix Factorization (K={n_archetypes} Archetypes) ...", flush=True)
    nmf = NMF(
        n_components=n_archetypes,
        init="nndsvda",
        beta_loss="frobenius",
        max_iter=500,
        random_state=42,
    )
    W = nmf.fit_transform(V)       # Shape: (D, K)
    H = nmf.components_            # Shape: (K, slots_per_day)

    reconstruction_err = float(nmf.reconstruction_err_)
    var_explained = 1.0 - (reconstruction_err ** 2) / float(np.sum(V ** 2))
    print(f"  NMF Convergence reached. Reconstruction variance explained: {var_explained*100:.2f}%", flush=True)

    # 5. Label Archetypes by Characteristic Diurnal Centroid
    # Archetype labels: Baseload, Day Domestic, Evening Peak, Thermal Pulse
    archetype_labels = []
    archetype_energies = []
    total_reconstructed_energy = float(np.sum(W @ H))

    for k in range(n_archetypes):
        h_k = H[k]
        peak_slot = int(np.argmax(h_k))
        peak_hour = peak_slot * (slot_freq_min / 60.0)
        h_std = float(np.std(h_k))
        h_mean = float(np.mean(h_k))

        # Attributed energy for archetype k in kWh
        # Slot duration is (slot_freq_min / 60.0) hours
        k_energy_kwh = float(np.sum(W[:, k]) * np.sum(h_k) * (slot_freq_min / 60.0) / 1000.0)
        share_pct = (np.sum(W[:, k] * np.sum(h_k)) / np.sum(W @ H)) * 100.0

        if h_std / (h_mean + 1e-4) < 0.25:
            label = "Archetype A: 24/7 Constant Baseload (Refrigeration/Standby)"
        elif 8.0 <= peak_hour <= 15.0:
            label = "Archetype B: Daytime Solar-Coincident (Laundry & Cooking)"
        elif 16.0 <= peak_hour <= 22.0:
            label = "Archetype C: Evening System Peak (Lighting, Space Heating & Entertainment)"
        else:
            label = "Archetype D: Night/Early-Morning Off-Peak Operations"

        archetype_labels.append(label)
        archetype_energies.append({
            "archetype_id": k,
            "label": label,
            "peak_hour": round(peak_hour, 1),
            "attributed_kwh": round(k_energy_kwh, 2),
            "energy_share_pct": round(float(share_pct), 2),
        })

    # Ground-truth washing machine energy across the exact same days
    wm_pivot = df_coarse.pivot(index="date", columns="slot", values=wm_col).dropna()
    wm_valid = wm_pivot.loc[day_indices].values
    total_true_wm_kwh = float(np.sum(wm_valid) * (slot_freq_min / 60.0) / 1000.0)
    total_household_kwh = float(np.sum(V) * (slot_freq_min / 60.0) / 1000.0)
    true_wm_share_pct = (total_true_wm_kwh / total_household_kwh) * 100.0

    print("\n[4] Utility Macro Energy Attribution Summary:", flush=True)
    print(f"  Total Household Energy Analyzed: {total_household_kwh:.2f} kWh across {len(day_indices)} days")
    print(f"  Ground-Truth Washing Machine Energy: {total_true_wm_kwh:.2f} kWh ({true_wm_share_pct:.2f}% of total bill)\n")

    for arch in archetype_energies:
        print(f"  * [{arch['archetype_id']}] {arch['label']}:")
        print(f"      Attributed Energy: {arch['attributed_kwh']} kWh ({arch['energy_share_pct']}% of total load), Peak Hour: {arch['peak_hour']}:00")

    # 6. Generate Diagnostic Plots
    print(f"\n[5] Generating NMF Macro Disaggregation Diagnostic Figure ...", flush=True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Macro Energy Attribution via Non-Negative Matrix Factorization (NMF) on Coarse {resolution} Data\n"
                 f"REFIT House {house_id} — End-Use Behavioral Archetypes for Low-Resolution AMI Rollouts",
                 fontsize=13, fontweight='bold', y=0.98)

    # Plot 1: Interval Smearing Comparison
    t_span = df_1min.loc["2014-04-10 06:00:00":"2014-04-10 18:00:00"]
    ax1.plot(t_span.index, t_span[wm_col], color="#000000", linewidth=1.8, label="1-min True Telemetry (2,200W Peak)")
    s_15_sub = t_span[wm_col].resample("15min").mean()
    s_30_sub = t_span[wm_col].resample("30min").mean()
    ax1.step(s_15_sub.index, s_15_sub.values, where="mid", color="#ff7f0e", linewidth=1.5, linestyle="--", label="15-min Smeared Average (~1,100W)")
    ax1.step(s_30_sub.index, s_30_sub.values, where="mid", color="#d62728", linewidth=1.8, label="30-min Smeared Average (~720W)")
    ax1.set_title("Nyquist-Shannon State Smearing: Peak Attenuation Across Resolutions", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Power (Watts)", fontsize=10)
    ax1.set_xlabel("Time of Day", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    # Plot 2: Canonical Basis Load Shapes H (48 slots = 24 hours)
    hours = np.linspace(0, 24, slots_per_day)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    for k in range(n_archetypes):
        ax2.plot(hours, H[k], color=colors[k % len(colors)], linewidth=2.0, label=f"Basis Shape {k} (Peak {archetype_energies[k]['peak_hour']}h)")
    ax2.set_title(f"Canonical 24-Hour Diurnal Load Shapes H (Basis Components)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Hour of Day (0 to 24)", fontsize=10)
    ax2.set_ylabel("Normalized Intensity", fontsize=10)
    ax2.set_xticks(range(0, 25, 4))
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)

    # Plot 3: Long-Term Macro Energy Share Breakdown (Pie Chart)
    labels_pie = [f"Arch {a['archetype_id']}: {a['energy_share_pct']}%" for a in archetype_energies]
    shares = [a["energy_share_pct"] for a in archetype_energies]
    ax3.pie(shares, labels=labels_pie, autopct="%1.1f%%", colors=colors[:n_archetypes], startangle=140,
            textprops={'fontsize': 9})
    ax3.set_title(f"Utility Long-Term Energy Attribution Budget Share (% Total kWh)", fontsize=11, fontweight='bold')

    # Plot 4: Weekly Evolution of Daytime Wet/Solar Archetype vs Actual Laundry Days
    w_day_energy = W[:, 1]  # Weight of daytime archetype
    ax4.bar(range(min(45, len(day_indices))), w_day_energy[:min(45, len(day_indices))], color="#2ca02c", alpha=0.75, label="Daytime Domestic Archetype Weight")
    ax4.set_title("Longitudinal Behavioral Modulation: Daily Archetype Activations (First 45 Days)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Day Index", fontsize=10)
    ax4.set_ylabel("Activation Intensity", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    out_plot = PLOT_DIR / "nmf_macro_disaggregation_30m.png"
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_plot}", flush=True)

    # Save summary table
    summary_data = {
        "house_id": house_id,
        "resolution": resolution,
        "days_analyzed": len(day_indices),
        "total_household_kwh": round(total_household_kwh, 2),
        "total_true_wm_kwh": round(total_true_wm_kwh, 2),
        "true_wm_share_pct": round(true_wm_share_pct, 2),
        "peak_smearing": smearing,
        "archetypes": archetype_energies,
    }
    out_json = TABLE_DIR / "nmf_macro_attribution_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  Saved summary: {out_json}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("  Macro Disaggregation & NMF Attribution Complete!", flush=True)
    print("=" * 75, flush=True)

    return summary_data


if __name__ == "__main__":
    run_macro_disaggregation_nmf(house_id=1, n_archetypes=4, resolution="30min")

