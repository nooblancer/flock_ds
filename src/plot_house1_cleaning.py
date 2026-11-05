"""
Generate before/after comparison plots for House 1 cleaning pipeline.
"""
import sys, os
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["MPLCONFIGDIR"] = os.path.expanduser("~/.matplotlib_tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from src.config import OUTPUT_DIR, PLOT_DIR, RAW_DIR, TABLE_DIR

def main():
    print("[Plots] Loading cleaned 1-min data ...")
    clean_path = OUTPUT_DIR / "house1_cleaned_1min.csv"
    df_clean = pd.read_csv(clean_path, parse_dates=["Time"], index_col="Time")
    print(f"  Clean data shape: {df_clean.shape}, range: {df_clean.index.min()} to {df_clean.index.max()}")

    # Find a great 48h window where Washing Machine (Appliance5) is active
    # Resample daily max of Appliance5
    wm_daily = df_clean["Appliance5"].resample("D").max()
    active_days = wm_daily[wm_daily > 1000].index
    print(f"  Found {len(active_days)} days with high washing machine activity (>1000W)")
    
    # Select a representative active period
    # Let's pick 2014-04-10 or another prominent active period
    target_date = active_days[len(active_days) // 4]
    zoom_start = pd.Timestamp(target_date).floor("D")
    zoom_end = zoom_start + pd.Timedelta(hours=48)
    print(f"  Selected 48h zoom window: {zoom_start} to {zoom_end}")

    # Load raw data for this 48h period specifically
    print("[Plots] Loading raw data for the 48h window ...")
    raw_parts = []
    for p_name in ["RAW_House1_Part1.csv", "RAW_House1_Part2.csv"]:
        p_path = RAW_DIR / p_name
        # Read in chunks to find relevant dates quickly without loading entire file in memory
        for chunk in pd.read_csv(p_path, chunksize=100000):
            chunk["Time"] = pd.to_datetime(chunk["Time"])
            sub = chunk[(chunk["Time"] >= zoom_start - pd.Timedelta(hours=1)) & 
                        (chunk["Time"] <= zoom_end + pd.Timedelta(hours=1))]
            if not sub.empty:
                raw_parts.append(sub)

    if raw_parts:
        df_raw_zoom = pd.concat(raw_parts, ignore_index=True).sort_values("Time").drop_duplicates(subset="Time")
        df_raw_zoom = df_raw_zoom.set_index("Time")
    else:
        df_raw_zoom = pd.DataFrame()
    print(f"  Raw rows in 48h window: {len(df_raw_zoom)}")

    channels = [
        ("Aggregate", "Whole-House Aggregate", "Power (W)"),
        ("Appliance5", "Washing Machine (Appliance 5 - Beko)", "Power (W)"),
        ("Appliance6", "Dishwasher (Appliance 6 - Bosch)", "Power (W)"),
    ]

    # ────────────────────────────────────────────────────────────────────────
    # Plot 1: 48-Hour High-Resolution Before vs After Comparison
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating 48h before vs after comparison ...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col', sharey='row')
    fig.suptitle(f"REFIT House 1: Raw (~8s) vs. Cleaned & Resampled (1-min)\n48-Hour Trace ({zoom_start.strftime('%Y-%m-%d')} to {zoom_end.strftime('%Y-%m-%d')})", 
                 fontsize=14, fontweight='bold', y=0.98)

    for i, (col, title, ylabel) in enumerate(channels):
        # Raw plot (Left column)
        ax_raw = axes[i, 0]
        if not df_raw_zoom.empty and col in df_raw_zoom.columns:
            ax_raw.plot(df_raw_zoom.index, df_raw_zoom[col], color="#1f77b4", linewidth=0.8, alpha=0.85, label="Raw (~8s)")
        ax_raw.set_title(f"RAW: {title}", fontsize=11, fontweight='semibold')
        ax_raw.set_ylabel(ylabel, fontsize=10)
        ax_raw.grid(True, linestyle="--", alpha=0.4)
        ax_raw.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

        # Cleaned plot (Right column)
        ax_clean = axes[i, 1]
        clean_sub = df_clean.loc[zoom_start:zoom_end]
        if col in clean_sub.columns:
            ax_clean.plot(clean_sub.index, clean_sub[col], color="#d62728" if "Washing" in title else "#2ca02c", 
                         linewidth=1.0, alpha=0.9, label="Cleaned (1-min)")
        ax_clean.set_title(f"CLEANED: {title}", fontsize=11, fontweight='semibold')
        ax_clean.grid(True, linestyle="--", alpha=0.4)
        ax_clean.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout()
    p1 = PLOT_DIR / "house1_raw_vs_clean_48h.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 2: Full Study Timeline (Aggregate, WM, Dishwasher) & Outage Map
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating full timeline comparison & quality trace ...")
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig2.suptitle("REFIT House 1: Cleaned 1-Minute Longitudinal Energy Traces & Outage Flags (2013-2015)",
                  fontsize=14, fontweight='bold', y=0.98)

    # 1. Aggregate
    axes2[0].plot(df_clean.index, df_clean["Aggregate"], color="#1f77b4", linewidth=0.4, alpha=0.7)
    axes2[0].set_ylabel("Aggregate (W)", fontsize=11)
    axes2[0].set_title("Household Aggregate Power (Cleaned 1-min)", fontsize=11, fontweight='semibold')
    axes2[0].grid(True, linestyle="--", alpha=0.4)
    axes2[0].set_ylim(0, 10000)

    # 2. Washing Machine & Dishwasher overlay
    axes2[1].plot(df_clean.index, df_clean["Appliance5"], color="#d62728", linewidth=0.5, alpha=0.75, label="Washing Machine (Beko)")
    axes2[1].plot(df_clean.index, df_clean["Appliance6"], color="#2ca02c", linewidth=0.5, alpha=0.6, label="Dishwasher (Bosch)")
    axes2[1].set_ylabel("Appliance Power (W)", fontsize=11)
    axes2[1].set_title("Sub-metered Appliances: Washing Machine vs Dishwasher", fontsize=11, fontweight='semibold')
    axes2[1].grid(True, linestyle="--", alpha=0.4)
    axes2[1].legend(loc="upper right", framealpha=0.9)
    axes2[1].set_ylim(0, 3500)

    # 3. Outage Flag / Data Quality
    # Resample outage percentage to daily
    daily_outage = df_clean["Outage"].resample("D").mean() * 100
    axes2[2].fill_between(daily_outage.index, 0, daily_outage.values, color="#e377c2", alpha=0.6, step="mid")
    axes2[2].set_ylabel("Outage Rate (%)", fontsize=11)
    axes2[2].set_title("Data Quality Diagnostics: Daily Missing / Outage Percentage (Flagged, Not Blindly Filled)", 
                       fontsize=11, fontweight='semibold')
    axes2[2].set_ylim(0, 105)
    axes2[2].grid(True, linestyle="--", alpha=0.4)
    axes2[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    p2 = PLOT_DIR / "house1_raw_vs_clean_full.png"
    fig2.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ────────────────────────────────────────────────────────────────────────
    # Plot 3: Data Quality Diagnostics (Spike removal & Interval Distribution)
    # ────────────────────────────────────────────────────────────────────────
    print("[Plots] Generating data quality diagnostic plot ...")
    fig3, (ax_int, ax_box) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Load interval distribution from inspection json
    import json
    insp_path = TABLE_DIR / "raw_house1_inspection.json"
    with open(insp_path, "r") as f:
        insp = json.load(f)
    
    dist = insp["interval_distribution"]
    cats = list(dist.keys())
    counts = [dist[c] for c in cats]
    
    bars = ax_int.bar(cats, counts, color="#3b528b", edgecolor="black", alpha=0.85)
    ax_int.set_title("Distribution of Raw Polling Intervals", fontsize=12, fontweight='bold')
    ax_int.set_xlabel("Interval Bin", fontsize=10)
    ax_int.set_ylabel("Reading Count (Log Scale)", fontsize=10)
    ax_int.set_yscale("log")
    ax_int.tick_params(axis='x', rotation=45)
    ax_int.grid(True, linestyle="--", alpha=0.4, which="both")
    # Annotate the nominal peak
    for bar in bars:
        height = bar.get_height()
        if height > 1_000_000:
            ax_int.annotate(f"{height/1e6:.1f}M",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Boxplot of Cleaned 1-minute Appliance Power Distributions (Non-Zero Operating Power)
    power_active = {
        "Aggregate": df_clean["Aggregate"].dropna(),
        "Washing Machine": df_clean["Appliance5"][df_clean["Appliance5"] > 5].dropna(),
        "Dishwasher": df_clean["Appliance6"][df_clean["Appliance6"] > 5].dropna(),
        "Fridge": df_clean["Appliance1"][df_clean["Appliance1"] > 5].dropna(),
    }
    ax_box.boxplot([s.values for s in power_active.values()], tick_labels=list(power_active.keys()), 
                   patch_artist=True, boxprops=dict(facecolor="#5ec962", alpha=0.7),
                   medianprops=dict(color="black", linewidth=1.5))
    ax_box.set_title("Active Power Distribution by Channel (Cleaned 1-Min)", fontsize=12, fontweight='bold')
    ax_box.set_ylabel("Power (W)", fontsize=10)
    ax_box.set_yscale("log")
    ax_box.grid(True, linestyle="--", alpha=0.4, which="both")

    plt.tight_layout()
    p3 = PLOT_DIR / "house1_data_quality.png"
    fig3.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {p3}")

    print("[*] All plots successfully generated!")

if __name__ == "__main__":
    main()

