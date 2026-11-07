"""NILM-Flow | Advanced Energy Disaggregation & Appliance Load Profiling Dashboard.

Interactive portfolio application demonstrating end-to-end non-intrusive load disaggregation,
cycle detection, and macro energy attribution on smart-meter telemetry.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- 1. Page Configuration & Theme Settings ---
st.set_page_config(
    page_title="NILM-Flow | Energy Disaggregation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for minimalist aesthetic (off-white, dark text, sharp borders, clean cards)
st.markdown(
    """
    <style>
    /* Global layout & typography */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1300px;
    }
    /* Header & Footer clean SaaS styling without breaking the sidebar toggle */
    #MainMenu {
        visibility: hidden !important;
    }
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        pointer-events: none !important;
    }
    /* Ensure the sidebar toggle button remains visible and interactive */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        pointer-events: auto !important;
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        margin-top: 0.5rem !important;
        margin-left: 0.5rem !important;
    }
    [data-testid="stSidebarCollapsedControl"] button {
        color: #111827 !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover {
        background-color: #F3F4F6 !important;
        border-color: #9CA3AF !important;
    }

    
    /* Clean metric card containers */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.82rem;
        color: #4B5563;
        margin-top: 0.35rem;
    }

    /* Structured content cards */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
    }
    .content-box h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #111827;
        font-size: 1.05rem;
    }
    .content-box p {
        color: #374151;
        font-size: 0.90rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }

    /* Sidebar minimal styling */
    [data-testid="stSidebar"] {
        background-color: #F3F4F1;
        border-right: 1px solid #E5E7EB;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. Paths & Cached Data Ingestion ---
PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


@st.cache_data
def load_inspection_summary():
    path = TABLES_DIR / "raw_house1_inspection.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {
        "raw_record_count": 4377877,
        "duplicate_timestamps": 645514,
        "outages_exceeding_30min": 23,
        "resampled_1min_rows": 513322,
        "resampled_nan_fraction": 0.1466,
    }


@st.cache_data
def load_eda_summary():
    path = TABLES_DIR / "wm_cross_house_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_metrics():
    path = TABLES_DIR / "nilm_model_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# --- 3. Sidebar Navigation ---
st.sidebar.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <span style="font-size: 1.15rem; font-weight: 800; color: #111827; letter-spacing: -0.02em;">⚡ NILM-FLOW</span><br>
        <span style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Energy Disaggregation Engine</span>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.sidebar.radio(
    "Navigation",
    options=[
        "Overview & Methodology",
        "Exploratory Data Analysis",
        "Model Validation & Results",
        "Utility Insights & Physics",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size: 0.80rem; color: #4B5563; line-height: 1.5;">
        <strong>Dataset:</strong> REFIT Electrical Load (UK)<br>
        <strong>Target:</strong> Washing Machine (Wet Load)<br>
        <strong>Nominal Raw:</strong> ~8s Polled Active Power<br>
        <strong>Cleaned Rate:</strong> 1-Minute Resampled<br>
        <strong>Validation:</strong> Unseen Household (House 20)
    </div>
    """,
    unsafe_allow_html=True,
)


# --- 4. Module 1: Overview & Methodology ---
def render_overview():
    st.markdown("## Non-Intrusive Load Monitoring (NILM)")
    st.markdown(
        "<p style='color: #4B5563; font-size: 1.05rem; margin-top: -0.4rem;'>"
        "Disaggregating individual domestic washing machine loads from whole-home smart-meter telemetry."
        "</p>",
        unsafe_allow_html=True,
    )

    insp = load_inspection_summary()

    # Metric Cards Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Analyzed Cohort</div>
                <div class="metric-value">5 Homes</div>
                <div class="metric-subtext">Houses 1, 2, 3, 5, 20 with sub-metering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Raw Telemetry Audited</div>
                <div class="metric-value">{insp.get('raw_record_count', 4377877):,}</div>
                <div class="metric-subtext">House 1 Part 1 (~8s wireless polling)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Resampled Resolution</div>
                <div class="metric-value">1-Minute</div>
                <div class="metric-subtext">Preserves physical energy conservation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Primary Load Type</div>
                <div class="metric-value">2.0–2.4 kW</div>
                <div class="metric-subtext">Resistive heating + drum motor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Engineering Pipeline Architecture")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        st.markdown(
            """
            <div class="content-box">
                <h4>1. Ingestion & Audit</h4>
                <p>Loads raw telemetry (~8s polling). Sorts chronologically and deduplicates multi-channel sensor bursts logging in the same second.</p>
                <code>drop_duplicates(subset=['Time'])</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col2:
        st.markdown(
            """
            <div class="content-box">
                <h4>2. Cleaning & Resampling</h4>
                <p>Resamples to 1-minute uniform bins using mean active power. Bridges packet drops &le;3 min; preserves multi-hour outages as NaN.</p>
                <code>resample('1min').mean()</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col3:
        st.markdown(
            """
            <div class="content-box">
                <h4>3. Feature Engineering</h4>
                <p>Derives rolling means (15m, 60m) to reject 2-minute kettle spikes, and rolling std to capture cyclic drum agitation.</p>
                <code>rolling(15).mean(), std()</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_col4:
        st.markdown(
            """
            <div class="content-box">
                <h4>4. Disaggregation Model</h4>
                <p>Trains an XGBoost gradient boosted tree regressor across 3 households (H1, H2, H5). Evaluates zero-shot on unseen House 20.</p>
                <code>xgb.XGBRegressor(max_depth=5)</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Stage 1 Verification: Raw vs. Resampled Telemetry (House 1)")
    c_img, c_desc = st.columns([2.5, 1.2])
    with c_img:
        plot_path = PLOTS_DIR / "house1_raw_vs_clean.png"
        if not plot_path.exists():
            plot_path = PLOTS_DIR / "house1_raw_vs_clean_48h.png"
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
    with c_desc:
        st.markdown(
            f"""
            <div class="content-box" style="margin-top: 0.5rem;">
                <h4>Telemetry Audit Findings</h4>
                <ul style="font-size: 0.88rem; color: #374151; padding-left: 1.2rem; line-height: 1.6;">
                    <li><strong>Duplicates Dropped:</strong> {insp.get('duplicate_timestamps', 645514):,} rows resolved.</li>
                    <li><strong>Major Outages (>30m):</strong> {insp.get('outages_exceeding_30min', 23)} distinct periods.</li>
                    <li><strong>Max Outage:</strong> {insp.get('max_outage_hours', 998.4):.1f} hours preserved as NaN to avoid phantom consumption.</li>
                    <li><strong>1-Minute Series:</strong> {insp.get('resampled_1min_rows', 513322):,} uniform minutes.</li>
                    <li><strong>Energy Preservation:</strong> Mean power binning satisfies 1 W &times; 60 s = 60 Joules.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- 5. Module 2: Exploratory Data Analysis ---
def render_eda():
    st.markdown("## Exploratory Data Analysis & Cycle Signatures")
    st.markdown(
        "<p style='color: #4B5563; font-size: 1.05rem; margin-top: -0.4rem;'>"
        "Empirical identification of active washing cycles and cross-household behavioral variations."
        "</p>",
        unsafe_allow_html=True,
    )

    eda_df = load_eda_summary()

    # Top metric highlights
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Duty Cycle Inactivity</div>
                <div class="metric-value">92.8% – 98.1%</div>
                <div class="metric-subtext">Washing machines remain idle >92% of all minutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Mean Peak Heating Power</div>
                <div class="metric-value">1,922 W – 2,320 W</div>
                <div class="metric-subtext">Standardized UK internal resistive heating coils</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Cycle Duration Range</div>
                <div class="metric-value">34.6m – 116.6m</div>
                <div class="metric-subtext">Quick wash (H1) vs. extended cycle (H2)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 24-hour trace & cycle detection logic
    st.markdown("### Representative 24-Hour Whole-Home Trace (House 1)")
    trace_path = PLOTS_DIR / "wm_representative_24h_trace.png"
    if trace_path.exists():
        st.image(str(trace_path), use_container_width=True)

    col_code, col_cycle_plot = st.columns([1.2, 1.8])
    with col_code:
        st.markdown("#### Domain Cycle Detection Rule")
        st.markdown(
            """
            Washing cycles are extracted from 1-minute power telemetry using physical domain boundaries:
            """
        )
        st.code(
            """# 1. Standby threshold (>15W separates idle electronics)
# 2. Bridge brief soaking/draining pauses (3 minutes)
is_active = (
    (series_power > 15.0)
    .rolling(3, min_periods=1)
    .max()
    .astype(bool)
)

# 3. Require duration >= 15m and peak >= 500W
#    (validates 2 kW heating coil or spin extraction)
cycle_groups = (~is_active).cumsum()
""",
            language="python",
        )
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #4B5563; line-height: 1.5;">
                This rule rejects momentary button presses and sensor noise blips while maintaining continuity during clothes soaking and water draining.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_cycle_plot:
        st.markdown("#### Detected Cycle Signature Detail")
        cycle_path = PLOTS_DIR / "wm_detected_cycles.png"
        if not cycle_path.exists():
            cycle_path = PLOTS_DIR / "wm_detected_cycles_detail.png"
        if cycle_path.exists():
            st.image(str(cycle_path), use_container_width=True)

    # Cross-Household Comparison Section
    st.markdown("### Cross-Household Comparison (Houses 1, 2, 3, 5, 20)")
    c_tab, c_chart = st.columns([1.4, 1.6])
    with c_tab:
        if not eda_df.empty:
            display_cols = [
                "House",
                "Appliance",
                "Inactive Time (%)",
                "Mean Duration (min)",
                "Mean Peak Power (W)",
                "Mean Energy per Cycle (kWh)",
            ]
            st.dataframe(
                eda_df[display_cols],
                hide_index=True,
                use_container_width=True,
            )
    with c_chart:
        comp_path = PLOTS_DIR / "wm_cross_house_comparison.png"
        if comp_path.exists():
            st.image(str(comp_path), use_container_width=True)


# --- 6. Module 3: Model Validation & Results ---
def render_validation():
    st.markdown("## Disaggregation Model & Leakage-Free Validation")
    st.markdown(
        "<p style='color: #4B5563; font-size: 1.05rem; margin-top: -0.4rem;'>"
        "Evaluating zero-shot disaggregation generalization on an unseen test household (House 20)."
        "</p>",
        unsafe_allow_html=True,
    )

    metrics_df = load_model_metrics()

    # Leakage Prevention Split Diagram
    st.markdown(
        """
        <div class="content-box">
            <h4>Validation Design: Strict Cross-Household Split</h4>
            <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 260px; background: #F9FAFB; padding: 0.75rem 1rem; border-radius: 4px; border: 1px solid #E5E7EB;">
                    <strong style="color: #1F2937;">Training Cohort: Houses 1, 2, 5</strong><br>
                    <span style="font-size: 0.85rem; color: #4B5563;">334,510 1-minute samples | Brands: Beko, LG, AEG | Cycle durations: 35m – 116m</span>
                </div>
                <div style="flex: 1; min-width: 260px; background: #EFF6FF; padding: 0.75rem 1rem; border-radius: 4px; border: 1px solid #BFDBFE;">
                    <strong style="color: #1E40AF;">Test Cohort: House 20 (Unseen)</strong><br>
                    <span style="font-size: 0.85rem; color: #3B82F6;">95,760 1-minute samples | Completely withheld from training</span>
                </div>
            </div>
            <p style="font-size: 0.88rem; color: #4B5563; margin-top: 0.8rem; margin-bottom: 0;">
                <strong>Why this prevents data leakage:</strong> Random train/test splits shuffle adjacent 1-minute readings from the same physical cycle into both sets (temporal autocorrelation $r > 0.85$). In-home splits also allow models to memorize household-specific baseload patterns. Evaluating on an unseen house strictly tests physical feature generalization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">XGBoost MAE</div>
                <div class="metric-value">8.52 W</div>
                <div class="metric-subtext">Mean absolute error across 95k test minutes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">XGBoost RMSE</div>
                <div class="metric-value">87.08 W</div>
                <div class="metric-subtext">Improved from 95.3 W baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Energy Accuracy</div>
                <div class="metric-value">85.35%</div>
                <div class="metric-subtext">14.65% relative energy attribution error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Energy Attribution</div>
                <div class="metric-value">7.60 / 8.91 kWh</div>
                <div class="metric-subtext">Predicted vs. Actual on unseen House 20</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Benchmark Comparison Table
    st.markdown("### Benchmark Comparison Against Reference Baselines")
    if not metrics_df.empty:
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.markdown(
        """
        <div class="content-box">
            <h4>Impact of Class Imbalance: The MAE Sparsity Paradox</h4>
            <p style="font-size: 0.88rem; color: #374151;">
                Notice that the <strong>Always-Off baseline</strong> achieves an MAE of 5.58 W — lower than the XGBoost model (8.52 W). This occurs because washing machines are inactive for >97% of the time; predicting zero everywhere scores 0 error on 97% of timestamps. However, Always-Off fails completely on utility energy tracking (100% missed energy).
            </p>
            <p style="font-size: 0.88rem; color: #374151; margin-bottom: 0;">
                The XGBoost regressor captures genuine load events, achieving an energy attribution accuracy of <strong>85.35%</strong> on an unseen home without overfitting to background baseload noise.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual vs Predicted Trace
    st.markdown("### Disaggregation Trace on Unseen House 20")
    pred_path = PLOTS_DIR / "nilm_actual_vs_predicted.png"
    if not pred_path.exists():
        pred_path = PLOTS_DIR / "nilm_loho_house20_actual_vs_predicted.png"
    if not pred_path.exists():
        pred_path = PLOTS_DIR / "nilm_actual_vs_predicted_test.png"
    if pred_path.exists():
        st.image(str(pred_path), use_container_width=True)

    # Failure Modes
    st.markdown("### Failure Modes & Physical Limits")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.markdown(
            """
            <div class="content-box">
                <h4>1. Competing Resistive Heating Loads</h4>
                <p>Electric kettles and space heaters also draw 2.0–2.5 kW with unity power factor. The 15-minute rolling average rejects short 2-minute kettle boils, but prolonged space heating or repeated cooking events can occasionally induce false-positive predictions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f_col2:
        st.markdown(
            """
            <div class="content-box">
                <h4>2. Low-Power Agitation Blending</h4>
                <p>Following the water-heating phase, drum agitation draws 150–350 W. When background baseload (refrigeration, TVs, lighting) fluctuates simultaneously, these oscillations can blend into the aggregate noise floor, causing slight under-prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- 7. Module 4: Utility Insights & Physics ---
def render_utility_physics():
    st.markdown("## Utility Demand Flexibility & Grid Physics")
    st.markdown(
        "<p style='color: #4B5563; font-size: 1.05rem; margin-top: -0.4rem;'>"
        "Commercial grid value, behavioral demand response, and electrical transfer constraints."
        "</p>",
        unsafe_allow_html=True,
    )

    u_col1, u_col2, u_col3 = st.columns(3)

    with u_col1:
        st.markdown(
            """
            <div class="content-box">
                <h4>1. SmartShift Laundry (DSM)</h4>
                <p><strong>The Opportunity:</strong> Cross-household EDA indicates >60% of laundry cycles coincide with the morning peak (08:00–12:00), with 2.0 kW heating driving ~80% of cycle energy.</p>
                <p><strong>Utility Actions:</strong></p>
                <ul style="font-size: 0.85rem; color: #374151; padding-left: 1.1rem; line-height: 1.5;">
                    <li><strong>Time-of-Day (ToD) Tariff:</strong> 15% discount for shifting wash starts into solar hours (13:00–16:00) or off-peak night windows (22:00–06:00).</li>
                    <li><strong>Eco-Wash Nudges:</strong> Behavioral prompts encouraging 30°C washes, reducing per-cycle heating energy by ~40% (0.20 kWh/cycle).</li>
                </ul>
                <p><strong>M&V Framework:</strong> Randomized Controlled Trial (RCT) with 2,000 households tracking Peak Coincidence Ratio and Average kWh per wash.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with u_col2:
        st.markdown(
            """
            <div class="content-box">
                <h4>2. Grid Physics & Voltage Sags</h4>
                <p><strong>Ohm's & Joule's Law:</strong> Heating elements are fixed resistive conductors (R). Active power scales quadratically with voltage:</p>
                <div style="text-align: center; font-weight: 600; margin: 0.5rem 0; color: #1E3A8A;">
                    P(t) = V(t)² / R
                </div>
                <p><strong>UK vs. India Grid Transfer:</strong></p>
                <ul style="font-size: 0.85rem; color: #374151; padding-left: 1.1rem; line-height: 1.5;">
                    <li>UK voltage is regulated at 230 V (&plusmn;1%). Indian distribution feeders regularly sag to 180–190 V during evening peaks (-20% sag).</li>
                    <li><strong>Ohmic Warping:</strong> (0.80 V)² = 0.64 &rarr; power drops by <strong>36%</strong>. A 2,200 W element collapses to ~1,400 W.</li>
                    <li><strong>Solution:</strong> AMI smart meters must measure V_rms alongside power to dynamically normalize readings.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with u_col3:
        st.markdown(
            """
            <div class="content-box">
                <h4>3. Data Resolution Smearing</h4>
                <p><strong>15m & 30m Smart Meters:</strong> In commercial rollouts, meters report at 15 or 30-minute intervals, integrating active power over time:</p>
                <div style="text-align: center; font-size: 0.88rem; font-weight: 600; margin: 0.5rem 0; color: #1E3A8A;">
                    P̄_30m = (2300W &times; 10m + 150W &times; 20m) / 30m &approx; 867 W
                </div>
                <p><strong>The Smearing Dilemma:</strong></p>
                <ul style="font-size: 0.85rem; color: #374151; padding-left: 1.1rem; line-height: 1.5;">
                    <li>Sharp 2.3 kW step changes and rolling standard deviations are washed out into a shallow ~870 W block.</li>
                    <li>Cycle boundaries and peak wattages are lost, blending into continuous appliances (refrigeration, cooking).</li>
                    <li>Disaggregation must shift from edge-detection regression to energy-balance blind source separation (e.g. NMF).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- 8. Master View Controller ---
if nav_selection == "Overview & Methodology":
    render_overview()
elif nav_selection == "Exploratory Data Analysis":
    render_eda()
elif nav_selection == "Model Validation & Results":
    render_validation()
elif nav_selection == "Utility Insights & Physics":
    render_utility_physics()

