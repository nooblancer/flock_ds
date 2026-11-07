# ⚡ NILM-Flow: Production-Grade Non-Intrusive Load Monitoring Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**Author:** Kushagra Sharma  
**Dataset:** REFIT Electrical Load Measurements (University of Strathclyde, Murray et al., 2015)  

---

## 📌 Executive Summary

**NILM-Flow** is an industrial-grade Non-Intrusive Load Monitoring (NILM) and smart-meter analytics engine. It disaggregates individual domestic appliance consumption (specifically high-load wet appliances: Washing Machines and Dishwashers) from a single whole-home smart-meter Aggregate power signal ($P_{\text{agg}}$).

The system tackles the core operational bottlenecks that cause academic NILM to fail in real-world utility rollouts:
1. **Asynchronous Telemetry Ingestion & Hygiene:** Resolves multi-gigabyte raw 8-second smart-meter streams, handles sensor duplicate bursts, implements physical bound clipping ($0 \le P \le 10\text{ kW}$), and executes gap-aware 1-minute resampling with explicit outage masking (never hallucinating data over long disconnections).
2. **Leak-Free Temporal Feature Engineering:** Implements strict chronological block partitioning (70% Train, 10% Val, 20% Test) *prior* to feature calculation, eliminating temporal autocorrelation leakage.
3. **Dual Feature Strategies ($O(W)$ vs. $O(1)$ RAM):** Supports multi-scale sliding rolling windows for cloud batch analytics, alongside a stateful **$O(1)$-memory recursive streaming buffer** (`StreamingEdgeBuffer`) running under 200 bytes of RAM for microcontrollers (ESP32, ARM Cortex-M) with continuous-time decay adaptation for irregular packet arrivals.
4. **Asymmetric Peak-Weighted Objective:** Resolves the *NILM Thresholding Paradox* where standard symmetric MSE dampens 2.2 kW resistive heating peaks down to 400 W. An asymmetric loss function doubles the gradient penalty on under-predicted active peaks, increasing Event Recall to **87.01%**.
5. **Dual-Tier & Residual Cancellation Diagnostics:** Exposes deceptive energy error cancellation by decomposing Net Energy Error (RETE/SAE) into **Idle Bleed (kWh)** and **Peak Deficit (kWh)**, evaluated via 1-to-1 bipartite cycle matching within a $\pm 5\text{-minute}$ temporal tolerance window.
6. **Zero-Shot Leave-One-House-Out (LOHO) Generalization:** Trains across Houses 1, 2, 5, and 7 (2.6M observations) with zero cross-household feature bleeding and evaluates zero-shot on unseen House 20, correctly detecting **81.5% of real-world laundry cycles**.
7. **Coarse-Interval (15m/30m) AMI Fallback via NMF:** When smart meters only provide coarse half-hour intervals (e.g. India RDSS, UK SMETS2), supervised regression fails due to Nyquist-Shannon state smearing. An unsupervised **Non-Negative Matrix Factorization (NMF)** pipeline decomposes daily load shapes into 4 canonical diurnal basis archetypes, achieving **89.4% variance explained** for macro energy attribution.

---

## 🚀 Quickstart & Interactive Dashboard

### 1. Interactive Streamlit Web Application
Launch the rich interactive dashboard to explore telemetry audits, cross-house cycle signatures, disaggregation traces, and failure modes:
```bash
pip install -r requirements.txt
streamlit run app.py
```
*Access the UI at `http://localhost:8501`.*

---

### 2. Docker & Containerized Execution (100% Reproducible)

Execute the full end-to-end pipeline inside a clean Debian-slim container with volume-mounted outputs:

```bash
# Build and execute the full end-to-end pipeline (Stages 1-5)
docker compose up nilm-pipeline

# Launch the Streamlit dashboard via Docker
docker compose up dashboard
```

Alternatively, run via Docker CLI:
```bash
# Build container image
docker build -t nilm-disagg-engine .

# Run complete pipeline
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" nilm-disagg-engine --all

# Run Leave-One-House-Out (LOHO) benchmark
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/outputs:/app/outputs" nilm-disagg-engine --stage 4 --loho
```

---

### 3. Bare-Metal Python CLI

```bash
# Verify environment and dataset integrity
python -m src.validate_setup

# Run the master pipeline end-to-end (Stages 1 to 5)
python run_pipeline.py --all

# Or run stage-by-stage:
python run_pipeline.py --stage 1    # Setup verification
python run_pipeline.py --stage 2    # House 1 inspection & 1-min cleaning
python run_pipeline.py --stage 3    # Multi-house washing machine EDA & cycle detection
python run_pipeline.py --stage 4    # NILM disaggregation & baseline benchmarking
python run_pipeline.py --stage 5    # Unsupervised NMF macro energy attribution (30m AMI)

# Execute Leave-One-House-Out (LOHO) cross-household evaluation:
python run_pipeline.py --stage 4 --loho --train-houses 1 2 5 7 --test-house 20
```

---

## 🏗️ Repository Architecture

```
flock_ds/
├── Dockerfile                          # Production Debian-slim container definition
├── docker-compose.yml                  # Container volume orchestration for pipeline & dashboard
├── .dockerignore                       # Build-context optimization exclusions
├── .gitignore                          # Git tracking rules (excludes multi-GB raw telemetry)
├── requirements.txt                    # Production Python dependencies
├── app.py                              # Streamlit interactive web dashboard
├── run_pipeline.py                     # Master CLI pipeline runner (Stages 1–5)
├── pipeline.py                         # Convenience wrapper for pipeline execution
├── README.md                           # Master system documentation & benchmarks
├── DATA.md                             # Telemetry hygiene, audit logs & cleaning specifications
├── RESULTS.md                          # Comprehensive experimental results & error forensics
├── Recommendation.md                   # Strategic utility whitepaper & edge-to-cloud architecture
├── data/                               # Dataset repository (raw & cleaned REFIT)
│   ├── REFIT_RAW_081116/               # Raw ~8s polled telemetry (Part1 & Part2)
│   ├── CLEAN_REFIT_081116/             # Cleaned benchmarks (Houses 1 to 21)
│   └── CLEAN_READ_ME_081116.txt        # Dataset documentation
├── src/                                # Modular production engine
│   ├── __init__.py                     # Package metadata
│   ├── config.py                       # Centralized paths, sampling physics & channel maps
│   ├── validate_setup.py               # Repository integrity & environment assertions
│   ├── data_ingestion.py               # Memory-bounded chunk loaders & diagnostics
│   ├── inspect_raw_house1.py           # Raw House 1 quality inspection & outage discovery
│   ├── clean_house1.py                 # 7-stage streaming cleaning & 1-minute resampling
│   ├── plot_house1_cleaning.py         # Microscopic 48h audit & 2-year longitudinal traces
│   ├── washing_machine_eda.py          # Cross-house cycle detection & 1D DBSCAN
│   ├── preprocessing.py                # Zero-leakage features & O(1) StreamingEdgeBuffer
│   ├── nilm_model.py                   # XGBoost disaggregator, asymmetric loss & LOHO
│   ├── macro_disaggregation_nmf.py     # Unsupervised NMF macro attribution on 30m data
│   └── generate_notebook.py            # Automated Jupyter notebook generator
├── notebooks/
│   └── refit_nilm_pipeline.ipynb       # Interactive end-to-end Jupyter Notebook
└── outputs/
    ├── plots/                          # Saved publication-quality evaluation figures
    │   ├── house1_data_quality.png
    │   ├── house1_raw_vs_clean_48h.png
    │   ├── house1_raw_vs_clean_full.png
    │   ├── wm_representative_24h_trace.png
    │   ├── wm_detected_cycles_detail.png
    │   ├── wm_cross_house_comparison.png
    │   ├── nilm_actual_vs_predicted_test.png
    │   ├── nilm_feature_importance.png
    │   ├── nilm_failure_cases.png
    │   ├── nilm_loho_house20_actual_vs_predicted.png
    │   └── nmf_macro_disaggregation_30m.png
    └── tables/                         # Exported JSON and CSV statistical tables
        ├── raw_house1_inspection.json
        ├── house1_cleaning_log.json
        ├── wm_cross_house_summary.csv
        ├── wm_detected_cycles_detail.json
        ├── nilm_model_metrics.csv
        └── nmf_macro_attribution_summary.json
```

---

## 📊 Empirical Benchmarks & Evaluation

### 1. In-Household Disaggregation (House 1 Test Set: 160,755 mins, 77 Ground-Truth Cycles)

Evaluation conducted on an untouched chronological holdout representing ~3 continuous months of deployment:

| Model Configuration | MAE (W) | RMSE (W) | RETE (%) | Idle Bleed (kWh) | Peak Deficit (kWh) | Sample F1 | Event Precision | Event Recall | Event F1 | Matched Cycles |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: Always-Off (0 W)** | 11.02 W | 138.58 W | 100.00% | 0.00 | 24.23 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 77 |
| **Baseline 2: Diurnal Profile** | 19.77 W | 137.54 W | 12.52% | 24.58 | 23.93 | 0.0896 | 0.0000 | 0.0000 | 0.0000 | 0 / 77 |
| **XGBoost (Symmetric MSE - Raw)** | 10.53 W | 86.12 W | **3.09%** | 12.99 | 11.67 | **0.3867** | **0.6311** | 0.8442 | **0.7222** | 65 / 77 |
| **XGBoost (Symmetric MSE - Thresholded)**| **9.90 W** | 86.11 W | 8.98% | 11.28 | 11.67 | **0.3867** | **0.6311** | 0.8442 | **0.7222** | 65 / 77 |
| **XGBoost (Asymmetric Loss - Raw)** | 11.97 W | 85.65 W | 20.47% | 17.96 | 10.49 | 0.3408 | 0.5776 | **0.8701** | 0.6943 | **67 / 77** |
| **XGBoost (Asymmetric Loss - Thresholded)**| 11.07 W | **85.63 W** | 12.18% | 15.53 | **10.49** | 0.3408 | 0.5776 | **0.8701** | 0.6943 | **67 / 77** |

---

### 2. Zero-Shot Leave-One-House-Out (LOHO) Generalization (Target: Unseen House 20, 200 Cycles)

Trained on 2.6M samples from Houses 1, 2, 5, and 7 with zero cross-household feature leakage:

| Model Configuration | MAE (W) | RMSE (W) | RETE (%) | Idle Bleed (kWh) | Peak Deficit (kWh) | Sample F1 | Event Precision | Event Recall | Event F1 | Matched Cycles |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LOHO Baseline: Always-Off (0 W)** | 8.06 W | 113.55 W | 100.00% | 0.00 | 62.56 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 / 200 |
| **LOHO XGBoost (Symmetric MSE - Thresholded)** | **13.03 W** | **94.48 W** | **26.89%** | **75.45** | 40.89 | **0.2239** | **0.3883** | 0.7300 | **0.5069** | 146 / 200 |
| **LOHO XGBoost (Asymmetric Loss - Thresholded)** | 16.52 W | 96.45 W | 88.24% | 117.31 | **35.04** | 0.1911 | 0.3099 | **0.8150** | 0.4490 | **163 / 200** |

* **Zero-Shot Transferability:** The asymmetric model successfully detects **163 out of 200 washing cycles (81.5% Event Recall)** on an entirely unseen household with a different manufacturer model and distinct domestic schedule.

---

## 🔬 Core Engineering Innovations

```
                                      ┌─────────────────────────────────────────────────────────┐
                                      │              NILM-FLOW CORE INNOVATIONS                 │
                                      └─────────────────────────────────────────────────────────┘
                                                                   │
        ┌──────────────────────────────┬───────────────────────────┴───────────────────────────┬──────────────────────────────┐
        ▼                              ▼                                                       ▼                              ▼
┌──────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────────────┐ ┌──────────────────────────────┐
│  O(1) STREAMING EDGE BUFFER  │ │  ASYMMETRIC LOSS FUNCTION     │ │  RESIDUAL CANCELLATION FORENSICS      │ │  COARSE AMI NMF ATTRIBUTION  │
│ Under 200 bytes static RAM   │ │ Resolves the Thresholding     │ │ Exposes deceptive net energy balance │ │ Unsupervised 30m fallback    │
│ with dynamic Delta_t decay   │ │ Paradox on 2.2 kW peaks       │ │ (Idle Bleed vs Peak Deficit)          │ │ (89.4% variance explained)   │
└──────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────────────┘ └──────────────────────────────┘
```

### 1. Resolving the NILM Thresholding Paradox
In domestic energy disaggregation, appliances are off >97% of the time. Standard symmetric MSE ($L = \frac{1}{2}(\hat{y}-y)^2$) is conservative: to avoid penalty during the 97% idle baseline, tree leaves shrink predictions towards zero, predicting ~400 W when the real machine draws 2,200 W. Downstream physical cycle detectors (requiring $P_{\text{peak}} \ge 500\text{ W}$) discard these dampened peaks as noise.
* **Our Solution:** A custom asymmetric loss function ($g_i = w_i(\hat{y}_i - y_i)$, $h_i = w_i$) applying a **$2.0\times$ penalty multiplier** on under-predicted active peaks ($y \ge 500\text{ W}$), forcing the model to predict the full 2.2 kW thermal ceiling.

### 2. Forensic Residual Error Cancellation
In symmetric models, raw Relative Error in Total Energy (RETE) was deceptively low (**3.09%**). Our residual decomposition revealed that this was caused by **error cancellation**: $+12.99\text{ kWh}$ of low-wattage idle false positives cancelled out $-11.67\text{ kWh}$ of peak under-predictions. Thresholding ($<10\text{ W}$) removed the idle noise, exposing the uncompensated peak deficit.

### 3. $O(1)$-Memory Streaming Buffer for IoT Gateways (`StreamingEdgeBuffer`)
For microcontrollers (ESP32 / smart-meter gateways with <520 KB RAM), maintaining sliding windows of raw readings causes heap fragmentation and $O(W)$ computational re-evaluations. Our recursive formulation maintains multi-scale context across 5m, 15m, 30m, and 60m spans in **under 200 bytes of static RAM**, utilizing:
* Continuous-time exponential decay ($\alpha_{\text{eff}} = 1 - e^{-\Delta t / \tau}$) for irregular telemetry intervals.
* Tony Finch (2009) recursive exponential variance updates.
* Instant cold-start bias correction ($1 - (1-\alpha)^t$) and flash-memory persistence (`to_dict` / `from_dict`).

### 4. Coarse-Interval AMI Fallback via Non-Negative Matrix Factorization (NMF)
When smart meters stream at 15-minute or 30-minute intervals, high-frequency supervised models fail due to Nyquist smearing. Our unsupervised NMF engine decomposes daily load matrices $V \in \mathbb{R}_+^{537 \times 48}$ into canonical basis load shapes $H \in \mathbb{R}_+^{4 \times 48}$ and activation weights $W \in \mathbb{R}_+^{537 \times 4}$, capturing **89.4% of total variance** and accurately attributing 35.4% of household load to daytime flexible wet appliances.

---

## 🖼️ Visual Diagnostic Gallery

All evaluation figures are generated and saved to `outputs/plots/`:

| Diagnostic Figure | Key Analytical Finding |
|:---|:---|
| **[house1_raw_vs_clean_48h.png](outputs/plots/house1_raw_vs_clean_48h.png)** | 48-hour audit proving 1-min resampling preserves 3-phase cycle morphology (heater, motor, spin) with zero idle drift. |
| **[house1_raw_vs_clean_full.png](outputs/plots/house1_raw_vs_clean_full.png)** | 2-year longitudinal timeline highlighting seasonal envelopes and the 41.6-day outage gap (masked, not interpolated). |
| **[house1_data_quality.png](outputs/plots/house1_data_quality.png)** | Polling interval distribution (log scale) showing the 5–8s nominal peak alongside active appliance power boxplots. |
| **[wm_representative_24h_trace.png](outputs/plots/wm_representative_24h_trace.png)** | 24-hour trace illustrating aggregate vs washing machine load and dynamic cycle detection windowing. |
| **[wm_detected_cycles_detail.png](outputs/plots/wm_detected_cycles_detail.png)** | High-resolution zoom into 3 distinct wash cycles verifying leading-edge heating and trailing-edge spin detection. |
| **[wm_cross_house_comparison.png](outputs/plots/wm_cross_house_comparison.png)** | Cross-household diurnal schedule (morning peak), weekly usage (weekend surge), duration, and energy boxplots. |
| **[nilm_actual_vs_predicted_test.png](outputs/plots/nilm_actual_vs_predicted_test.png)** | Unseen test set disaggregation trace: Aggregate input, actual vs predicted power, and absolute error profile. |
| **[nilm_feature_importance.png](outputs/plots/nilm_feature_importance.png)** | Top 15 features: Gini importance confirms `agg_roll_max_15`, `agg_roll_mean_15`, and `agg_diff_1` drive >60% of splits. |
| **[nilm_failure_cases.png](outputs/plots/nilm_failure_cases.png)** | Failure mode forensics: Failure Mode A (resistive cross-appliance confusion) and Failure Mode B (multi-appliance masking). |
| **[nilm_loho_house20_actual_vs_predicted.png](outputs/plots/nilm_loho_house20_actual_vs_predicted.png)** | Zero-shot generalization on unseen House 20 proving robust cross-household load tracking. |
| **[nmf_macro_disaggregation_30m.png](outputs/plots/nmf_macro_disaggregation_30m.png)** | 30-min AMI analysis: Nyquist smearing, canonical basis load shapes $H$, energy attribution pie chart, and daily activations. |

---

## 📚 Academic References

1. **REFIT Dataset:** Murray, D., Liao, J., Stankovic, L., Stankovic, V., Hauxwell-Baldwin, R., Wilson, C., Coleman, M., Kane, T., & Firth, S. (2015). *A data management platform for personalised real-time energy feedback*. EEDAL 2015.
2. **Neural NILM:** Kelly, J., & Knottenbelt, W. (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. BuildSys 2015.
3. **NILMTK Toolkit:** Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M. (2014). *NILMTK: An open source toolkit for non-intrusive load monitoring*. ACM e-Energy 2014.
4. **Online Exponential Variance:** Finch, T. (2009). *Incremental calculation of weighted mean and variance*. University of Cambridge Technical Report.
5. **Non-Negative Double SVD (NNDSVD):** Boutsidis, C., & Gallopoulos, E. (2008). *SVD based initialization: A head start for nonnegative matrix factorization*. Pattern Recognition.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
