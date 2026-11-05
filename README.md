# Non-Intrusive Load Monitoring (NILM): REFIT Data Cleaning & Washing Machine Disaggregation Pipeline

**Candidate:** Kushagra Bhatore  
**Role:** Junior Data Scientist  
**Organization:** Flock Energy (Watt Wizards Pvt. Ltd.)  
**Dataset:** REFIT Electrical Load Measurements (University of Strathclyde, Murray et al., 2015)  

---

## 1. Project Overview & Architecture

This repository delivers an end-to-end, production-grade Non-Intrusive Load Monitoring (NILM) and smart-meter analytics pipeline. Built for electricity distribution companies (DISCOMs) and demand-side management teams, this project addresses two core challenges:
1. **Raw Telemetry Ingestion & Robust Cleaning:** Auditing high-frequency (~8-second), asynchronous, raw smart-meter readings for House 1, correcting timestamps, resolving duplicate records, eliminating impossible spikes, and implementing gap-aware 1-minute resampling.
2. **Washing Machine Analytics & Disaggregation Modeling:** Characterizing washing-machine operational cycles across $\ge 4$ households and training a rolling-context gradient boosting model (XGBoost) to accurately disaggregate washing-machine active power from the single whole-home Aggregate meter signal.

```
flock_ds/
├── data/                               # Dataset repository (raw & cleaned REFIT)
│   ├── REFIT_RAW_081116/               # Raw ~8s polled data (Part1 + Part2 per house)
│   ├── CLEAN_REFIT_081116/             # Official cleaned data (20 houses)
│   └── CLEAN_READ_ME_081116.txt        # Dataset documentation
├── src/                                # Modular source code
│   ├── __init__.py
│   ├── config.py                       # Central paths, appliance mappings, constants
│   ├── data_ingestion.py               # Robust loaders & summary diagnostic utilities
│   ├── inspect_raw_house1.py           # Raw House 1 quality inspection script
│   ├── clean_house1.py                 # 7-stage cleaning & 1-minute resampling pipeline
│   ├── plot_house1_cleaning.py         # Before/after cleaning visualization module
│   ├── washing_machine_eda.py          # Cross-house cycle detection & behavioral EDA
│   ├── nilm_model.py                   # XGBoost disaggregation model & baseline benchmarks
│   ├── validate_setup.py               # Repository smoke-test & integrity suite
│   └── generate_notebook.py            # Automated notebook generation script
├── notebooks/
│   └── refit_nilm_pipeline.ipynb       # Interactive end-to-end Jupyter Notebook
├── outputs/
│   ├── house1_cleaned_1min.csv         # Cleaned 1-minute dataset for House 1 (920k rows)
│   ├── plots/                          # Saved publication-quality evaluation figures
│   │   ├── house1_data_quality.png
│   │   ├── house1_raw_vs_clean_48h.png
│   │   ├── house1_raw_vs_clean_full.png
│   │   ├── wm_representative_24h_trace.png
│   │   ├── wm_detected_cycles_detail.png
│   │   ├── wm_cross_house_comparison.png
│   │   ├── nilm_actual_vs_predicted_test.png
│   │   ├── nilm_feature_importance.png
│   │   └── nilm_failure_cases.png
│   └── tables/                         # Exported JSON and CSV statistical tables
│       ├── raw_house1_inspection.json
│       ├── house1_cleaning_log.json
│       ├── wm_cross_house_summary.csv
│       ├── wm_detected_cycles_detail.json
│       └── nilm_model_metrics.csv
├── run_pipeline.py                     # Master one-click CLI pipeline runner
├── requirements.txt                    # Python environment dependencies
├── README.md                           # Repository overview, architecture, trade-offs
├── DATA.md                             # Raw data findings, quality log, cleaning steps
├── RESULTS.md                          # Washing machine EDA, model benchmarks, failure modes
└── Recommendation.md                   # One-page utility-facing executive brief
```

---

## 2. Quickstart & Setup Instructions

### Environment Setup
Clone the repository and install dependencies in Python 3.10+:
```bash
git clone https://github.com/kushagra-bhatore/flock_ds.git
cd flock_ds
pip install -r requirements.txt
```

### Verification & Smoke Test
Run the automated system integrity check to verify dataset paths and mappings:
```bash
python -m src.validate_setup
```

### Executing the Pipeline
You can run the entire pipeline end-to-end or execute individual stages:

```bash
# Run complete end-to-end pipeline (Stages 1-4)
python run_pipeline.py --all

# Or run stage-by-stage:
python run_pipeline.py --stage 1    # Setup verification
python run_pipeline.py --stage 2    # House 1 raw inspection & 1-min cleaning
python run_pipeline.py --stage 3    # Multi-house washing machine EDA & cycle detection
python run_pipeline.py --stage 4    # NILM disaggregation model & baseline benchmarking
```

### Interactive Exploration
Launch the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/refit_nilm_pipeline.ipynb
```

---

## 3. Assumptions & Preprocessing Decisions

1. **Active Power Focus:** Data is measured strictly as active power in Watts (W). Reactive power and power factor are not present in REFIT, bounding disaggregation to active load signatures.
2. **Deduplication Priority:** House 1 raw data contains an overlap between Part 1 (where absent sensors were recorded as 0) and Part 2 (where absent sensors were recorded as `NaN`). When duplicate timestamps occur, **Part 2 data is strictly prioritized** to preserve genuine `NaN` semantics and avoid conflating sensor downtime with active 0 W idle states.
3. **Gap-Aware Missing-Value Handling:** Short gaps ($\le 5\text{ minutes}$ / $\le 37$ consecutive readings) are linearly interpolated as plausible transient bridges. Longer gaps ($> 5\text{ minutes}$, including the 41.6-day outage in early 2014) are **flagged in an explicit `Outage` column and never interpolated**, ensuring synthetic data is never hallucinated.
4. **1-Minute Mean Resampling:** Instantaneous readings (~8-second polling) are aggregated into 1-minute bins using the arithmetic mean. Mean power over 1 minute is directly proportional to cumulative electrical energy ($\text{kWh} = \sum P_{\text{mean}} \times \frac{1}{60 \times 1000}$), smoothing high-frequency inductive phase noise while preserving rectangular thermal heating pulses.
5. **Physical Bound Clipping:** Active power cannot be negative ($P \ge 0\text{ W}$); spurious sub-meter readings exceeding 10,000 W (up to 98 kW in raw logs) are clipped to `NaN`.

---

## 4. Validation Design: Eliminating Temporal Leakage

A fundamental flaw in naive machine-learning workflows for time-series is random train/test shuffling. In NILM, random splitting introduces catastrophic data leakage because:
* Adjacent 1-minute power samples are highly autocorrelated ($r > 0.85$).
* Rolling-window context features (e.g. 15-minute moving average) span across consecutive time steps; random splitting allows test samples to directly inform training features.

To ensure strict real-world generalization, we enforce a **contiguous chronological block split** on REFIT House 1 (803,773 valid 1-minute observations):
* **Training Set (70%):** 562,641 minutes (2013-10-09 to 2014-12-26).
* **Validation Set (10%):** 80,377 minutes (2014-12-26 to 2015-02-20) for early stopping.
* **Evaluation / Test Set (20%):** 160,755 continuous minutes (2015-02-20 to 2015-07-10). Active wash minutes account for only **1.98% (3,187 minutes)** of the test set.

---

## 5. Modeling Approach & Architecture Trade-Offs

### Engineered Feature Matrix
Because a single static power measurement cannot differentiate a 2,000 W kettle (duration ~3 minutes) from a 2,000 W washing machine heater (duration ~15 minutes followed by cyclic motor agitation), we engineered **28 contextual features**:
* **Temporal Context Window:** Symmetric lag and lead features ($t \pm 1, 2, 3, 5, 10, 15\text{ minutes}$).
* **Multi-Scale Rolling Statistics:** Centered moving mean, standard deviation, min, max, and peak-to-peak dynamic range over 5m, 15m, 30m, and 60m windows.
* **Instantaneous Gradient:** Step differences ($\Delta P$) at 1m, 5m, and 15m intervals.
* **Calendar Priors:** Hour-of-day cyclical sine/cosine encodings and weekend flags.

### Architecture Trade-Offs
* **XGBoost vs. Deep Sequence Models (Seq2Point / CNN-LSTM):** While Seq2Point architectures (Kelly & Knottenbelt, 2018) are popular in academic literature, tree-based gradient boosting (XGBoost) with explicit multi-scale rolling statistics trains in **under 30 seconds on CPU**, requires zero GPU infrastructure, delivers clear feature interpretability (Gini importance), and achieves comparable or superior energy attribution accuracy.
* **Regression-Only Metrics vs. Event Detection:** Under extreme class imbalance (~98% inactive), the naive **Always-Off baseline achieves a deceptive MAE of 11.02 W** despite detecting zero cycles ($F_1 = 0$) and missing 100% of consumed electrical energy! We therefore evaluate across dual dimensions:
  * **Regression Metrics:** MAE, RMSE, and Relative Error in Total Energy (RETE).
  * **Event Classification Metrics:** Precision, Recall, and $F_1$-Score at the active threshold ($P \ge 15\text{ W}$).

### Empirical Performance Summary (Unseen Test Set)

| Model | MAE (W) | RMSE (W) | $F_1$-Score | Precision | Recall | Energy Error (RETE %) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: Always-Off (0 W)** | 11.02 W | 138.58 W | 0.0000 | 0.0000 | 0.0000 | 100.00% |
| **Baseline 2: Diurnal Profile** | 19.77 W | 137.54 W | 0.0896 | 0.0480 | 0.6803 | 12.52% |
| **XGBoost Regressor (Raw)** | **9.55 W** | **84.68 W** | **0.4122** | **0.2656** | **0.9194** | **5.35%** |
| **XGBoost Regressor (Thresholded)**| **8.94 W** | **84.67 W** | **0.4122** | **0.2656** | **0.9194** | **10.97%** |

* **Key Takeaway:** XGBoost captures **91.94% of active operating minutes** and estimates cumulative test energy with **94.65% accuracy (RETE of 5.35%)**, outperforming heuristic baselines by 38.9% in RMSE.

---

## 6. AI Tools Used

In accordance with academic integrity and assignment guidelines:
* **Antigravity (AI Pair-Programming Agent):** Utilized for pipeline scaffolding, shell script orchestration, unit-testing syntax, and LaTeX documentation formatting.
* **Claude Opus 4.6 & Gemini 3.8 Flash:** Employed for exploratory brainstorming of physical cycle detection thresholds, vectorizing numpy array operations for high-speed gap bridging, and cross-validating econometric Difference-in-Differences formulations.
* **Verification Protocol:** All generated code, analytical thresholds, and mathematical logic were independently checked, tested via smoke-test scripts, and evaluated against raw ground-truth telemetry.

---

## 7. Possible Future Improvements

1. **Multimodal Disaggregation (Real + Reactive Power):** Incorporating reactive power (VAR) or apparent power (VA) would enable the model to distinguish purely resistive 2.2 kW loads (electric ovens/kettles, $\text{PF} \approx 1.0$) from inductive washing machine electric motors ($\text{PF} \approx 0.75\text{--}0.85$), dramatically boosting precision.
2. **Semi-Supervised Domain Adaptation for Indian DISCOMs:** Applying adversarial domain adaptation (DANN) or synthetic profile generation to transfer UK-learned temporal dynamics to Indian top-loading non-heated washing machines and high-temperature environments.
3. **Sequence-to-Point Temporal Convolutions:** Benchmarking 1D Dilated Temporal Convolutional Networks (TCN) or lightweight Transformer attention layers for multi-state cycle tracking over 120-minute receptive fields.

---

## 8. Academic Citations & References

1. **REFIT Dataset:** Murray, D., Liao, J., Stankovic, L., Stankovic, V., Hauxwell-Baldwin, R., Wilson, C., Coleman, M., Kane, T., & Firth, S. (2015). *A data management platform for personalised real-time energy feedback*. In Proceedings of the 8th International Conference on Energy Efficiency in Domestic Appliances and Lighting (EEDAL).
2. **Seq2Point Disaggregation:** Kelly, J., & Knottenbelt, W. (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. In Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments (BuildSys), pp. 55–64.
3. **NILM Benchmark Metrics:** Batra, N., Kelly, J., Parson, O., Dutta, H., Knottenbelt, W., Rogers, A., Singh, A., & Srivastava, M. (2014). *NILMTK: An open source toolkit for non-intrusive load monitoring*. In Proceedings of the 5th International Conference on Future Energy Systems (e-Energy), pp. 265–276.
