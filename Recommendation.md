# Practical Implications & Utility Recommendations: Scalable Non-Intrusive Load Monitoring (NILM)

**Prepared for:** Distribution Companies (DISCOMs), Demand-Side Management (DSM) Teams & Utility Analytics  
**Context:** Flock Energy Take-Home Assignment — Section 4 Deliverable  
**Author:** Kushagra Bhatore  

---

## 1. Executive Summary & Suitable Uses of the Disaggregation Model

Modern power distribution utilities face unprecedented peak-demand challenges driven by peak load spikes, localized transformer overloads, and high evening power procurement costs. Non-Intrusive Load Monitoring (NILM) enables utilities to derive appliance-level visibility from whole-home smart-meter telemetry without deploying costly in-home sub-metering hardware.

Our empirical evaluation on longitudinal REFIT telemetry demonstrated that an optimized gradient boosting pipeline with rolling temporal context achieves:
* **91.9% event recall** in detecting active washing machine cycles.
* **5.35% relative error in cumulative electrical energy** (estimating 27.94 kWh vs. 29.52 kWh ground truth on unseen test data).
* **38.9% reduction in Root Mean Squared Error (RMSE)** compared to heuristic baselines.

### Where the Model Should Be Deployed
* **Targeted Demand-Side Management (DSM) Audits:** Identifying households with high washing machine frequencies (>7 cycles/week) and significant daytime peak-coincident usage for targeted demand flexibility.
* **Energy-Bill Breakdown & Customer Engagement:** Providing end-consumers with itemized monthly consumption estimates on utility mobile applications and web dashboards, increasing bill transparency and customer satisfaction.
* **Time-of-Day (ToD) Tariff Optimization:** Evaluating consumer responsiveness to differential pricing by tracking appliance shifts into off-peak solar/wind generation hours.

### Limits of the Model
* **Sub-Minute Phase Disaggregation:** The 1-minute model cannot reliably isolate low-power internal sub-components (e.g. 150 W drum agitation vs. water drainage pumps) when other household background loads (refrigeration, electronics) oscillate simultaneously.
* **Standalone Real-Time Billing:** Because the model exhibits a precision of ~27% due to occasional false positives from other resistive heating appliances (electric ovens, kettles), it is suited for **statistical aggregate reporting, monthly energy attribution, and behavioral nudges**, rather than legally binding sub-metered billing.

---

## 2. Practical Utility Opportunity: Dynamic Off-Peak Load Shifting & Cold-Water Nudges

### 2.1 The Opportunity
Our exploratory data analysis across 5 households revealed two critical operational characteristics:
1. **Pronounced Morning Peaking:** Over **62% of all wash cycles** across households are initiated between **08:00 AM and 13:00 PM**, coinciding with the morning system peak and rapid solar ramp-up periods.
2. **Thermal Domination of Electrical Energy:** In every analyzed household, peak active power reached **2.0–2.3 kW** during the first 10–20 minutes of the cycle, driven entirely by internal resistance water heating elements. Water heating accounts for **~75–85% of total electrical energy per cycle** (0.33–0.60 kWh).

We formulate a dual-pronged utility intervention: **"SmartShift Laundry"**:
* **Intervention A (Time Shifting):** Incentivize consumers to defer morning laundry to the afternoon solar peak (13:00–16:00) or night-time off-peak trough (22:00–05:00) through a 15% Time-of-Day (ToD) tariff credit or automated smart-plug schedule.
* **Intervention B (Cold-Water Wash Nudge):** In-app behavioral nudges encouraging 30°C/eco-mode cycles over 60°C washes, directly cutting cycle energy consumption by ~40% (saving ~0.20 kWh per cycle across an average 6 cycles/week = ~62 kWh/year per household).

### 2.2 Measurement & Verification (M&V) Framework: How to Prove It Worked
To rigorously measure whether the program achieved real demand reduction rather than secular trends or weather effects, we implement a **Randomized Controlled Trial (RCT) with a Difference-in-Differences (DiD) Econometric Model**:

$$\Delta Y_{i,t} = \alpha + \beta_1 \cdot \text{Treated}_i + \beta_2 \cdot \text{Post}_t + \beta_{\text{DiD}} \cdot (\text{Treated}_i \times \text{Post}_t) + \mathbf{X}_{i,t}' \boldsymbol{\gamma} + \varepsilon_{i,t}$$

1. **Cohort Selection:** Identify 5,000 smart-metered households with detected laundry frequency $\ge 3$ cycles/week. Randomize into **Treatment Group** (receiving ToD incentives and behavioral app alerts) and **Control Group** (standard flat tariff).
2. **Metric 1 — Peak Coincidence Ratio:** Measure the percentage of detected wash cycles initiated during critical grid peak hours (e.g., 08:00–11:00 AM and 18:00–21:00 PM). Target: a 30% reduction in peak-hour cycle starts ($\beta_{\text{DiD}} < 0$).
3. **Metric 2 — Average Energy per Detected Cycle:** Track estimated kWh per wash. A statistically significant drop from ~0.50 kWh to ~0.35 kWh indicates consumer adoption of cold-water wash settings.
4. **Metric 3 — Feeder-Level Net Load Relief:** Verify aggregate substation feeder telemetry during morning peak hours to confirm that localized transformer load factor improved without rebound spikes.

---

## 3. Structural Limitations & Challenges of Transferring UK Insights to India

Applying machine learning models trained on UK datasets (such as REFIT) to Indian electrical utilities (e.g., BSES Delhi, Tata Power Mumbai, JVVNL Rajasthan) introduces profound domain-shift challenges across four key dimensions:

| Dimension | UK Household Reality (REFIT) | Indian Household Reality (DISCOMs) | Technical Impact on NILM |
|:---|:---|:---|:---|
| **Washing Machine Topologies** | Fully automatic front-loading washers with built-in 2.0–2.3 kW internal water-heating resistance elements. | Preponderance of top-loading semi-automatic or fully automatic washers **without internal heating elements** (draw 250–500 W pure motor load), or fed by external geysers. | The dominant 2 kW rectangular heating pulse disappears completely. The model must isolate low-power 300 W inductive motor pulses against large background noise. |
| **Dominant Overlapping Loads** | Dishwashers, electric kettles, and space heaters. | **Air Conditioners (1.5–2.5 kW inverters)**, dedicated instant/storage water heaters (Geysers, 2–3 kW), induction cooktops, and submersible borewell water pumps (1–2 HP). | High false-positive rates: Geysers and water pumps have electrical step-signatures nearly identical to UK washing machine heaters. |
| **Grid Power Quality & Voltage Stability** | Highly stable 230 V (±1%) grid with virtually zero voltage sag or unscheduled outages. | Significant voltage fluctuations (180 V to 260 V), harmonic distortion, and localized load-shedding/inverter battery transitions. | Active power drawn by resistive elements varies quadratically with voltage ($P = V^2 / R$). A 2 kW element drops to ~1.2 kW at 180 V, violating fixed threshold assumptions. |
| **Sub-Metered Baseline Availability** | Ground-truth IAM datasets (REFIT, UK-DALE) available for supervised training. | Extreme scarcity of high-resolution sub-metered ground truth data in Indian residential dwellings. | Supervised models cannot be trained directly; requires unsupervised domain adaptation, semi-supervised transfer learning, or synthetic load profile generation. |

---

## 4. Analytical Assessment: What Intensifies at 15-Minute or 30-Minute Resolutions?

Current widespread smart-meter rollouts under India's Revamped Distribution Sector Scheme (RDSS) and global Advanced Metering Infrastructure (AMI) record interval data at **15-minute or 30-minute intervals**, rather than 1-minute or 8-second resolution. Downgrading to this temporal frequency introduces several profound data-science bottlenecks:

1. **Sub-Interval Averaging & Signature Smearing:**
   A 2.3 kW water heating phase lasting 8 minutes in a 40-minute wash cycle represents a distinct rectangular step at 1-minute resolution. However, when integrated over a 30-minute sampling interval, that 8-minute pulse is averaged over the entire half hour:
   $$\bar{P}_{30\text{m}} = \frac{2300\text{ W} \times 8\text{ min} + 150\text{ W} \times 22\text{ min}}{30\text{ min}} \approx 723\text{ W}$$
   The sharp 2.3 kW transient is completely flattened into an ambiguous ~700 W bump, indistinguishable from a refrigerator defrost cycle, television entertainment center, or slow cooker.

2. **Violation of the Nyquist-Shannon Sampling Principle for Appliance States:**
   A washing machine transitions through distinct operational states (fill $\rightarrow$ heat $\rightarrow$ agitate $\rightarrow$ soak $\rightarrow$ rinse $\rightarrow$ spin) every 2 to 7 minutes. At 15- or 30-minute sampling, the sampling rate is substantially below the Nyquist rate of the underlying appliance state machine. All state transitions occur within a single sample point, stripping away the multi-scale temporal context features (`agg_roll_mean_15`, `agg_diff_5`) that our feature importance analysis proved were the most decisive predictors.

3. **Transformation from Event Detection to Ill-Posed Energy Decomposition:**
   At 1-minute resolution, NILM operates as an **event-detection and pattern-matching problem** (high signal-to-noise ratio). At 15/30-minute resolution, it degrades into an **under-determined blind source separation problem** where multiple combinations of appliances can produce the exact same half-hourly average kWh. 

4. **Strategic Shift in Modeling Paradigm:**
   At 15- or 30-minute resolution, point-wise regression and sequence-to-point neural networks fail. Utilities must shift toward:
   * **Hidden Markov Models (HMMs) and Factorial HMMs (FHMM)** that enforce probabilistic state transitions.
   * **Macroscopic Energy Attribution:** Transitioning from predicting instantaneous kW traces to predicting **daily or weekly appliance energy budget shares** via non-negative matrix factorization (NMF) or aggregate load decomposition.

