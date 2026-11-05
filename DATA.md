# DATA.md — REFIT Dataset: Sources, Coverage & Quality Notes

## 1  Data Source

| Item | Detail |
|------|--------|
| **Dataset** | REFIT Electrical Load Measurements (Murray et al., 2015) |
| **URL** | <https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned> |
| **Licence** | CC-BY 4.0 |
| **Citation** | Murray, Liao, Stankovic et al., "A data management platform for personalised real-time energy feedback", EEDAL 2015 |

The REFIT dataset contains sub-metered electricity readings from **20 UK households** (House 1 – House 21, skipping House 14) in the Loughborough area, collected between **October 2013 and July 2015**.

## 2  What Is Measured

| Aspect | Value |
|--------|-------|
| **Quantity** | Active power |
| **Unit** | Watts (W) |
| **Sensors per house** | 1 aggregate (current clamp on mains) + 9 Individual Appliance Monitors (IAMs) |
| **Raw polling interval** | ~8 seconds (6-8 s; sensors not synchronised, polled sequentially) |
| **Cleaned dataset resolution** | Same 8-second base rate (no resampling applied by REFIT authors) |

## 3  Raw vs Cleaned Dataset

| Property | RAW | CLEANED |
|----------|-----|---------|
| Daylight-saving correction | ✗ (UTC timestamp, but label says local) | ✓ Corrected to UK local time |
| IAM column alignment | May be swapped after sensor resets | ✓ Columns re-aligned to correct appliance |
| NaN handling | Part1 → 0; Part2 → NaN | Forward-filled |
| Spike removal | None | IAM values > 4,000 W replaced with 0 |
| Issues flag | Not present | `Issues` column = 1 when ΣIAMs > Aggregate |
| File naming | `RAW_House{id}_Part{1,2}.csv` | `CLEAN_House{id}.csv` |

### Part1 vs Part2 (Raw)
- **Part1**: Sensors not yet recording → value set to **0** (indistinguishable from a genuine 0 W reading).
- **Part2**: Same situation → value set to **NaN**.

## 4  House 1 — Raw Data Profile

| Metric | Part 1 | Part 2 |
|--------|--------|--------|
| Rows | 4,377,877 | 4,140,886 |
| Date range | 2013-10-09 → 2014-10-01 | 2014-06-27 → 2015-07-10 |
| Overlap window | 2014-06-27 → 2014-10-01 (~3 months) |

**Combined (deduplicated):** ~6.96 M rows, Oct 2013 → Jul 2015.

### House 1 Appliance Mapping

| Column | Appliance | Make / Model |
|--------|-----------|-------------|
| Aggregate | Whole-house | Current clamp |
| Appliance1 | Fridge | Hotpoint RLA50P |
| Appliance2 | Freezer 1 | Beko CF393APW |
| Appliance3 | Freezer 2 | Unknown |
| Appliance4 | Washer Dryer | Creda T522VW |
| Appliance5 | **Washing Machine** | **Beko WMC6140** |
| Appliance6 | Dishwasher | Bosch (unknown model) |
| Appliance7 | Computer | Lenovo H520s |
| Appliance8 | Television Site | Toshiba 32BL502b |
| Appliance9 | Electric Heater | GLEN 2172 |

## 5  Washing-Machine Channels Across All Houses

The following 19 houses have a dedicated washing-machine IAM channel:

| House | Column | Make / Model |
|-------|--------|-------------|
| 1 | Appliance5 | Beko WMC6140 |
| 2 | Appliance2 | LG F1289TD |
| 3 | Appliance6 | Unknown |
| 4 | Appliance4 | Unknown |
| 5 | Appliance3 | Unknown |
| 6 | Appliance2 | Unknown |
| 7 | Appliance5 | Bosch (unknown) |
| 8 | Appliance4 | Unknown |
| 9 | Appliance3 | Bosch Classixx 6 1200 Express |
| 10 | Appliance5 | Beko WI1382 |
| 11 | Appliance3 | Unknown |
| 13 | Appliance3 | Unknown |
| 15 | Appliance3 | Beko WMB91242LB |
| 16 | Appliance5 | Bosch WAB24262GB/01 |
| 17 | Appliance4 | Bosch Exxcel 8 Vario Perfect |
| 18 | Appliance5 | Unknown |
| 19 | Appliance2 | Bosch WAE24060GB/03 |
| 20 | Appliance4 | Unknown |
| 21 | Appliance3 | Beko WMB81241LW |

**House 12** has no identified washing-machine channel.

## 6  Known Data Limitations

1. **Missing-data episodes**: February 2014 is a known gap across multiple houses. Causes include household internet outages, hardware failures, and network routing issues.
2. **Sensor non-synchronisation**: The 9 IAMs and aggregate clamp are polled sequentially over a 6-8 s window, so readings within the same row are not truly simultaneous.
3. **Unmonitored loads**: Only 9 of ~15-49 appliances per household are sub-metered. The "remainder" (Aggregate − ΣIAMs) can be large.
4. **Appliance swaps**: Occupants occasionally moved or replaced IAM-monitored appliances despite instructions not to. The cleaned dataset corrects known swaps; unknown swaps remain.
5. **Part1 zero-ambiguity**: In raw Part1 files, 0 W could mean the appliance is genuinely off *or* the sensor was unavailable.
6. **UK-specific context**: All households are in Loughborough, UK. Appliance types, voltage (230 V / 50 Hz), and usage patterns may differ significantly from other geographies (e.g., India).

## 7  Raw Data Quality Findings (House 1)

### 7.1 Scale
- **Part1**: 4,377,877 rows (Oct 2013 → Oct 2014)
- **Part2**: 4,140,886 rows (Jun 2014 → Jul 2015)
- **Combined**: 8,518,763 rows (with 3-month overlap Jun-Oct 2014)

### 7.2 Duplicate Timestamps
- **1,412,513 rows** share a timestamp with at least one other row (650,962 unique timestamps).
- Duplicates arise from the Part1/Part2 overlap and from sensors being polled at the same second.

### 7.3 Sampling Intervals
| Bucket | Count | % |
|--------|------:|--:|
| < 1 s | 980,086 | 11.5% |
| 1-5 s | 1,670,154 | 19.6% |
| **5-8 s** | **3,905,017** | **45.8%** |
| 8-10 s | 160,980 | 1.9% |
| 10-15 s | 1,026,975 | 12.1% |
| 15-30 s | 9,166 | 0.1% |
| > 30 s | 4,833 | 0.06% |

Median interval = **6 s**, confirming the 6-8 s polling specification.

### 7.4 Major Data Gaps (>1 hour)
| Gap | Duration |
|-----|----------|
| 2014-01-24 → 2014-03-06 | **998 hours (41.6 days)** — the known Feb 2014 outage |
| 2015-05-10 → 2015-05-19 | 207 hours (8.6 days) |
| 2015-06-10 → 2015-06-18 | 180 hours (7.5 days) |
| 2015-04-03 → 2015-04-07 | 105 hours (4.4 days) |
| + 17 more | 2-55 hours each |

### 7.5 Missing (NaN) Values
- **Aggregate**: 2.2% NaN
- **Appliance5 (Washing Machine)**: 5.8% NaN
- **Appliance9 (Electric Heater)**: 22.4% NaN — highest among all channels

### 7.6 Extreme Values
- No negative readings found.
- **21,097 aggregate readings > 10 kW** (max 29,159 W) — likely sensor spikes.
- IAM spikes reach up to **98,301 W** — clearly spurious.

### 7.7 Zero-Reading Prevalence
- Appliances are off most of the time: **Appliance5 (WM) = 92% zeros**, Appliance6 (Dishwasher) = 95%.
- Part1 zeros are ambiguous (could be sensor-unavailable rather than genuinely off).

## 8  Cleaning Pipeline Design

### Stage 1 — Load & Concatenate
Combine Part1 (0-for-missing) and Part2 (NaN-for-missing) into a single frame.

### Stage 2 — Sort
Ensure strict chronological order.

### Stage 3 — Deduplicate
**Rule**: When the same timestamp exists in both Part1 and Part2, keep the Part2 row (NaN semantics are more trustworthy than ambiguous zeros). Otherwise keep first occurrence.

### Stage 4 — Clip Impossible Values
- Negative → 0 (sensor noise)
- Above 10,000 W → NaN (spurious spike, removed rather than forward-filled)

### Stage 5 — Gap-Aware Missing-Value Handling
- **Short gaps (≤ 5 min / ~37 readings)**: linear interpolation (plausible bridge).
- **Long gaps (> 5 min)**: left as NaN — these represent real outages and fabricating data would be dishonest.

### Stage 6 — Resample to 1-Minute
Mean aggregation of ~7-8 raw readings per minute. Mean power over 1 minute is proportional to energy consumed in that minute.

### Stage 7 — Outage Flagging
Any 1-minute bin with no valid readings gets `Outage = 1`.

## 9  Cleaning Pipeline Execution Results & Impact (House 1)

The pipeline was executed end-to-end on House 1 with the following quantitative outcomes:

| Metric | Raw Combined | After Cleaning & Resampling | Notes / Rationale |
|--------|--------------|-----------------------------|-------------------|
| **Total Rows** | 8,518,763 | 920,031 (1-min bins) | ~9.25 raw readings condensed per 1-minute bin |
| **Duplicates Removed** | 761,551 | 0 | Prioritized Part2 rows (retaining NaN distinction) |
| **Spikes Removed (>10 kW)** | 19,819 across channels | 0 remaining | 19,161 aggregate spikes + 658 IAM spikes removed |
| **Short Gaps Interpolated** | 390 gaps (<= 5 min) | Bridged | Linear interpolation applied up to max 37 readings |
| **Long Outages (>1 hr)** | 21 distinct periods | Flagged (`Outage = 1`) | Gaps left as NaN / flagged to avoid data fabrication |
| **Total Outage Minutes** | N/A | 115,353 minutes (12.54%) | 41.6-day outage in Feb 2014 accounts for majority |
| **Cleaned Time Range** | 2013-10-09 14:06 to 2015-07-10 11:56 | 2013-10-09 14:06:00 to 2015-07-10 11:56:00 | Continuous 1-minute datetime index |

### 9.1 Visual Verification
The cleaning results are visually confirmed across three saved figures:
1. `outputs/plots/house1_raw_vs_clean_48h.png`: Direct side-by-side comparison of 48 hours of continuous data (April 10-12, 2014) across Aggregate, Washing Machine (Appliance 5), and Dishwasher (Appliance 6). Shows how 1-minute mean resampling preserves distinct motor-spin and heating-element power profiles while eliminating high-frequency sensor noise.
2. `outputs/plots/house1_raw_vs_clean_full.png`: Longitudinal multi-year overview displaying whole-home aggregate, washing machine, and dishwasher consumption alongside the daily outage percentage trace.
3. `outputs/plots/house1_data_quality.png`: Diagnostic distribution plots illustrating the log-scale polling intervals (confirming the 6–8s peak) and active-power boxplots per sub-metered channel.

## 10  Remaining Quality Concerns & Recommendations

1. **Sub-metering Under-coverage:** In House 1, the 9 IAMs capture only a subset of total electrical appliances. Large aggregate spikes occur when unmonitored loads (e.g., electric showers, kettle, space heaters) are switched on.
2. **Ambiguity of Zeros in Part 1:** In Part 1, unplugged or non-reporting sensors recorded 0 W rather than NaN. While deduplication prioritized Part 2 during the overlap, early Part 1 non-zero inactive periods may blend standby consumption with zero readings.
3. **Sensor Non-Synchronicity:** In raw data, IAM readings occur 0.5–2 seconds apart rather than strictly simultaneously. The 1-minute mean resampling effectively eliminates this phase discrepancy for NILM applications.
4. **Outage Continuity:** Long outages (such as the 41.6-day outage in early 2014) must be masked out during sequence modeling and model evaluation so that synthetic zero-padding does not distort model loss or metrics.


