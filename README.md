# Smart-Meter Telemetry Cleaning and Appliance Disaggregation (NILM)

This repository provides an end-to-end data engineering, exploratory analysis, and Non-Intrusive Load Monitoring (NILM) disaggregation pipeline using high-resolution domestic electricity data from the UK REFIT dataset.

The project demonstrates:
1. Auditing and cleaning uncalibrated 8-second smart-meter telemetry into a standardized 1-minute uniform time series with physically grounded outage handling.
2. Characterizing washing-machine behavioral usage across multiple households using domain-specific cycle detection heuristics.
3. Training and evaluating machine-learning disaggregation models to isolate washing-machine power strictly from whole-home aggregate telemetry without target leakage.

---

## 1. Repository Structure

```
.
|-- data/
|   |-- CLEAN_READ_ME_081116.txt        # Official REFIT channel mapping and metadata
|   |-- raw/                            # Uncalibrated 8-second telemetry (Part 1 & Part 2)
|   `-- clean/                          # Cleaned reference datasets (Houses 1-7)
|-- figures/
|   |-- sec1_before_after_cleaning.png  # Sensor bit-flip removal & 1-minute resampling
|   |-- sec2_24hr_trace.png             # 24-hour trace: Aggregate vs Washing Machine
|   |-- sec2_detected_cycles.png        # Zoomed-in cycle with heating vs agitation phases
|   |-- sec2_cross_house_comparison.png # Diurnal, day-of-week, and duration distributions
|   `-- sec3_actual_vs_predicted.png    # Test set disaggregation & failure case analysis
|-- src/
|   |-- 01_clean_raw.py                 # Telemetry audit, spike removal, and resampling
|   |-- 02_eda_washing_machine.py       # Cross-household cycle detection and analysis
|   `-- 03_train_evaluate_nilm.py       # Feature extraction, baselines, LightGBM models
|-- run_pipeline.py                     # Master execution runner
|-- requirements.txt                    # Minimal environment dependencies
|-- DATA.md                             # Section 1 data audit and preprocessing report
|-- RESULTS.md                          # Sections 2 & 3 empirical results and model benchmarks
`-- Recommendation.md                   # Section 4 utility executive briefing
```

---

## 2. Environment Setup & Execution

### Requirements
The pipeline requires Python 3.10+ with standard scientific libraries:
```bash
pip install -r requirements.txt
```

Minimal dependencies:
- `pandas>=2.2.0`
- `numpy>=1.26.0`
- `matplotlib>=3.8.0`
- `seaborn>=0.13.0`
- `scikit-learn>=1.4.0`
- `lightgbm>=4.3.0`

### End-to-End Execution
To run all three stages sequentially and reproduce all figures and metrics:
```bash
python run_pipeline.py
```

### Modular Step-by-Step Execution
Individual components can also be run independently:

```bash
# 1. Audit raw House 1 telemetry and generate before/after cleaning plot
python src/01_clean_raw.py

# 2. Run multi-household washing machine exploratory analysis and cycle detection
python src/02_eda_washing_machine.py

# 3. Train and benchmark NILM disaggregation models against baselines
python src/03_train_evaluate_nilm.py
```

---

## 3. Engineering Decisions & Validation Architecture

### Preprocessing & Resampling
- **Physical Boundary Clipping:** Spikes above 20,000 W in aggregate demand (exceeding standard UK 100A / 230V service fuses) and above 4,000 W in appliance channels (exceeding 13A BS 1363 sockets) are identified as 16-bit ADC register overflows and eliminated.
- **Uniform 1-Minute Cadence:** Resolves 6-8s irregular sensor polling delays while maintaining sufficient temporal resolution to distinguish motor agitation bursts from resistive water-heating steps.
- **Outage Flagging vs. Forward-Filling:** Short gaps ($\le 5$ min) representing transient packet retries are interpolated; long gaps (>5 min) are strictly preserved as missing and tagged with an `is_outage` flag to prevent fabricating fictitious energy consumption.

### Leakage-Free Temporal Validation
Domestic appliance cycles exhibit strong autocorrelation over 30 to 90 minutes. Random K-fold splitting or overlapping sliding windows leak future cycle information into the training set. To ensure realistic generalization:
- **Strict Chronological Partition:** Training on the earliest 70% of the timeline, validation on the next 15%, and testing on the final 15% (unseen future calendar months).
- **Causal Feature Extraction:** All rolling statistical features (mean, std, min, max over 5, 15, and 30 minutes) use backward-looking windows to prevent lookahead bias.

### Model Performance Summary (House 1 Test Data)

| Model Architecture | MAE (W) | RMSE (W) | NDE | Total Energy Error (%) | F1-Score ($\ge 20\text{W}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Always-Off Baseline** | 11.04 W | 137.03 W | 1.0000 | 100.0% | 0.0000 |
| **Hourly Mean Profile Baseline** | 19.80 W | 136.04 W | 0.9857 | 12.28% | 0.1269 |
| **Direct LightGBM Regressor** | **11.14 W** | **98.21 W** | **0.5137** | **7.09%** | 0.4723 |
| **Two-Stage (Classifier + Regressor)** | 13.09 W | 136.80 W | 0.9966 | 44.37% | **0.6280** |

*Note: While Always-Off achieves a misleadingly low MAE of 11.04 W due to appliance sparsity (inactive 98.4% of the time), it detects zero cycles. The Direct LightGBM model achieves an NDE of 0.5137 (~49% error reduction) and a 7.09% Total Energy Error over 3.5 months of continuous test data.*

---

## 4. Key Trade-Offs & Future Improvements

1. **Trade-Off between Energy Tracking and State Precision:**
   - The *Direct Regressor* minimizes energy error (7.09% error) and captures peak heating pulses, but introduces false-positive tail noise when other high-power resistive loads are active.
   - The *Two-Stage Architecture* maximizes event state detection (F1 = 0.6280, Precision = 56.31%), but threshold gating clips low-power tumble agitation tails.
2. **Future Enhancements:**
   - **Sequence-to-Point / Temporal Convolutional Networks:** Exploring 1D Dilated CNNs or Sequence-to-Sequence (Seq2Seq) neural architectures to capture extended 60-minute cycle context without manual rolling feature engineering.
   - **Multi-Modal Feature Fusion:** Incorporating reactive power ($Q$), voltage, and apparent power if available to separate pure resistive heating from inductive motor rotations.
   - **Cross-Household Domain Adaptation:** Fine-tuning base models trained on high-occupancy homes to transfer to low-occupancy homes using self-supervised contrastive learning.

---

## 5. Note on AI Tools Used

In accordance with transparent engineering practice: an AI coding assistant was used to assist with syntax boilerplate, Matplotlib visualization styling, and Markdown table formatting. All domain logic, physical threshold definitions, cycle detection heuristics, feature engineering design, validation strategies, model benchmarking, and analytical interpretations were manually conceived and engineered.
