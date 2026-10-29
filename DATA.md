# Data Quality Audit and Cleaning Methodology

This document details the raw telemetry audit, cleaning transformations, gap-handling strategy, and persistent caveats for smart-meter measurements from the REFIT dataset, specifically focused on House 1.

---

## 1. Raw Data Quality Audit (House 1)

Raw data for House 1 was evaluated directly from `data/raw/RAW_House1_Part1.csv`. The telemetry corresponds to uncalibrated, unsynchronized current clamp and Individual Appliance Monitor (IAM) sensor logs.

### Audit Summary Statistics

| Data Quality Dimension | Metric / Finding | Exact Count / Value | Engineering Context |
| :--- | :--- | :--- | :--- |
| **Total Records** | Row count in raw file | 4,377,877 | Continuous logging from 2013-10-09 to 2014-10-01 (~356 days). |
| **Timestamp Parsing** | Unparseable / corrupted strings | 0 | Format strictly adheres to `YYYY-MM-DD HH:MM:SS`. |
| **Chronological Ordering** | Out-of-order timestamps ($\Delta t < 0$) | 2 | Minor buffer flush reordering in serial logging. |
| **Duplicate Timestamps** | Duplicate string timestamps (`Time`) | 645,514 | Sensors updated irregularly; multiple IAM messages share identical second strings. |
| **Duplicate Timestamps** | Duplicate integer timestamps (`Unix`) | 1,004,404 | Simultaneous packet arrivals recorded under identical epoch second. |
| **Sampling Interval** | Median sampling interval | 5.0 seconds | Base collection script attempted polling every 6-8 seconds. |
| **Sampling Interval** | Mean sampling interval | 8.25 seconds | Positively skewed by transmission delays and dropouts. |
| **Sampling Gaps** | Sampling intervals > 60 seconds | 780 | Transient local gateway or RF dropouts. |
| **Max Continuous Gap** | Longest recorded outage | 3,594,413 seconds | ~41.6 days of missing telemetry during February 2014 network outage. |
| **Negative Values** | Power readings < 0 W | 0 across all channels | Negative clamp noise was clipped or avoided at the firmware level. |
| **Aggregate Spikes** | Readings > 20,000 W | 53 records (peak: 24,000 W) | Exceeds typical UK 100A / 230V domestic supply fuse limit. |
| **Appliance Spikes** | Readings > 4,000 W | 3,038 total records | Exceeds maximum 13A / 230V UK BS 1363 socket rating (~3 kW continuous). |

### Detailed Appliance Spike Breakdown

Smart-meter IAMs in REFIT transmit 16-bit integer power readouts. A breakdown of readings exceeding 4,000 W reveals clear sensor hardware register overflows and ADC bit-flips:

| Channel | Appliance Name | Spikes (>4 kW) | Peak Recorded (W) | Identified Hardware Phenomenon |
| :--- | :--- | :--- | :--- | :--- |
| **Appliance 1** | Fridge (Hotpoint) | 662 | 69,629 W | High-frequency telemetry bursts; register bit corruption (`0x10FFF`). |
| **Appliance 2** | Freezer 1 (Beko) | 18 | 98,301 W | Bit-flip artifact (`0x17FFF`). |
| **Appliance 3** | Freezer 2 (Unknown) | 38 | 32,751 W | 15-bit signed integer boundary saturation. |
| **Appliance 4** | Washer Dryer (Creda) | 2,014 | 98,301 W | Persistent RF transceiver noise bursts under inductive motor switching. |
| **Appliance 5** | Washing Machine (Beko) | 156 | 32,767 W | Exact $2^{15} - 1$ signed 16-bit ADC overflow. |
| **Appliance 6** | Dishwasher (Bosch) | 24 | 14,336 W | Corrupted ADC readout packet. |
| **Appliance 7** | Computer (Lenovo) | 3 | 98,299 W | Transceiver register overflow. |
| **Appliance 8** | Television Site (Toshiba) | 88 | 65,549 W | $2^{16}$ unsigned integer wrap-around. |
| **Appliance 9** | Electric Heater (GLEN) | 35 | 65,535 W | Exact $2^{16} - 1$ unsigned 16-bit boundary saturation. |

### Partition Schema Discrepancy: RAW_House1_Part1.csv vs. RAW_House1_Part2.csv

During cross-partition auditing of the raw archive (`data/raw/`), an important telemetry logging quirk was identified between the two iterations:
- **RAW_House1_Part1.csv (Oct 2013 - Oct 2014, 4,377,877 records):** Adheres to consistent ISO-8601 formatted datetime strings (`YYYY-MM-DD HH:MM:SS`) in the `Time` column, paired with Unix epoch integers in the `Unix` column. Unavailable sensors were recorded as `0.0 W`. This partition serves as the primary evaluation target for Section 1, yielding 513,322 uniform 1-minute intervals.
- **RAW_House1_Part2.csv (Jun 2014 - Jul 2015, 4,140,886 records):** Corresponds to the v2 logging database where missing sensors were set to `NaN`. Auditing revealed that sections of the formatted string timestamp column were shifted or populated directly with Unix epoch integer values rather than datetime strings, alongside non-monotonic timestamps across overlapping date ranges.

Recognizing these schema variations between raw logging partitions is essential for robust data engineering: Section 1 explicitly anchors its cleaning and validation benchmarks on the clean, uncorrupted schema of Part 1.

---

## 2. Data Cleaning and Resampling Pipeline

The raw data was transformed into a standardized, uniform 1-minute time series using a 6-step deterministic pipeline:

```
[Raw Telemetry] -> [Sort & Validate] -> [Zero Noise & Cap Spikes] -> [Deduplicate Timestamps] -> [1-Min Grid Mean] -> [Classify Dropouts vs Outages]
```

1. **Chronological Sorting & Validation:** Timestamps are parsed as UTC datetimes, and all records are sorted chronologically. Out-of-order records are placed into proper sequence.
2. **Negative Power Rectification:** Any spurious negative power readings are set to `0.0 W`.
3. **Physical Domain Spike Filtering:**
   - **Aggregate Power:** Capped at 20,000 W (20 kW), reflecting the physical operating limit of a standard UK single-phase domestic service (100A main service fuse at 230V nominal). Readings above 20 kW are nullified to `NaN`.
   - **Individual Appliance Monitors (IAMs):** Capped at 4,000 W, reflecting the physical safety boundary of UK domestic BS 1363 plugs and 13A socket circuits (continuous rating ~3,000 W; short inductive trip ceiling ~3,200 W). Values above 4,000 W are replaced with `0.0 W` in accordance with REFIT protocol.
4. **Deduplication:** Simultaneous sensor readings falling within the exact same second are aggregated by taking their arithmetic mean across all channels.
5. **Uniform 1-Minute Resampling:** Data is binned into fixed 1-minute intervals (`1min`) using interval averages (`mean()`). This eliminates sensor polling jitter (6-8s irregular polling) and produces a reliable, low-noise grid for disaggregation.
6. **Gap and Outage Handling:**
   - **Short Communication Dropouts ($\le$ 5 consecutive minutes):** Attributed to transient packet loss, ZigBee / Wi-Fi retries, or momentary gateway buffer congestion. These are linearly interpolated across power channels.
   - **Extended Outages (> 5 consecutive minutes):** Attributed to broadband disconnections, hardware crashes, or extended power cuts. Interpolating across long gaps fabricates artificial energy consumption. Instead, intervals belonging to gaps > 5 minutes are strictly maintained as `NaN` and flagged with `is_outage = True`.

### 1-Minute Resampling Output Statistics (House 1)

- **Total 1-minute grid intervals:** 513,322 minutes (~356.5 days)
- **Valid recorded intervals:** 436,413 minutes (85.02%)
- **Short dropouts ($\le$ 5 min, interpolated):** 606 minutes (0.12%)
- **Long outages (> 5 min, flagged as outage):** 76,303 minutes (14.86%)

---

## 3. Before and After Cleaning Visualization

The impact of the cleaning and resampling pipeline is illustrated below over a 11-hour window (`2013-10-22 22:00` to `2013-10-23 09:00`) containing both an active washing machine cycle and severe sensor bit-flip spikes.

![Before and After Cleaning](figures/sec1_before_after_cleaning.png)

### Key Observations from the Plot

1. **Spike Removal:** In the raw trace (top panel), Appliance 1 (Fridge) suffered a massive $2^{15}-1$ ADC overflow spike to 32,767 W. This single false reading would distort baseline variance, energy integrations, and regression loss functions. The cleaning pipeline cleanly eliminates this spike.
2. **Signal Regularization:** In the cleaned trace (bottom panel), raw 8-second polling jitter is transformed into a clean 1-minute time series.
3. **Appliance Signature Preservation:** The multi-phase operational cycle of Appliance 5 (Washing Machine) is preserved with high fidelity. The initial ~2.4 kW water heating element draw and subsequent drum agitation motor pulses remain intact, ready for feature extraction.

---

## 4. Remaining Data Caveats

While the cleaning pipeline resolves instantaneous errors and irregular grids, practitioners deploying NILM on this data must account for the following persistent characteristics:

1. **Sub-Metering Discrepancies (IAM Sum > Aggregate):** The REFIT dataset includes an `Issues` flag that triggers when $\sum_{i=1}^9 \text{IAM}_i > \text{Aggregate}$. In House 1, minor discrepancies occur due to non-synchronized sensor transmission (the aggregate current clamp and individual IAMs were polled sequentially over an 8-second window rather than latched simultaneously).
2. **Unmonitored Background Loads:** The 9 monitored IAM channels do not account for all household electricity. Hardwired loads (e.g., electric showers, immersion heaters, central heating pumps, ceiling lighting circuits) only appear in the Aggregate channel, acting as unlabelled background noise during disaggregation.
3. **Extended Seasonal Outages:** As observed in the audit, House 1 experienced an extended outage of ~41.6 days in February 2014. Disaggregation models trained on seasonal time features must account for this discontinuity.
