import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def audit_raw_data(df, name="House 1 Raw"):
    audit = {}
    audit["total_rows"] = len(df)
    
    # 1. Timestamp parse errors
    ts = pd.to_datetime(df["Time"], errors="coerce")
    audit["parse_errors"] = int(ts.isna().sum())
    
    # 2. Out of order timestamps
    time_diffs = ts.diff().dt.total_seconds()
    audit["out_of_order"] = int((time_diffs < 0).sum())
    
    # 3. Duplicate timestamps
    audit["duplicate_timestamps"] = int(df["Time"].duplicated().sum())
    audit["duplicate_unix"] = int(df["Unix"].duplicated().sum())
    
    # 4. Sampling intervals
    valid_diffs = time_diffs[time_diffs > 0]
    audit["sampling_median_sec"] = float(valid_diffs.median()) if len(valid_diffs) > 0 else 0.0
    audit["sampling_mean_sec"] = float(valid_diffs.mean()) if len(valid_diffs) > 0 else 0.0
    audit["sampling_min_sec"] = float(valid_diffs.min()) if len(valid_diffs) > 0 else 0.0
    audit["sampling_max_sec"] = float(valid_diffs.max()) if len(valid_diffs) > 0 else 0.0
    audit["intervals_le_0s"] = int((time_diffs <= 0).sum())
    audit["intervals_gt_60s"] = int((time_diffs > 60).sum())
    
    # 5. Negative values
    power_cols = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    audit["negative_counts"] = {col: int((df[col] < 0).sum()) for col in power_cols if col in df.columns}
    
    # 6. Impossible spikes
    audit["agg_spikes_gt_20kw"] = int((df["Aggregate"] > 20000).sum()) if "Aggregate" in df.columns else 0
    audit["agg_max_w"] = float(df["Aggregate"].max()) if "Aggregate" in df.columns else 0.0
    
    app_cols = [f"Appliance{i}" for i in range(1, 10)]
    audit["app_spikes_gt_4kw"] = {
        col: int((df[col] > 4000).sum()) for col in app_cols if col in df.columns
    }
    audit["app_max_w"] = {
        col: float(df[col].max()) for col in app_cols if col in df.columns
    }
    
    # 7. NaN counts
    audit["nan_counts"] = {col: int(df[col].isna().sum()) for col in power_cols if col in df.columns}
    
    return audit

def print_audit_report(audit):
    print("Dataset Audit Summary:")
    print(f"  Total records: {audit['total_rows']:,}")
    print(f"  Timestamp parsing errors: {audit['parse_errors']}")
    print(f"  Out-of-order timestamps: {audit['out_of_order']}")
    print(f"  Duplicate timestamps (Time): {audit['duplicate_timestamps']:,}")
    print(f"  Duplicate Unix timestamps: {audit['duplicate_unix']:,}")
    print(f"  Nominal sampling interval (median): {audit['sampling_median_sec']:.1f}s (mean: {audit['sampling_mean_sec']:.2f}s)")
    print(f"  Sampling gaps: {audit['intervals_gt_60s']:,} intervals > 60s, max gap: {audit['sampling_max_sec']:.1f}s")
    print(f"  Aggregate spikes (>20 kW): {audit['agg_spikes_gt_20kw']:,} (peak: {audit['agg_max_w']:,.0f} W)")
    total_app_spikes = sum(audit["app_spikes_gt_4kw"].values())
    print(f"  Total appliance spikes (>4 kW): {total_app_spikes:,}")
    for col, count in audit["app_spikes_gt_4kw"].items():
        if count > 0:
            print(f"    - {col}: {count:,} spikes (peak: {audit['app_max_w'][col]:,.0f} W)")

def clean_and_resample(df):
    power_cols = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    available_power = [c for c in power_cols if c in df.columns]
    
    # 1. Parse timestamps and sort chronologically
    df_clean = df.copy()
    df_clean["DateTime"] = pd.to_datetime(df_clean["Time"])
    df_clean = df_clean.sort_values("DateTime")
    
    # 2. Zero negative noise readings
    for col in available_power:
        df_clean.loc[df_clean[col] < 0, col] = 0.0
        
    # 3. Clean impossible spikes:
    # 20 kW single-phase limit for household aggregate (100A * 230V)
    if "Aggregate" in df_clean.columns:
        df_clean.loc[df_clean["Aggregate"] > 20000, "Aggregate"] = np.nan
        
    # 4 kW limit for 13A UK domestic appliances (BS 1363 limit ~3.1 kW continuous)
    for col in [c for c in available_power if c != "Aggregate"]:
        df_clean.loc[df_clean[col] > 4000, col] = 0.0
        
    # 4. Deduplicate timestamps by averaging simultaneous readings
    df_clean = df_clean.groupby("DateTime")[available_power].mean()
    
    # 5. Resample to strict 1-minute grid using interval mean
    df_1min = df_clean.resample("1min").mean()
    
    # 6. Handle missing intervals:
    # Classify gaps: <= 5 minutes are communication dropouts; > 5 minutes are outages.
    is_missing = df_1min["Aggregate"].isna()
    gap_id = (~is_missing).cumsum()
    gap_sizes = is_missing.groupby(gap_id).transform("sum")
    
    is_outage = is_missing & (gap_sizes > 5)
    is_dropout = is_missing & (gap_sizes <= 5)
    
    # Interpolate short dropouts linearly across all power channels
    df_filled = df_1min.interpolate(method="linear", limit=5, limit_direction="forward")
    
    # Restore NaNs during true long outages so we do not fabricate synthetic power
    for col in available_power:
        df_filled.loc[is_outage, col] = np.nan
        
    df_filled["is_outage"] = is_outage
    df_filled["is_dropout"] = is_dropout
    
    return df_filled, is_missing, is_outage, is_dropout

def plot_before_after(df_raw, df_1min, output_path="figures/sec1_before_after_cleaning.png"):
    start_time = pd.Timestamp("2013-10-22 22:00:00")
    end_time = pd.Timestamp("2013-10-23 09:00:00")
    
    # Raw slice
    df_raw_slice = df_raw.copy()
    if "DateTime" not in df_raw_slice.columns:
        df_raw_slice["DateTime"] = pd.to_datetime(df_raw_slice["Time"])
    raw_window = df_raw_slice[(df_raw_slice["DateTime"] >= start_time) & (df_raw_slice["DateTime"] <= end_time)]
    
    # Cleaned slice
    clean_window = df_1min.loc[start_time:end_time]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Panel 1: Raw telemetry showing erroneous ADC spikes
    ax1.plot(raw_window["DateTime"], raw_window["Aggregate"], label="Aggregate (Raw)", color="#2c3e50", alpha=0.8, linewidth=1.0)
    ax1.plot(raw_window["DateTime"], raw_window["Appliance1"], label="Appliance 1: Fridge (Raw)", color="#e74c3c", alpha=0.9, linewidth=1.2)
    ax1.plot(raw_window["DateTime"], raw_window["Appliance5"], label="Appliance 5: Washing Machine (Raw)", color="#3498db", alpha=0.8, linewidth=1.0)
    ax1.set_title("Before Cleaning: Raw 8-Second Telemetry with Sensor Bit-Errors (>30 kW Spike on Fridge)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Power (W)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(-500, 35000)
    ax1.annotate("Sensor Register Overflow\n(32,767 W ADC Spike)", 
                 xy=(pd.Timestamp("2013-10-22 23:26:16"), 32767), 
                 xytext=(pd.Timestamp("2013-10-23 01:00:00"), 28000),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight="semibold")
    
    # Panel 2: Cleaned 1-minute grid with true operational signatures
    ax2.plot(clean_window.index, clean_window["Aggregate"], label="Aggregate (1-min Mean)", color="#2c3e50", linewidth=1.2)
    ax2.plot(clean_window.index, clean_window["Appliance1"], label="Appliance 1: Fridge (Cleaned)", color="#e74c3c", linewidth=1.0)
    ax2.plot(clean_window.index, clean_window["Appliance5"], label="Appliance 5: Washing Machine (Cleaned)", color="#2980b9", linewidth=1.2)
    ax2.set_title("After Cleaning: 1-Minute Resampled Data with Outlier Removal and Despiking", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Power (W)", fontsize=11)
    ax2.set_xlabel("Timestamp", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(-100, 4500)
    
    # Highlight the washing machine cycle
    ax2.annotate("Active Washing Machine Cycle\n(~2.4 kW Water Heating Element)", 
                 xy=(pd.Timestamp("2013-10-23 07:05:00"), 2300), 
                 xytext=(pd.Timestamp("2013-10-23 03:00:00"), 3200),
                 arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight="semibold", color="#2980b9")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean raw House 1 smart-meter telemetry")
    parser.add_argument("--raw_part1", default="data/raw/RAW_House1_Part1.csv", help="Path to Part 1 raw CSV")
    parser.add_argument("--raw_part2", default="data/raw/RAW_House1_Part2.csv", help="Path to Part 2 raw CSV")
    parser.add_argument("--output_plot", default="figures/sec1_before_after_cleaning.png", help="Path to before/after figure")
    args = parser.parse_args()
    
    print(f"Loading raw data: {args.raw_part1}")
    df_raw = pd.read_csv(args.raw_part1)
    
    audit = audit_raw_data(df_raw, name="RAW House 1 Part 1")
    print_audit_report(audit)
    
    print("\nExecuting 1-minute cleaning and resampling pipeline...")
    df_1min, is_missing, is_outage, is_dropout = clean_and_resample(df_raw)
    
    total_grid_mins = len(df_1min)
    missing_count = int(is_missing.sum())
    dropout_count = int(is_dropout.sum())
    outage_count = int(is_outage.sum())
    
    print(f"1-Minute Resampling Statistics:")
    print(f"  Grid duration: {total_grid_mins:,} minutes ({total_grid_mins / 1440:.1f} days)")
    print(f"  Valid recorded intervals: {total_grid_mins - missing_count:,} ({(1 - missing_count/total_grid_mins)*100:.2f}%)")
    print(f"  Short dropouts (<=5 min, interpolated): {dropout_count:,} ({dropout_count/total_grid_mins*100:.2f}%)")
    print(f"  Long outages (>5 min, flagged): {outage_count:,} ({outage_count/total_grid_mins*100:.2f}%)")
    
    print(f"\nGenerating before/after cleaning visualization...")
    plot_before_after(df_raw, df_1min, output_path=args.output_plot)
    print("Section 1 execution completed successfully.")

if __name__ == "__main__":
    main()
