import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HOUSE_CONFIG = {
    "House 1": {"path": "data/clean/CLEAN_House1.csv", "wm_col": "Appliance5", "occupants": 2, "model": "Beko WMC6140"},
    "House 2": {"path": "data/clean/CLEAN_House2.csv", "wm_col": "Appliance2", "occupants": 4, "model": "LG F1289TD"},
    "House 3": {"path": "data/clean/CLEAN_House3.csv", "wm_col": "Appliance6", "occupants": 2, "model": "Standard UK Front-Loader"},
    "House 5": {"path": "data/clean/CLEAN_House5.csv", "wm_col": "Appliance3", "occupants": 4, "model": "AEG L99695HWD"}
}

def load_and_resample_household(csv_path, wm_col):
    df = pd.read_csv(csv_path, usecols=["Time", "Aggregate", wm_col])
    df["DateTime"] = pd.to_datetime(df["Time"])
    
    # Exclude invalid negative readings
    df.loc[df["Aggregate"] < 0, "Aggregate"] = np.nan
    df.loc[df[wm_col] < 0, wm_col] = 0.0
    
    # Resample to strict 1-minute intervals by taking mean of readings
    df_1min = df.set_index("DateTime")[["Aggregate", wm_col]].resample("1min").mean()
    return df_1min

def detect_cycles(series_wm, active_thresh=20.0, max_pause_min=15, min_duration_min=15, min_peak_w=150.0):
    """
    Deterministic rule-based washing machine cycle detection:
    1. Active threshold: Power >= 20W (filters quiescent standby/LED vampire draw).
    2. Pause bridging: Intersession pauses <= 15 min are bridged into a single continuous cycle
       (covers motor idle between wash tumbling, drain, and rinse fills).
    3. Minimum duration: Total cycle span >= 15 min (eliminates transient spikes/accidental button taps).
    4. Peak power threshold: Peak >= 150W (confirms motor agitation or heating activation).
    """
    # Create filled series for indexing, treating missing intervals as 0W inactive
    s = series_wm.fillna(0.0)
    active_indices = np.where(s.values >= active_thresh)[0]
    
    if len(active_indices) == 0:
        return []
    
    cycles = []
    start_idx = active_indices[0]
    end_idx = active_indices[0]
    
    for idx in active_indices[1:]:
        gap = idx - end_idx - 1
        if gap <= max_pause_min:
            end_idx = idx
        else:
            duration = end_idx - start_idx + 1
            cycle_vals = s.iloc[start_idx:end_idx + 1]
            peak = float(cycle_vals.max())
            if duration >= min_duration_min and peak >= min_peak_w:
                energy_kwh = float(cycle_vals.sum() / 60000.0)
                cycles.append({
                    "start_time": s.index[start_idx],
                    "end_time": s.index[end_idx],
                    "duration_min": duration,
                    "peak_w": peak,
                    "mean_w": float(cycle_vals.mean()),
                    "energy_kwh": energy_kwh,
                    "hour": s.index[start_idx].hour,
                    "day_of_week": s.index[start_idx].day_name()
                })
            start_idx = idx
            end_idx = idx
            
    # Process final trailing cycle
    duration = end_idx - start_idx + 1
    cycle_vals = s.iloc[start_idx:end_idx + 1]
    peak = float(cycle_vals.max())
    if duration >= min_duration_min and peak >= min_peak_w:
        energy_kwh = float(cycle_vals.sum() / 60000.0)
        cycles.append({
            "start_time": s.index[start_idx],
            "end_time": s.index[end_idx],
            "duration_min": duration,
            "peak_w": peak,
            "mean_w": float(cycle_vals.mean()),
            "energy_kwh": energy_kwh,
            "hour": s.index[start_idx].hour,
            "day_of_week": s.index[start_idx].day_name()
        })
        
    return cycles

def plot_24hr_trace(df_1min, wm_col, output_path="figures/sec2_24hr_trace.png"):
    # Target date with representative daytime laundry activity in House 1: 2013-10-23
    start_ts = pd.Timestamp("2013-10-23 00:00:00")
    end_ts = pd.Timestamp("2013-10-23 23:59:00")
    slice_df = df_1min.loc[start_ts:end_ts]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(slice_df.index, slice_df["Aggregate"], label="Household Aggregate", color="#7f8c8d", alpha=0.7, linewidth=1.2)
    ax.plot(slice_df.index, slice_df[wm_col], label="Washing Machine (Appliance 5)", color="#2980b9", linewidth=1.8)
    
    ax.set_title("Representative 24-Hour Power Trace (House 1 - 2013-10-23): Aggregate vs Washing Machine", fontsize=12, fontweight="bold")
    ax.set_ylabel("Active Power (W)", fontsize=11)
    ax.set_xlabel("Time of Day", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(-50, slice_df["Aggregate"].max() * 1.08)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_detected_cycles(df_1min, wm_col, cycles_df, output_path="figures/sec2_detected_cycles.png"):
    # Zoom in on detected morning wash cycle in House 1: 2013-10-23
    start_ts = pd.Timestamp("2013-10-23 05:45:00")
    end_ts = pd.Timestamp("2013-10-23 08:00:00")
    slice_df = df_1min.loc[start_ts:end_ts]
    
    # Filter cycles within this time frame
    sub_cycles = cycles_df[(cycles_df["start_time"] >= start_ts) & (cycles_df["end_time"] <= end_ts)]
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(slice_df.index, slice_df["Aggregate"], label="Aggregate Load", color="#95a5a6", alpha=0.5, linewidth=1.0)
    ax.plot(slice_df.index, slice_df[wm_col], label="Washing Machine Load", color="#2980b9", linewidth=2.0)
    
    # Mark cycle boundaries
    for idx, c in sub_cycles.iterrows():
        ax.axvspan(c["start_time"], c["end_time"], color="#3498db", alpha=0.15, label="Detected Wash Cycle" if idx == sub_cycles.index[0] else "")
        ax.axvline(c["start_time"], color="#27ae60", linestyle="--", linewidth=1.5, label="Cycle Start" if idx == sub_cycles.index[0] else "")
        ax.axvline(c["end_time"], color="#c0392b", linestyle="--", linewidth=1.5, label="Cycle End" if idx == sub_cycles.index[0] else "")
        
        # Annotate peak and energy
        mid_time = c["start_time"] + (c["end_time"] - c["start_time"]) / 2
        ax.text(mid_time, 2100, f"Duration: {c['duration_min']:.0f} min\nEnergy: {c['energy_kwh']:.2f} kWh\nPeak: {c['peak_w']:.0f} W",
                horizontalalignment="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2980b9", alpha=0.9))

    # Phase annotations on cycle
    ax.annotate("Resistive Water Heating\n(~2.3 kW Element)", 
                xy=(pd.Timestamp("2013-10-23 06:03:00"), 2320), 
                xytext=(pd.Timestamp("2013-10-23 05:48:00"), 2650),
                arrowprops=dict(facecolor="#c0392b", shrink=0.05, width=1, headwidth=5),
                fontsize=9, fontweight="semibold", color="#c0392b")
    
    ax.annotate("Tumble Agitation &\nInter-Phase Pauses (100-350 W)", 
                xy=(pd.Timestamp("2013-10-23 06:22:00"), 150), 
                xytext=(pd.Timestamp("2013-10-23 06:10:00"), 1200),
                arrowprops=dict(facecolor="#2980b9", shrink=0.05, width=1, headwidth=5),
                fontsize=9, fontweight="semibold", color="#2980b9")
    
    ax.annotate("Secondary Heating / Spin Phase\n(~2.3 kW / 800 W)", 
                xy=(pd.Timestamp("2013-10-23 06:43:00"), 2360), 
                xytext=(pd.Timestamp("2013-10-23 06:55:00"), 2700),
                arrowprops=dict(facecolor="#8e44ad", shrink=0.05, width=1, headwidth=5),
                fontsize=9, fontweight="semibold", color="#8e44ad")

    ax.set_title("Zoomed View of Detected Washing Machine Cycle: Multi-Phase Thermal & Mechanical Signatures", fontsize=12, fontweight="bold")
    ax.set_ylabel("Power (W)", fontsize=11)
    ax.set_xlabel("Time", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(-50, 3200)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def plot_cross_house_comparison(all_cycles_df, output_path="figures/sec2_cross_house_comparison.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = {"House 1": "#2980b9", "House 2": "#e67e22", "House 3": "#27ae60", "House 5": "#8e44ad"}
    
    # 1. Hourly start profile
    hourly_counts = all_cycles_df.groupby(["hour", "house"]).size().reset_index(name="cycle_count")
    for house in palette:
        sub = hourly_counts[hourly_counts["house"] == house]
        axes[0, 0].plot(sub["hour"], sub["cycle_count"], marker="o", label=house, color=palette[house], linewidth=1.8)
    axes[0, 0].set_title("Cycle Starts by Hour of Day", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Hour of Day (0 - 23)", fontsize=10)
    axes[0, 0].set_ylabel("Total Cycle Starts", fontsize=10)
    axes[0, 0].set_xticks(range(0, 24, 2))
    axes[0, 0].legend(loc="upper right", frameon=True)
    
    # 2. Day of week profile
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = all_cycles_df.groupby(["day_of_week", "house"]).size().reindex(
        pd.MultiIndex.from_product([days_order, palette.keys()], names=["day_of_week", "house"]), fill_value=0
    ).reset_index(name="cycle_count")
    
    sns.barplot(data=dow_counts, x="day_of_week", y="cycle_count", hue="house", palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title("Cycle Starts by Day of Week", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Day of Week", fontsize=10)
    axes[0, 1].set_ylabel("Total Cycle Starts", fontsize=10)
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend(loc="upper right", frameon=True)
    
    # 3. Cycle duration distribution
    sns.boxplot(data=all_cycles_df, x="house", y="duration_min", hue="house", palette=palette, ax=axes[1, 0], showfliers=False, legend=False)
    axes[1, 0].set_title("Cycle Duration Distribution (Excluding Outliers)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Household", fontsize=10)
    axes[1, 0].set_ylabel("Duration (Minutes)", fontsize=10)
    
    # 4. Energy consumption per cycle
    sns.boxplot(data=all_cycles_df, x="house", y="energy_kwh", hue="house", palette=palette, ax=axes[1, 1], showfliers=False, legend=False)
    axes[1, 1].set_title("Energy Consumption per Cycle (Excluding Outliers)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Household", fontsize=10)
    axes[1, 1].set_ylabel("Energy per Cycle (kWh)", fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    summary_records = []
    all_cycles = []
    house_1_data = None
    
    for house_name, cfg in HOUSE_CONFIG.items():
        print(f"Processing {house_name} ({cfg['model']})...")
        df_1min = load_and_resample_household(cfg["path"], cfg["wm_col"])
        
        if house_name == "House 1":
            house_1_data = (df_1min, cfg["wm_col"])
            
        start_date = df_1min.index.min().strftime("%Y-%m-%d")
        end_date = df_1min.index.max().strftime("%Y-%m-%d")
        total_mins = len(df_1min)
        missing_mins = int(df_1min[cfg["wm_col"]].isna().sum())
        pct_missing = (missing_mins / total_mins) * 100.0
        
        # Zero / standby (<20W)
        wm_series = df_1min[cfg["wm_col"]].fillna(0.0)
        standby_mins = int((wm_series < 20.0).sum())
        pct_standby = (standby_mins / total_mins) * 100.0
        
        cycles = detect_cycles(df_1min[cfg["wm_col"]])
        for c in cycles:
            c["house"] = house_name
            c["occupants"] = cfg["occupants"]
        all_cycles.extend(cycles)
        
        cdf = pd.DataFrame(cycles)
        summary_records.append({
            "House": house_name,
            "Occupancy": cfg["occupants"],
            "Appliance Model": cfg["model"],
            "Date Range": f"{start_date} to {end_date}",
            "Total 1-min Intervals": total_mins,
            "Missing Intervals (%)": pct_missing,
            "Quiescent / Standby (%)": pct_standby,
            "Detected Cycles": len(cdf),
            "Median Duration (min)": cdf["duration_min"].median() if len(cdf) > 0 else 0,
            "Mean Duration (min)": cdf["duration_min"].mean() if len(cdf) > 0 else 0,
            "Median Peak (W)": cdf["peak_w"].median() if len(cdf) > 0 else 0,
            "Median Energy (kWh)": cdf["energy_kwh"].median() if len(cdf) > 0 else 0,
            "Mean Energy (kWh)": cdf["energy_kwh"].mean() if len(cdf) > 0 else 0
        })

    summary_df = pd.DataFrame(summary_records)
    all_cycles_df = pd.DataFrame(all_cycles)
    
    print("\nSummary Table Across Monitored Homes:")
    for _, row in summary_df.iterrows():
        print(f"  {row['House']} ({row['Appliance Model']}, {row['Occupancy']} occupants):")
        print(f"    Timeline: {row['Date Range']} ({row['Total 1-min Intervals']:,} minutes)")
        print(f"    Missing: {row['Missing Intervals (%)']:.2f}%, Standby/Off: {row['Quiescent / Standby (%)']:.2f}%")
        print(f"    Cycles detected: {row['Detected Cycles']:,}")
        print(f"    Duration: median {row['Median Duration (min)']:.1f} min (mean {row['Mean Duration (min)']:.1f} min)")
        print(f"    Peak power: median {row['Median Peak (W)']:.0f} W")
        print(f"    Energy per cycle: median {row['Median Energy (kWh)']:.3f} kWh (mean {row['Mean Energy (kWh)']:.3f} kWh)")
        
    # Generate Section 2 visualizations
    if house_1_data is not None:
        h1_df, h1_col = house_1_data
        h1_cycles = all_cycles_df[all_cycles_df["house"] == "House 1"]
        print("\nGenerating Figure 2.1: 24-Hour Representative Trace...")
        plot_24hr_trace(h1_df, h1_col)
        print("Generating Figure 2.2: Zoomed Detected Wash Cycles...")
        plot_detected_cycles(h1_df, h1_col, h1_cycles)
        
    print("Generating Figure 2.3: Cross-Household Comparison...")
    plot_cross_house_comparison(all_cycles_df)
    
    print("Section 2 exploratory analysis completed successfully.")

if __name__ == "__main__":
    main()
