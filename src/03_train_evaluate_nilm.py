import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, f1_score, precision_score, recall_score

def build_features(df_1min):
    agg = df_1min["Aggregate"]
    feats = pd.DataFrame(index=df_1min.index)
    
    # 1. Instantaneous and delta features (causal edge detection)
    feats["agg"] = agg
    feats["delta_1"] = agg.diff().fillna(0.0)
    feats["delta_2"] = feats["delta_1"].diff().fillna(0.0)
    
    # 2. Backward rolling window features (causal: strictly past and current observations)
    for w in [5, 15, 30]:
        r = agg.rolling(w, min_periods=1)
        feats[f"mean_{w}"] = r.mean()
        feats[f"std_{w}"] = r.std().fillna(0.0)
        feats[f"min_{w}"] = r.min()
        feats[f"max_{w}"] = r.max()
        feats[f"range_{w}"] = feats[f"max_{w}"] - feats[f"min_{w}"]
        feats[f"diff_from_min_{w}"] = agg - feats[f"min_{w}"]
        
    # 3. Cyclical time features (capture human activity habits)
    feats["hour_sin"] = np.sin(2.0 * np.pi * feats.index.hour / 24.0)
    feats["hour_cos"] = np.cos(2.0 * np.pi * feats.index.hour / 24.0)
    feats["dow_sin"] = np.sin(2.0 * np.pi * feats.index.dayofweek / 7.0)
    feats["dow_cos"] = np.cos(2.0 * np.pi * feats.index.dayofweek / 7.0)
    feats["is_weekend"] = (feats.index.dayofweek >= 5).astype(float)
    
    return feats

def compute_metrics(y_true, y_pred, active_thresh=20.0):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    
    denom_nde = float(np.sum(y_true ** 2))
    nde = float(np.sum((y_pred - y_true) ** 2) / denom_nde) if denom_nde > 0 else 1.0
    
    sum_true = float(np.sum(y_true))
    tee = float(abs(np.sum(y_pred) - sum_true) / sum_true * 100.0) if sum_true > 0 else 0.0
    
    # State classification metrics (ON >= 20W)
    state_true = (y_true >= active_thresh).astype(int)
    state_pred = (y_pred >= active_thresh).astype(int)
    
    prec = float(precision_score(state_true, state_pred, zero_division=0))
    rec = float(recall_score(state_true, state_pred, zero_division=0))
    f1 = float(f1_score(state_true, state_pred, zero_division=0))
    
    return {
        "MAE (W)": mae,
        "RMSE (W)": rmse,
        "NDE": nde,
        "Total Energy Error (%)": tee,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

def train_and_evaluate(data_path="data/clean/CLEAN_House1.csv", output_plot="figures/sec3_actual_vs_predicted.png"):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, usecols=["Time", "Aggregate", "Appliance5", "Appliance6"])
    df["DateTime"] = pd.to_datetime(df["Time"])
    
    # 1-minute grid resampling
    df_1min = df.set_index("DateTime")[["Aggregate", "Appliance5", "Appliance6"]].resample("1min").mean()
    df_clean = df_1min.dropna(subset=["Aggregate"]).copy()
    
    print(f"Total valid 1-minute records: {len(df_clean):,}")
    print("Constructing causal rolling features...")
    feats = build_features(df_clean)
    
    y_power = df_clean["Appliance5"].fillna(0.0)
    y_state = (y_power >= 20.0).astype(int)
    
    # Strict chronological train / validation / test partition (70% / 15% / 15%)
    n = len(feats)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    
    X_train, y_train_state, y_train_pow = feats.iloc[:n_train], y_state.iloc[:n_train], y_power.iloc[:n_train]
    X_val, y_val_state, y_val_pow = feats.iloc[n_train:n_val], y_state.iloc[n_train:n_val], y_power.iloc[n_train:n_val]
    X_test, y_test_state, y_test_pow = feats.iloc[n_val:], y_state.iloc[n_val:], y_power.iloc[n_val:]
    
    test_slice = df_clean.iloc[n_val:].copy()
    
    print(f"Temporal Split:")
    print(f"  Train:      {X_train.index.min().strftime('%Y-%m-%d')} to {X_train.index.max().strftime('%Y-%m-%d')} ({len(X_train):,} intervals)")
    print(f"  Validation: {X_val.index.min().strftime('%Y-%m-%d')} to {X_val.index.max().strftime('%Y-%m-%d')} ({len(X_val):,} intervals)")
    print(f"  Test:       {X_test.index.min().strftime('%Y-%m-%d')} to {X_test.index.max().strftime('%Y-%m-%d')} ({len(X_test):,} intervals)")
    
    # Baseline 1: Always Off
    pred_off = np.zeros(len(y_test_pow))
    metrics_off = compute_metrics(y_test_pow.values, pred_off)
    
    # Baseline 2: Hourly Mean Usage Profile
    hourly_means = y_train_pow.groupby(y_train_pow.index.hour).mean()
    pred_hourly = X_test.index.hour.map(hourly_means).values
    metrics_hourly = compute_metrics(y_test_pow.values, pred_hourly)
    
    # Model 1: Direct LightGBM Regressor with post-thresholding
    print("\nTraining Model 1: Direct LightGBM Regressor...")
    reg_direct = lgb.LGBMRegressor(
        n_estimators=150, max_depth=7, learning_rate=0.06, random_state=42, verbose=-1, n_jobs=-1
    )
    reg_direct.fit(X_train, y_train_pow)
    raw_pred_test = reg_direct.predict(X_test)
    pred_direct = np.where(raw_pred_test < 15.0, 0.0, raw_pred_test)
    metrics_direct = compute_metrics(y_test_pow.values, pred_direct)
    
    # Model 2: Two-Stage Architecture (Classifier + Active Regressor)
    print("Training Model 2: Two-Stage Classifier + Active Regressor...")
    clf = lgb.LGBMClassifier(
        n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, y_train_state)
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    # Optimize classification probability threshold on validation set
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.2, 0.9, 0.05):
        score = f1_score(y_val_state, (val_probs >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_th = score, float(th)
            
    print(f"  Optimized Stage-1 ON threshold: {best_th:.2f} (Val F1: {best_f1:.4f})")
    
    # Stage 2: Regressor trained strictly on active intervals
    active_mask = (y_train_pow >= 20.0)
    reg_active = lgb.LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1, n_jobs=-1
    )
    reg_active.fit(X_train[active_mask], y_train_pow[active_mask])
    
    test_probs = clf.predict_proba(X_test)[:, 1]
    pred_two_stage_state = (test_probs >= best_th).astype(int)
    active_pred = np.maximum(0.0, reg_active.predict(X_test))
    pred_two_stage = np.where(pred_two_stage_state == 1, active_pred, 0.0)
    metrics_two_stage = compute_metrics(y_test_pow.values, pred_two_stage)
    
    # Print comparison table
    results_df = pd.DataFrame([
        {"Model": "Baseline 1: Always-Off", **metrics_off},
        {"Model": "Baseline 2: Hourly Mean Profile", **metrics_hourly},
        {"Model": "Model 1: Direct LightGBM Regressor", **metrics_direct},
        {"Model": "Model 2: Two-Stage (Clf + Reg)", **metrics_two_stage}
    ])
    
    print("\nNILM Disaggregation Test Results Summary:")
    print(results_df.to_string(index=False))
    
    # Attach predictions to test slice for plotting
    test_slice["pred_pow"] = pred_direct
    test_slice["pred_two_stage"] = pred_two_stage
    
    # Visualizations: Multi-cycle test window + Failure case
    plot_evaluation_and_failure(test_slice, output_path=output_plot)
    
    return results_df

def plot_evaluation_and_failure(test_slice, output_path="figures/sec3_actual_vs_predicted.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    
    # Panel 1: Multi-cycle accurate disaggregation window (2015-03-25 to 2015-03-26)
    start_c = pd.Timestamp("2015-03-25 06:00:00")
    end_c = pd.Timestamp("2015-03-26 22:00:00")
    c_slice = test_slice.loc[start_c:end_c]
    
    ax1.plot(c_slice.index, c_slice["Aggregate"], label="Aggregate Power", color="#bdc3c7", linewidth=1.0, alpha=0.8)
    ax1.plot(c_slice.index, c_slice["Appliance5"], label="Actual Washing Machine (Ground Truth)", color="#2c3e50", linewidth=1.8)
    ax1.plot(c_slice.index, c_slice["pred_pow"], label="NILM Predicted Power (LightGBM)", color="#e74c3c", linestyle="--", linewidth=1.5)
    
    ax1.set_title("Evaluation on Test Data: Multi-Cycle Disaggregation Performance", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Power (W)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim(-50, 3200)
    
    ax1.annotate("Water Heating Peak Reconstructed\n(~2.3 kW Thermal Burst)", 
                 xy=(pd.Timestamp("2015-03-25 10:15:00"), 2200), 
                 xytext=(pd.Timestamp("2015-03-25 12:30:00"), 2600),
                 arrowprops=dict(facecolor="#e74c3c", shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="semibold", color="#e74c3c")
    
    # Panel 2: Real-world Failure Case Analysis - Overlapping 2kW Resistive Load (Dishwasher)
    # Window: 2015-03-27 12:00 to 18:00
    start_f = pd.Timestamp("2015-03-27 12:00:00")
    end_f = pd.Timestamp("2015-03-27 18:00:00")
    f_slice = test_slice.loc[start_f:end_f]
    
    ax2.plot(f_slice.index, f_slice["Aggregate"], label="Aggregate Power", color="#bdc3c7", linewidth=1.0, alpha=0.8)
    ax2.plot(f_slice.index, f_slice["Appliance6"], label="Interfering Appliance (Dishwasher)", color="#e67e22", linewidth=1.8)
    ax2.plot(f_slice.index, f_slice["Appliance5"], label="Actual Washing Machine (Ground Truth: 0 W)", color="#2c3e50", linewidth=2.0)
    ax2.plot(f_slice.index, f_slice["pred_pow"], label="NILM Predicted Power (False Positive)", color="#c0392b", linestyle="--", linewidth=1.5)
    
    ax2.set_title("NILM Failure Mode Analysis: Resistive Heating Element Ambiguity (Dishwasher vs Washing Machine)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Power (W)", fontsize=10)
    ax2.set_xlabel("Timestamp", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.set_ylim(-50, 3200)
    
    ax2.annotate("Interfering 2.2 kW Dishwasher Element\nCauses False Positive Washing Machine Prediction", 
                 xy=(pd.Timestamp("2015-03-27 14:15:00"), 2250), 
                 xytext=(pd.Timestamp("2015-03-27 14:45:00"), 2700),
                 arrowprops=dict(facecolor="#c0392b", shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="semibold", color="#c0392b")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate NILM disaggregation models")
    parser.add_argument("--data", default="data/clean/CLEAN_House1.csv", help="Clean dataset path")
    parser.add_argument("--output_plot", default="figures/sec3_actual_vs_predicted.png", help="Output figure path")
    args = parser.parse_args()
    
    train_and_evaluate(data_path=args.data, output_plot=args.output_plot)
    print("Section 3 disaggregation pipeline completed successfully.")

if __name__ == "__main__":
    main()
