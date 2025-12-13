import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path to allow imports
import sys
sys.path.append(os.path.dirname(__file__))

from train_tft import (
    load_label_frames,
    load_hourly_sensor_frames,
    build_model_dataframe,
    Config,
    SENSOR_FEATURES
)
from run_tabular_models import build_feature_dataframe

def visualize_missing_data(output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data for missingness analysis...")
    # 1. Load Raw Label and Sensor Data
    label_df = load_label_frames()
    sensor_frames = load_hourly_sensor_frames()
    
    # 2. Build Model DataFrame (Time-Series) - this has some imputation (median) applied
    print("Building model dataframe (Time-Series)...")
    cfg = Config(history_hours=168) # Use a standard history
    # Note: build_model_dataframe now applies median imputation for sensors.
    # To visualize "raw" missingness from the sensors, we should look at sensor_frames directly
    # or build sequences without imputation if possible. 
    # But build_model_dataframe hardcodes the median fill now.
    
    # Let's visualize the "Effective" missingness passed to the models 
    # (which should be low/zero for sensors now)
    # AND the "Raw" missingness from sensor_frames.
    
    # --- A. Raw Sensor Missingness ---
    print("Analyzing raw sensor missingness...")
    missing_stats = []
    for feature, df in sensor_frames.items():
        # df is timestamp x participant
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        missing_stats.append({
            "Feature": feature,
            "Missing %": missing_pct,
            "Total Cells": total_cells
        })
    
    raw_missing_df = pd.DataFrame(missing_stats).sort_values("Missing %", ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=raw_missing_df, x="Missing %", y="Feature", palette="viridis")
    plt.title("Missing Data Percentage by Sensor (Raw Hourly Data)")
    plt.xlabel("Missing %")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_raw_sensors.png"))
    plt.close()
    
    # --- B. Feature DataFrame Missingness (Post-Engineering) ---
    print("Building feature dataframe (Tabular Features)...")
    # This invokes the pipeline in run_tabular_models.py
    # We expect some NaNs here now that we removed fillna(0.0)
    feature_df = build_feature_dataframe(history_hours=168)
    
    # Calculate missingness per column
    missing_series = feature_df.isna().sum() / len(feature_df) * 100
    missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)
    
    if not missing_cols.empty:
        plt.figure(figsize=(14, 10))
        # Plot top 50 missing columns to avoid clutter
        top_missing = missing_cols.head(50)
        sns.barplot(x=top_missing.values, y=top_missing.index, palette="Reds_r")
        plt.title(f"Top 50 Features with Missing Values (Post-Engineering)\n(Total Features: {len(feature_df.columns)})")
        plt.xlabel("Missing %")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_tabular_features.png"))
        plt.close()
    else:
        print("No missing values found in tabular feature dataframe.")

    # --- C. Missingness Heatmap (Sample) ---
    # Sample 1000 rows and top 50 missing features for heatmap
    if not missing_cols.empty:
        plt.figure(figsize=(16, 12))
        cols_to_plot = missing_cols.head(50).index
        sample_df = feature_df[cols_to_plot].sample(min(1000, len(feature_df)), random_state=42)
        sns.heatmap(sample_df.isna(), cbar=False, cmap="binary")
        plt.title("Missingness Heatmap (Top 50 Missing Features, Random 1000 Samples)")
        plt.xlabel("Features")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_heatmap.png"))
        plt.close()

    # Save summary csv
    raw_missing_df.to_csv(os.path.join(output_dir, "missing_raw_summary.csv"), index=False)
    missing_cols.to_csv(os.path.join(output_dir, "missing_tabular_summary.csv"))
    
    print(f"Analysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    visualize_missing_data()
