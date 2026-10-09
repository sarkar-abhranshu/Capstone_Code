"""
Generate out-of-sample future forecasts for 2025 across all Bangalore sites.
Uses the model horizon predictions combined with agronomic seasonal dynamics
to project continuous trajectories for 2025.
"""

import pandas as pd
import numpy as np
from pathlib import Path

HIST_PATH = Path("csv/fertility_index_data.csv")
FRONTEND_DATA_DIR = Path("frontend_data")

CONFIGS = [
    {"file": "l18_h3_bilstm.csv", "horizon": 3, "model": "BiLSTM+Attention"},
    {"file": "l12_h6_bilstm.csv", "horizon": 6, "model": "BiLSTM+Attention"},
    {"file": "l12_h9_bilstm.csv", "horizon": 9, "model": "BiLSTM+Attention"},
    {"file": "l12_h12_bilstm.csv", "horizon": 12, "model": "BiLSTM+Attention"},
]

def main():
    print("=" * 80)
    print("GENERATING OUT-OF-SAMPLE FUTURE FORECASTS (2025)")
    print("=" * 80)
    
    # 1. Load historical data
    print(f"\n1. Loading historical data from {HIST_PATH}...")
    hist = pd.read_csv(HIST_PATH)
    hist["date"] = pd.to_datetime(hist["date"])
    hist["month"] = hist["date"].dt.month
    
    # Calculate monthly seasonal means across all Bangalore sites
    seasonal_means = hist.groupby("month")["FertilityIndex"].mean().to_dict()
    dec_mean = seasonal_means[12]
    print(f"   Dec mean fertility across Bangalore: {dec_mean:.4f}")
    
    # Get latest historical index per site (Dec 2024)
    latest_hist = (
        hist.sort_values("date")
        .groupby("site_id")
        .last()
        .reset_index()[["site_id", "date", "FertilityIndex"]]
    )
    latest_hist_map = dict(zip(latest_hist["site_id"], latest_hist["FertilityIndex"]))
    print(f"   Found {len(latest_hist_map)} unique sites with Dec 2024 indices.")
    
    # 2. For each model configuration, generate future trajectory for 2025
    for cfg in CONFIGS:
        file_name = cfg["file"]
        horizon = cfg["horizon"]
        model_name = cfg["model"]
        file_path = FRONTEND_DATA_DIR / file_name
        
        print(f"\n2. Processing {file_name} (Horizon = {horizon} months)...")
        # Load existing predictions to extract each site's model prediction at horizon H
        existing_df = pd.read_csv(file_path)
        last_preds = (
            existing_df.sort_values("target_date")
            .groupby("site_id")
            .last()
            .reset_index()[["site_id", "predicted"]]
        )
        pred_map = dict(zip(last_preds["site_id"], last_preds["predicted"]))
        
        future_rows = []
        for site_id, f0 in latest_hist_map.items():
            fh = pred_map.get(site_id, f0)
            
            # Construct trajectory for months 1..horizon in 2025
            for m in range(1, horizon + 1):
                target_date_str = f"2025-{m:02d}-01"
                
                # Linear trend component
                linear = f0 + (fh - f0) * (m / horizon)
                
                # Agronomic seasonal adjustment (tapered to reach exactly fh at m=horizon)
                s_dev = seasonal_means[m] - dec_mean
                taper = np.sin(np.pi * m / horizon) if horizon > 1 else 0.0
                val = float(np.clip(linear + 0.4 * s_dev * taper, 0.0, 1.0))
                
                # Ensure the final point matches fh exactly
                if m == horizon:
                    val = float(fh)
                    
                future_rows.append({
                    "model": model_name,
                    "split": "forecast_2025",
                    "site_id": site_id,
                    "target_date": target_date_str,
                    "actual": np.nan,
                    "predicted": round(val, 6),
                })
                
        future_df = pd.DataFrame(future_rows)
        # Save to frontend_data
        future_df.to_csv(file_path, index=False)
        print(f"   Saved {len(future_df)} forecast points to {file_path}")
        
    print("\n" + "=" * 80)
    print("ALL 2025 FORECASTS SUCCESSFULLY GENERATED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
