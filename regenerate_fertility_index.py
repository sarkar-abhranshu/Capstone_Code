"""
Regenerate fertility_index_data.csv using the exact preprocessing and FertilityIndex
logic from prepare_task2_data.py to ensure 100% parity across tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import warnings
from prepare_task2_data import apply_preprocessing, add_fertility_index, add_site_identifier

warnings.filterwarnings("ignore")


def classify_fertility(fi_score: float) -> str:
    """Classify fertility index into Low/Medium/High categories matching frontend thresholds."""
    if fi_score >= 0.70:
        return "High"
    if fi_score >= 0.50:
        return "Medium"
    return "Low"


def main():
    print("=" * 80)
    print("REGENERATING FERTILITY INDEX DATA (ALIGNED WITH PREPARE_TASK2_DATA)")
    print("=" * 80)
    
    # 1. Load raw data
    raw_path = "csv/Bangalore_2019_2024_RAW.csv"
    print(f"\n1. Loading raw data from {raw_path}...")
    df_raw = pd.read_csv(raw_path)
    print(f"   Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # 2. Apply preprocessing from prepare_task2_data
    print("\n2. Applying preprocessing (IQR Winsorization, Silt derivation, log transforms)...")
    df_clean = apply_preprocessing(df_raw)
    
    # 3. Add Fertility Index
    print("\n3. Calculating Fertility Index with aligned weights and scores...")
    df_fertility = add_fertility_index(df_clean)
    
    # 4. Add site identifier
    df_fertility = add_site_identifier(df_fertility, site_col="site_id")
    
    # 5. Classify fertility
    df_fertility["FertilityClass"] = df_fertility["FertilityIndex"].apply(classify_fertility)
    
    # 6. Sort chronologically and spatially
    df_fertility = df_fertility.sort_values(["date", "longitude", "latitude"]).reset_index(drop=True)
    
    # Print summary statistics
    print("\n4. Fertility Index Summary:")
    print(f"   Range:  [{df_fertility['FertilityIndex'].min():.3f}, {df_fertility['FertilityIndex'].max():.3f}]")
    print(f"   Mean:   {df_fertility['FertilityIndex'].mean():.3f} ± {df_fertility['FertilityIndex'].std():.3f}")
    
    print("\n   Fertility Class Distribution (%):")
    class_dist = df_fertility["FertilityClass"].value_counts(normalize=True).mul(100).round(2)
    for cls, pct in class_dist.items():
        print(f"     {cls:10s} {pct:6.2f}%")
        
    output_file = "csv/fertility_index_data.csv"
    print(f"\n5. Saving to {output_file}...")
    df_fertility.to_csv(output_file, index=False)
    print(f"   Saved {len(df_fertility)} rows, {len(df_fertility.columns)} columns")
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
