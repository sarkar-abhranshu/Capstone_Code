"""Task 2 data preparation for t+3 fertility index forecasting.

This script reuses the preprocessing pipeline from preprocess2.ipynb and the
fertility index construction from Task1_Option2_Improved (1).py.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DROP_COLUMNS = ["Iron", "Copper", "SAVI"]
OUTLIER_FEATURES = ["Nitrogen", "Slope", "AOD", "Rain"]

# Same core feature set defined in preprocess2.ipynb.
CORE_FEATURES = [
    "Rain_log",
    "Temp",
    "LST",
    "SoilMoisture",
    "NDVI",
    "green_fraction",
    "Clay",
    "Nitrogen_log",
    "pH",
    "BulkDensity",
    "Elevation",
    "Slope_log",
    "AOD",
    "NO2_log",
    "SO2_log",
    "Month_Sin",
    "Month_Cos",
    "longitude",
    "latitude",
]

# Same research-based weights from Task1_Option2_Improved (1).py.
FERTILITY_WEIGHTS = {
    "Nitrogen": 0.40,
    "pH": 0.25,
    "NDVI": 0.15,
    "SoilMoisture": 0.10,
    "Clay": 0.10,
}


def extract_coordinates(geo_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract longitude and latitude from the .geo JSON string."""
    try:
        geo_dict = json.loads(geo_str)
        coords = geo_dict["coordinates"]
        return float(coords[0]), float(coords[1])
    except Exception:
        return None, None


def normalize_minmax(values: pd.Series) -> np.ndarray:
    """Min-max normalize a numeric series into [0, 1]."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(values.to_numpy().reshape(-1, 1)).ravel()


def normalize_ph_optimal(
    ph_values: pd.Series, optimal: float = 6.5, tolerance: float = 1.0
) -> np.ndarray:
    """Score pH by distance from an agronomic optimum."""
    deviation = np.abs(ph_values.to_numpy() - optimal)
    scores = 1.0 - (deviation / tolerance)
    return np.clip(scores, 0.0, 1.0)


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing logic from preprocess2.ipynb."""
    df_cleaned = df.copy()

    # Drop redundant features (if present).
    to_drop = [col for col in DROP_COLUMNS if col in df_cleaned.columns]
    if to_drop:
        df_cleaned = df_cleaned.drop(columns=to_drop)

    # Parse date exactly as monthly entries first, fallback to generic parsing.
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], format="%Y-%m", errors="coerce")
    if df_cleaned["date"].isna().any():
        df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    if df_cleaned["date"].isna().any():
        raise ValueError("Date parsing failed for some rows in 'date' column.")
    df_cleaned["date"] = df_cleaned["date"].dt.to_period("M").dt.to_timestamp()

    # Extract spatial coordinates from .geo, same as preprocessing notebook.
    if ".geo" in df_cleaned.columns:
        coords = df_cleaned[".geo"].apply(extract_coordinates)
        df_cleaned["longitude"] = coords.str[0]
        df_cleaned["latitude"] = coords.str[1]

    if "longitude" not in df_cleaned.columns or "latitude" not in df_cleaned.columns:
        raise ValueError("Could not derive longitude/latitude from dataset.")

    if df_cleaned[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Missing longitude/latitude found after .geo coordinate extraction.")

    # Fill NO2 missing values with median (as in preprocess2.ipynb).
    if "NO2" in df_cleaned.columns and df_cleaned["NO2"].isna().any():
        df_cleaned["NO2"] = df_cleaned["NO2"].fillna(df_cleaned["NO2"].median())

    # Cap outliers with IQR-based Winsorization.
    for feature in OUTLIER_FEATURES:
        if feature not in df_cleaned.columns:
            continue
        q1 = df_cleaned[feature].quantile(0.25)
        q3 = df_cleaned[feature].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_cleaned[feature] = df_cleaned[feature].clip(lower=lower, upper=upper)

    # Log transforms from preprocess2.ipynb.
    # Guard against occasional negative sensor artifacts before log1p.
    df_cleaned["Rain_log"] = np.log1p(df_cleaned["Rain"].clip(lower=0))
    df_cleaned["NO2_log"] = np.log1p((df_cleaned["NO2"] * 1e6).clip(lower=0))
    df_cleaned["Slope_log"] = np.log1p(df_cleaned["Slope"].clip(lower=0))
    df_cleaned["Nitrogen_log"] = np.log1p(df_cleaned["Nitrogen"].clip(lower=0))
    so2_shifted = df_cleaned["SO2"] - df_cleaned["SO2"].min() + 1e-10
    df_cleaned["SO2_log"] = np.log1p(so2_shifted * 1e6)

    # Temporal features from actual date.
    df_cleaned["Month_Actual"] = df_cleaned["date"].dt.month
    df_cleaned["Year"] = df_cleaned["date"].dt.year
    df_cleaned["Month_Sin"] = np.sin(2 * np.pi * df_cleaned["Month_Actual"] / 12)
    df_cleaned["Month_Cos"] = np.cos(2 * np.pi * df_cleaned["Month_Actual"] / 12)

    missing_core = [feature for feature in CORE_FEATURES if feature not in df_cleaned.columns]
    if missing_core:
        raise ValueError(f"Missing required core features after preprocessing: {missing_core}")

    return df_cleaned


def add_fertility_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite fertility index using Task 1 Option 2 logic."""
    out = df.copy()
    out["N_score"] = normalize_minmax(out["Nitrogen"])
    out["pH_score"] = normalize_ph_optimal(out["pH"])
    out["NDVI_score"] = normalize_minmax(out["NDVI"])
    out["Moisture_score"] = normalize_minmax(out["SoilMoisture"])
    out["Clay_score"] = normalize_minmax(out["Clay"])

    out["FertilityIndex"] = (
        FERTILITY_WEIGHTS["Nitrogen"] * out["N_score"]
        + FERTILITY_WEIGHTS["pH"] * out["pH_score"]
        + FERTILITY_WEIGHTS["NDVI"] * out["NDVI_score"]
        + FERTILITY_WEIGHTS["SoilMoisture"] * out["Moisture_score"]
        + FERTILITY_WEIGHTS["Clay"] * out["Clay_score"]
    )
    return out


def add_site_identifier(df: pd.DataFrame, site_col: str = "site_id") -> pd.DataFrame:
    """Create stable site IDs from coordinates when explicit IDs are unavailable."""
    out = df.copy()

    if site_col in out.columns:
        out[site_col] = out[site_col].astype(str)
        return out

    # The raw file has ~805 unique .geo entries, so coordinate-based ID is stable.
    out[site_col] = (
        out["longitude"].round(6).astype(str) + "_" + out["latitude"].round(6).astype(str)
    )
    return out


def build_time_split_masks(target_dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the same time split policy as existing notebooks."""
    year = target_dates.dt.year
    month = target_dates.dt.month

    train_mask = (year == 2021) | ((year == 2022) & (month <= 6))
    val_mask = (year == 2022) & (month >= 7)
    test_mask = year == 2023
    return train_mask.to_numpy(), val_mask.to_numpy(), test_mask.to_numpy()


def build_sequences(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    lookback_months: int,
    forecast_horizon_months: int,
    site_col: str,
    target_col: str = "FertilityIndex",
    gap_fill_strategy: str = "strict",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, int]]:
    """Build supervised time-series samples for each site.

    Input window: t-6 ... t-1 (6 months)
    Target: t+3 (3 months ahead from reference t)

    Relative to the last observed month (t-1), the target is at +4 months.
    """
    required_columns = set(feature_columns) | {site_col, "date", target_col}
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build sequences. Missing columns: {missing}")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    metadata_records: List[Dict[str, object]] = []

    stats = {
        "total_sites": 0,
        "sites_insufficient_history": 0,
        "sites_no_valid_samples": 0,
        "missing_months_added": 0,
        "missing_values_filled": 0,
        "samples_skipped_missing_values": 0,
        "samples_created": 0,
    }

    valid_gap_fill = {"strict", "ffill", "interpolate"}
    if gap_fill_strategy not in valid_gap_fill:
        raise ValueError(
            f"Invalid gap_fill_strategy: {gap_fill_strategy}. "
            f"Choose from {sorted(valid_gap_fill)}"
        )

    target_offset = forecast_horizon_months + 1
    min_required_months = lookback_months + target_offset

    for site_id, site_df in df.groupby(site_col, sort=False):
        stats["total_sites"] += 1

        site_df = site_df.sort_values("date").drop_duplicates(subset="date", keep="last")
        if len(site_df) < min_required_months:
            stats["sites_insufficient_history"] += 1
            continue

        full_month_index = pd.date_range(
            start=site_df["date"].min(), end=site_df["date"].max(), freq="MS"
        )
        stats["missing_months_added"] += max(len(full_month_index) - len(site_df), 0)

        site_monthly = site_df.set_index("date").reindex(full_month_index)
        site_monthly.index.name = "date"
        site_monthly[site_col] = str(site_id)

        fill_columns = list(dict.fromkeys(list(feature_columns) + [target_col]))
        if gap_fill_strategy in {"ffill", "interpolate"}:
            before_missing = int(site_monthly[fill_columns].isna().sum().sum())
            if gap_fill_strategy == "interpolate":
                site_monthly[fill_columns] = (
                    site_monthly[fill_columns]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                )
            else:
                site_monthly[fill_columns] = site_monthly[fill_columns].ffill().bfill()
            after_missing = int(site_monthly[fill_columns].isna().sum().sum())
            stats["missing_values_filled"] += max(before_missing - after_missing, 0)

        created_for_site = 0
        max_window_end = len(site_monthly) - target_offset - 1
        for window_end in range(lookback_months - 1, max_window_end + 1):
            window_start = window_end - lookback_months + 1
            target_idx = window_end + target_offset

            x_window = site_monthly.iloc[window_start : window_end + 1]
            y_row = site_monthly.iloc[target_idx]

            if x_window[list(feature_columns)].isna().any().any() or pd.isna(y_row[target_col]):
                stats["samples_skipped_missing_values"] += 1
                continue

            X_list.append(x_window[list(feature_columns)].to_numpy(dtype=np.float32))
            y_list.append(float(y_row[target_col]))

            anchor_date = x_window.index[-1]  # t-1
            reference_date = anchor_date + pd.offsets.MonthBegin(1)  # t
            target_date = site_monthly.index[target_idx]  # t+3

            metadata_records.append(
                {
                    site_col: str(site_id),
                    "sequence_start_date": x_window.index[0],
                    "anchor_date": anchor_date,
                    "reference_date": reference_date,
                    "target_date": target_date,
                    "sample_index": len(X_list) - 1,
                }
            )

            created_for_site += 1
            stats["samples_created"] += 1

        if created_for_site == 0:
            stats["sites_no_valid_samples"] += 1

    if not X_list:
        raise ValueError(
            "No valid samples were created. Check missing months/values and lookback-horizon setup."
        )

    metadata = pd.DataFrame(metadata_records)
    metadata = metadata.sort_values(["target_date", site_col]).reset_index(drop=True)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    # Reorder arrays to match sorted metadata.
    sample_order = metadata["sample_index"].to_numpy(dtype=int)
    X = X[sample_order]
    y = y[sample_order]
    metadata = metadata.drop(columns=["sample_index"])

    return X, y, metadata, stats


def standardize_sequences(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Standardize feature channels across all samples and timesteps."""
    n_samples, n_steps, n_features = X.shape
    reshaped = X.reshape(-1, n_features)
    scaler = StandardScaler()
    reshaped_scaled = scaler.fit_transform(reshaped)
    X_scaled = reshaped_scaled.reshape(n_samples, n_steps, n_features).astype(np.float32)
    return X_scaled, scaler


def save_prepared_outputs(
    output_path: Path,
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    feature_columns: Sequence[str],
    scaler: StandardScaler,
    lookback_months: int,
    forecast_horizon_months: int,
    site_col: str,
    gap_fill_strategy: str,
) -> Dict[str, Path]:
    """Persist arrays, metadata, scaler, and config for Task 2 training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    train_mask, val_mask, test_mask = build_time_split_masks(metadata["target_date"])

    np.savez_compressed(
        output_path,
        X_raw=X_raw,
        X_scaled=X_scaled,
        y=y,
        X_train=X_scaled[train_mask],
        y_train=y[train_mask],
        X_val=X_scaled[val_mask],
        y_val=y[val_mask],
        X_test=X_scaled[test_mask],
        y_test=y[test_mask],
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_names=np.asarray(feature_columns),
        site_ids=np.asarray(metadata[site_col].astype(str).tolist(), dtype="U64"),
        sequence_start_dates=np.asarray(
            metadata["sequence_start_date"].dt.strftime("%Y-%m-%d").tolist(), dtype="U10"
        ),
        anchor_dates=np.asarray(
            metadata["anchor_date"].dt.strftime("%Y-%m-%d").tolist(), dtype="U10"
        ),
        reference_dates=np.asarray(
            metadata["reference_date"].dt.strftime("%Y-%m-%d").tolist(), dtype="U10"
        ),
        target_dates=np.asarray(
            metadata["target_date"].dt.strftime("%Y-%m-%d").tolist(), dtype="U10"
        ),
        lookback_months=np.int32(lookback_months),
        forecast_horizon_months=np.int32(forecast_horizon_months),
    )

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.csv")
    metadata.to_csv(metadata_path, index=False)

    scaler_path = output_path.with_name(f"{output_path.stem}_scaler.pkl")
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    config_path = output_path.with_name(f"{output_path.stem}_config.json")
    config = {
        "lookback_months": lookback_months,
        "forecast_horizon_months": forecast_horizon_months,
        "input_definition": "t-6 ... t-1",
        "target_definition": "t+3",
        "site_column": site_col,
        "gap_fill_strategy": gap_fill_strategy,
        "feature_columns": list(feature_columns),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "npz": output_path,
        "metadata_csv": metadata_path,
        "scaler_pkl": scaler_path,
        "config_json": config_path,
    }


def load_prepared_data(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load the prepared npz payload for training/validation scripts."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Prepared data file not found: {npz_path}")
    # allow_pickle=True keeps backward compatibility with earlier object-dtype saves.
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def prepare_task2_dataset(
    input_csv: Path,
    output_npz: Path,
    lookback_months: int = 6,
    forecast_horizon_months: int = 3,
    site_col: str = "site_id",
    gap_fill_strategy: str = "strict",
) -> Dict[str, object]:
    """End-to-end Task 2 data preparation."""
    raw_df = pd.read_csv(input_csv)
    preprocessed_df = apply_preprocessing(raw_df)
    preprocessed_df = add_fertility_index(preprocessed_df)
    preprocessed_df = add_site_identifier(preprocessed_df, site_col=site_col)
    preprocessed_df = preprocessed_df.sort_values([site_col, "date"]).reset_index(drop=True)

    # Include FI lag values as autoregressive context along with core drivers.
    feature_columns = list(CORE_FEATURES) + ["FertilityIndex"]
    X_raw, y, metadata, stats = build_sequences(
        preprocessed_df,
        feature_columns=feature_columns,
        lookback_months=lookback_months,
        forecast_horizon_months=forecast_horizon_months,
        site_col=site_col,
        target_col="FertilityIndex",
        gap_fill_strategy=gap_fill_strategy,
    )

    X_scaled, scaler = standardize_sequences(X_raw)
    paths = save_prepared_outputs(
        output_path=output_npz,
        X_raw=X_raw,
        X_scaled=X_scaled,
        y=y,
        metadata=metadata,
        feature_columns=feature_columns,
        scaler=scaler,
        lookback_months=lookback_months,
        forecast_horizon_months=forecast_horizon_months,
        site_col=site_col,
        gap_fill_strategy=gap_fill_strategy,
    )

    train_mask, val_mask, test_mask = build_time_split_masks(metadata["target_date"])
    summary = {
        "stats": stats,
        "paths": paths,
        "num_samples": int(len(y)),
        "num_features": int(X_raw.shape[2]),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Task 2 data for t+3 forecasting.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Bangalore_Monthly_Final_Corrected.csv"),
        help="Path to Bangalore monthly CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("task2_processed_data.npz"),
        help="Path for saved npz dataset.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=6,
        help="Number of historical months in each input window.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Forecast horizon (months ahead from reference t).",
    )
    parser.add_argument(
        "--site-col",
        type=str,
        default="site_id",
        help="Column name to use as site ID (created if absent).",
    )
    parser.add_argument(
        "--gap-fill",
        type=str,
        default="strict",
        choices=["strict", "ffill", "interpolate"],
        help="How to handle missing months/values after monthly reindexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_task2_dataset(
        input_csv=args.input,
        output_npz=args.output,
        lookback_months=args.lookback,
        forecast_horizon_months=args.horizon,
        site_col=args.site_col,
        gap_fill_strategy=args.gap_fill,
    )

    print("=" * 80)
    print("TASK 2 DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Total samples: {summary['num_samples']:,}")
    print(f"Input shape: ({summary['num_samples']}, {args.lookback}, {summary['num_features']})")
    print(f"Gap fill strategy: {args.gap_fill}")
    print(f"Train samples: {summary['train_samples']:,}")
    print(f"Validation samples: {summary['val_samples']:,}")
    print(f"Test samples: {summary['test_samples']:,}")
    print("\nSite-level checks:")
    for key, value in summary["stats"].items():
        print(f"  {key}: {value:,}")
    print("\nSaved files:")
    for name, path in summary["paths"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
