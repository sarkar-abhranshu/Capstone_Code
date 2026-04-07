"""Walk-forward validation for Task 2 (t+3 fertility forecasting)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prepare_task2_data import load_prepared_data, prepare_task2_dataset


def get_model_factory(seed: int) -> Tuple[Callable[[], object], str]:
    """Return preferred regressor factory (XGBoost, else RandomForest fallback)."""
    try:
        import xgboost as xgb

        def factory() -> object:
            return xgb.XGBRegressor(
                n_estimators=250,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
            )

        return factory, "XGBoost"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        def factory() -> object:
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=-1,
            )

        return factory, "RandomForestFallback"


def run_walk_forward_validation(
    X_flat: np.ndarray,
    y: np.ndarray,
    target_dates: pd.Series,
    site_ids: np.ndarray,
    model_factory: Callable[[], object],
    min_train_months: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward validation with monthly forecast steps."""
    target_dates = pd.Series(pd.to_datetime(target_dates), name="target_date")
    unique_months = np.array(sorted(target_dates.dropna().unique()))

    prediction_rows = []
    monthly_rows = []

    for month_idx, forecast_month in enumerate(unique_months):
        if month_idx < min_train_months:
            continue

        train_mask = (target_dates < forecast_month).to_numpy()
        test_mask = (target_dates == forecast_month).to_numpy()

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = model_factory()
        model.fit(X_flat[train_mask], y[train_mask])
        month_pred = model.predict(X_flat[test_mask])
        month_true = y[test_mask]

        rmse = float(np.sqrt(mean_squared_error(month_true, month_pred)))
        mae = float(mean_absolute_error(month_true, month_pred))
        monthly_rows.append(
            {
                "target_month": pd.Timestamp(forecast_month),
                "RMSE": rmse,
                "MAE": mae,
                "n_samples": int(test_mask.sum()),
            }
        )

        month_sites = site_ids[test_mask]
        month_dates = target_dates[test_mask].to_numpy()
        for i in range(len(month_pred)):
            prediction_rows.append(
                {
                    "site_id": str(month_sites[i]),
                    "target_date": pd.Timestamp(month_dates[i]),
                    "actual": float(month_true[i]),
                    "predicted": float(month_pred[i]),
                }
            )

    if not prediction_rows:
        raise ValueError(
            "Walk-forward produced no predictions. Reduce min_train_months or check date coverage."
        )

    predictions_df = pd.DataFrame(prediction_rows).sort_values(["target_date", "site_id"])
    monthly_df = pd.DataFrame(monthly_rows).sort_values("target_month").reset_index(drop=True)
    return predictions_df, monthly_df


def plot_validation_results(
    predictions_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    plot_path: Path,
    model_name: str,
) -> None:
    """Plot actual-vs-predicted trend and RMSE-over-time trend."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    monthly_actual_pred = (
        predictions_df.groupby("target_date")[["actual", "predicted"]].mean().reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(
        monthly_actual_pred["target_date"],
        monthly_actual_pred["actual"],
        marker="o",
        linewidth=2,
        label="Actual (monthly mean)",
    )
    axes[0].plot(
        monthly_actual_pred["target_date"],
        monthly_actual_pred["predicted"],
        marker="o",
        linewidth=2,
        label="Predicted (monthly mean)",
    )
    axes[0].set_title(f"Walk-Forward: Actual vs Predicted ({model_name})")
    axes[0].set_ylabel("Fertility Index")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        monthly_df["target_month"],
        monthly_df["RMSE"],
        marker="o",
        linewidth=2,
        color="tab:red",
    )
    axes[1].set_title("RMSE Over Time (Walk-Forward)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("Target Month")
    axes[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2 walk-forward validation.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("task2_processed_data.npz"),
        help="Prepared data file from prepare_task2_data.py.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Bangalore_Monthly_Final_Corrected.csv"),
        help="Raw CSV path (used if --data does not exist).",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Regenerate prepared dataset before validation.",
    )
    parser.add_argument(
        "--gap-fill",
        type=str,
        default="strict",
        choices=["strict", "ffill", "interpolate"],
        help="Missing-month/value handling during auto-preparation.",
    )
    parser.add_argument(
        "--min-train-months",
        type=int,
        default=12,
        help="Minimum number of target months before first walk-forward prediction.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("task2_walk_forward_plot.png"),
        help="Path to save walk-forward plot.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("task2_walk_forward_predictions.csv"),
        help="Path to save per-sample walk-forward predictions.",
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=Path("task2_walk_forward_monthly_metrics.csv"),
        help="Path to save monthly walk-forward metrics.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("task2_walk_forward_summary.json"),
        help="Path to save summary metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.force_prepare or not args.data.exists():
        print("Prepared dataset missing or refresh requested. Running data preparation...")
        prepare_task2_dataset(
            input_csv=args.input,
            output_npz=args.data,
            lookback_months=6,
            forecast_horizon_months=3,
            site_col="site_id",
            gap_fill_strategy=args.gap_fill,
        )

    payload = load_prepared_data(args.data)
    X_raw = payload["X_raw"].astype(np.float32)
    y = payload["y"].astype(np.float32)
    target_dates = pd.to_datetime(payload["target_dates"].astype(str))
    site_ids = payload["site_ids"].astype(str)

    X_flat = X_raw.reshape(X_raw.shape[0], -1)
    model_factory, model_name = get_model_factory(seed=args.seed)

    predictions_df, monthly_df = run_walk_forward_validation(
        X_flat=X_flat,
        y=y,
        target_dates=target_dates,
        site_ids=site_ids,
        model_factory=model_factory,
        min_train_months=args.min_train_months,
    )

    overall_rmse = float(np.sqrt(mean_squared_error(predictions_df["actual"], predictions_df["predicted"])))
    overall_mae = float(mean_absolute_error(predictions_df["actual"], predictions_df["predicted"]))
    error_series = predictions_df["predicted"] - predictions_df["actual"]

    rmse_mean = float(monthly_df["RMSE"].mean())
    rmse_std = float(monthly_df["RMSE"].std(ddof=0))
    rmse_cv = float(rmse_std / rmse_mean) if rmse_mean > 0 else float("nan")

    summary = {
        "model_used": model_name,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "monthly_rmse_mean": rmse_mean,
        "monthly_rmse_std": rmse_std,
        "monthly_rmse_cv": rmse_cv,
        "prediction_bias_mean": float(error_series.mean()),
        "prediction_bias_std": float(error_series.std(ddof=0)),
        "num_forecast_months": int(monthly_df.shape[0]),
        "num_predictions": int(predictions_df.shape[0]),
    }

    plot_validation_results(
        predictions_df=predictions_df,
        monthly_df=monthly_df,
        plot_path=args.plot_output,
        model_name=model_name,
    )

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.monthly_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(args.predictions_output, index=False)
    monthly_df.to_csv(args.monthly_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print("TASK 2 WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Monthly RMSE mean: {rmse_mean:.4f}")
    print(f"Monthly RMSE std: {rmse_std:.4f}")
    print(f"Monthly RMSE CV: {rmse_cv:.4f}")
    print(f"Saved plot: {args.plot_output}")
    print(f"Saved predictions: {args.predictions_output}")
    print(f"Saved monthly metrics: {args.monthly_output}")
    print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
