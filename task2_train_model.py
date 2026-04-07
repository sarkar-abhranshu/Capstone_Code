"""Train and compare Task 2 forecasting models (XGBoost vs LSTM)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from prepare_task2_data import load_prepared_data, prepare_task2_dataset


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducible runs across NumPy/Python/TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow is optional in environments where only tree models are used.
        pass


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE and MAE in a consistent format."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae}


def flatten_lag_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3D lag sequence input to 2D for tree-based models."""
    return X.reshape(X.shape[0], -1)


def scale_sequences_from_train(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scale sequence features using statistics from the training split only."""
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, n_features)
    val_2d = X_val.reshape(-1, n_features)
    test_2d = X_test.reshape(-1, n_features)

    train_scaled = scaler.fit_transform(train_2d).reshape(X_train.shape).astype(np.float32)
    val_scaled = scaler.transform(val_2d).reshape(X_val.shape).astype(np.float32)
    test_scaled = scaler.transform(test_2d).reshape(X_test.shape).astype(np.float32)

    return train_scaled, val_scaled, test_scaled, scaler


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train XGBoost regressor and return validation/test-ready outputs."""
    try:
        import xgboost as xgb
    except Exception as exc:
        raise ImportError(
            "XGBoost is not available. Install with: pip install xgboost"
        ) from exc

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict(X_val)
    return model, val_pred, np.array([], dtype=np.float32)


def build_lstm_model(input_shape: Tuple[int, int], learning_rate: float = 1e-3):
    """Build a compact 2-layer LSTM for multivariate monthly sequences."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except Exception as exc:
        raise ImportError(
            "TensorFlow is not available. Install with: pip install tensorflow"
        ) from exc

    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """Train LSTM with early stopping to limit overfitting."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "TensorFlow is not available. Install with: pip install tensorflow"
        ) from exc

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    val_pred = model.predict(X_val, verbose=0).ravel()
    return model, history, val_pred


def choose_better_model(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Select best model by lower test RMSE, then lower test MAE."""
    available_models = list(results.keys())
    if not available_models:
        return "No model available"
    if len(available_models) == 1:
        return available_models[0]

    ranking = sorted(
        available_models,
        key=lambda model: (
            results[model]["test"]["RMSE"],
            results[model]["test"]["MAE"],
        ),
    )
    return ranking[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 2 XGBoost and LSTM models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("task2_processed_data.npz"),
        help="Prepared data file generated by prepare_task2_data.py.",
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
        help="Regenerate prepared dataset even if --data exists.",
    )
    parser.add_argument(
        "--gap-fill",
        type=str,
        default="strict",
        choices=["strict", "ffill", "interpolate"],
        help="Missing-month/value handling during auto-preparation.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="LSTM training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="LSTM batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-xgboost", action="store_true", help="Skip XGBoost training.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("task2_model_metrics.json"),
        help="Path to save metrics JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("task2_model_predictions.csv"),
        help="Path to save validation/test predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

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
    train_mask = payload["train_mask"].astype(bool)
    val_mask = payload["val_mask"].astype(bool)
    test_mask = payload["test_mask"].astype(bool)
    target_dates = pd.to_datetime(payload["target_dates"].astype(str))
    site_ids = payload["site_ids"].astype(str)

    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "Train/validation/test split is empty for one or more sets. Check target date range."
        )

    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    X_val_raw, y_val = X_raw[val_mask], y[val_mask]
    X_test_raw, y_test = X_raw[test_mask], y[test_mask]

    X_train_scaled, X_val_scaled, X_test_scaled, _ = scale_sequences_from_train(
        X_train_raw, X_val_raw, X_test_raw
    )

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    prediction_tables: List[pd.DataFrame] = []

    if not args.skip_xgboost:
        print("\nTraining XGBoost...")
        try:
            X_train_flat = flatten_lag_sequences(X_train_raw)
            X_val_flat = flatten_lag_sequences(X_val_raw)
            X_test_flat = flatten_lag_sequences(X_test_raw)

            xgb_model, xgb_val_pred, _ = train_xgboost(
                X_train_flat, y_train, X_val_flat, y_val, seed=args.seed
            )
            xgb_test_pred = xgb_model.predict(X_test_flat)

            results["XGBoost"] = {
                "val": regression_metrics(y_val, xgb_val_pred),
                "test": regression_metrics(y_test, xgb_test_pred),
            }

            prediction_tables.append(
                pd.DataFrame(
                    {
                        "model": "XGBoost",
                        "split": "val",
                        "site_id": site_ids[val_mask],
                        "target_date": target_dates[val_mask].strftime("%Y-%m-%d"),
                        "actual": y_val,
                        "predicted": xgb_val_pred,
                    }
                )
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "model": "XGBoost",
                        "split": "test",
                        "site_id": site_ids[test_mask],
                        "target_date": target_dates[test_mask].strftime("%Y-%m-%d"),
                        "actual": y_test,
                        "predicted": xgb_test_pred,
                    }
                )
            )
        except Exception as exc:
            print(f"XGBoost training skipped: {exc}")

    if not args.skip_lstm:
        print("\nTraining LSTM...")
        try:
            lstm_model, history, lstm_val_pred = train_lstm(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            _ = history
            lstm_test_pred = lstm_model.predict(X_test_scaled, verbose=0).ravel()

            results["LSTM"] = {
                "val": regression_metrics(y_val, lstm_val_pred),
                "test": regression_metrics(y_test, lstm_test_pred),
            }

            prediction_tables.append(
                pd.DataFrame(
                    {
                        "model": "LSTM",
                        "split": "val",
                        "site_id": site_ids[val_mask],
                        "target_date": target_dates[val_mask].strftime("%Y-%m-%d"),
                        "actual": y_val,
                        "predicted": lstm_val_pred,
                    }
                )
            )
            prediction_tables.append(
                pd.DataFrame(
                    {
                        "model": "LSTM",
                        "split": "test",
                        "site_id": site_ids[test_mask],
                        "target_date": target_dates[test_mask].strftime("%Y-%m-%d"),
                        "actual": y_test,
                        "predicted": lstm_test_pred,
                    }
                )
            )
        except Exception as exc:
            print(f"LSTM training skipped: {exc}")

    print("\n" + "=" * 80)
    print("TASK 2 MODEL COMPARISON")
    print("=" * 80)

    if not results:
        raise RuntimeError("No models were trained successfully.")

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"  Validation -> RMSE: {metrics['val']['RMSE']:.4f}, MAE: {metrics['val']['MAE']:.4f}")
        print(f"  Test       -> RMSE: {metrics['test']['RMSE']:.4f}, MAE: {metrics['test']['MAE']:.4f}")

    best_model = choose_better_model(results)
    print(f"\nBest model (lower test RMSE/MAE): {best_model}")

    # Save comparison outputs.
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_model": best_model,
        "results": results,
        "assumption": "Input window t-6..t-1 predicts target at t+3 (4 months ahead of last observed point).",
    }
    args.metrics_output.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if prediction_tables:
        predictions_df = pd.concat(prediction_tables, ignore_index=True)
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(args.predictions_output, index=False)
        print(f"Saved predictions: {args.predictions_output}")

    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
