"""Train and compare Task 2 forecasting models (XGBoost vs LSTM)."""

from __future__ import annotations

import argparse
import json
import os
import random
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from prepare_task2_data import load_prepared_data, prepare_task2_dataset


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducible runs across NumPy/Python/TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except Exception:
        pass


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics in a consistent format."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


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
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
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


def build_bilstm_model(input_shape: Tuple[int, int], learning_rate: float = 1e-3):
    """Build an attention-augmented bidirectional LSTM for monthly sequences."""
    return build_bilstm_attention_model(input_shape=input_shape, learning_rate=learning_rate)


def build_bilstm_attention_model(
    input_shape: Tuple[int, int],
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """Build a BiLSTM with temporal attention over the hidden sequence."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except Exception as exc:
        raise ImportError(
            "TensorFlow is not available. Install with: pip install tensorflow"
        ) from exc

    inputs = layers.Input(shape=input_shape)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True)
    )(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_2, return_sequences=True)
    )(x)

    query = layers.Lambda(lambda tensor: tensor[:, -1:, :], name="attention_query")(x)
    context = layers.Attention(use_scale=True, name="temporal_attention")([query, x])
    context = layers.Flatten(name="attention_context_flatten")(context)
    query = layers.Flatten(name="attention_query_flatten")(query)
    x = layers.Concatenate(name="attention_concatenate")([query, context])
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bilstm_attention")
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


def train_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """Train BiLSTM with early stopping to limit overfitting."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "TensorFlow is not available. Install with: pip install tensorflow"
        ) from exc

    model = build_bilstm_model((X_train.shape[1], X_train.shape[2]))
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


def tune_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
):
    """Grid-search a small attention-BiLSTM space and return the best model by val RMSE."""
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError(
            "TensorFlow is not available. Install with: pip install tensorflow"
        ) from exc

    search_space = {
        "lstm_units_1": [32],
        "lstm_units_2": [16],
        "dropout": [0.1],
        "learning_rate": [1e-3],
        "batch_size": [64],
    }

    candidate_rows: List[Dict[str, float]] = []
    best_result: Dict[str, object] | None = None

    for lstm_units_1, lstm_units_2, dropout, learning_rate, batch_size_candidate in product(
        search_space["lstm_units_1"],
        search_space["lstm_units_2"],
        search_space["dropout"],
        search_space["learning_rate"],
        search_space["batch_size"],
    ):
        tf.keras.backend.clear_session()
        model = build_bilstm_attention_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units_1=lstm_units_1,
            lstm_units_2=lstm_units_2,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size_candidate,
            callbacks=callbacks,
            verbose=0,
        )
        val_pred = model.predict(X_val, verbose=0).ravel()
        val_metrics = regression_metrics(y_val, val_pred)
        print(
            "Attention BiLSTM candidate "
            f"u1={lstm_units_1}, u2={lstm_units_2}, dropout={dropout}, lr={learning_rate}, "
            f"batch_size={batch_size_candidate} -> RMSE={val_metrics['RMSE']:.4f}, MAE={val_metrics['MAE']:.4f}, R2={val_metrics['R2']:.4f}"
        )
        candidate_row = {
            "lstm_units_1": float(lstm_units_1),
            "lstm_units_2": float(lstm_units_2),
            "dropout": float(dropout),
            "learning_rate": float(learning_rate),
            "batch_size": float(batch_size_candidate),
            "epochs_ran": float(len(history.history["loss"])),
            **val_metrics,
        }
        candidate_rows.append(candidate_row)

        if best_result is None or val_metrics["RMSE"] < best_result["val_metrics"]["RMSE"]:
            best_result = {
                "model": model,
                "history": history,
                "val_pred": val_pred,
                "val_metrics": val_metrics,
                "config": {
                    "lstm_units_1": lstm_units_1,
                    "lstm_units_2": lstm_units_2,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size_candidate,
                },
            }

    if best_result is None:
        raise RuntimeError("Attention BiLSTM tuning failed to evaluate any candidates.")

    return best_result, pd.DataFrame(candidate_rows).sort_values(["RMSE", "MAE"])


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
        "--tune-bilstm",
        action="store_true",
        help="Run a small attention-BiLSTM hyperparameter search and use the best candidate.",
    )
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
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("bilstm_attention_results") / "bilstm_attention_model.keras",
        help="Path to save the trained attention-BiLSTM model.",
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

    print("\nTraining BiLSTM+Attention...")
    try:
        bilstm_tuning_table = None
        if args.tune_bilstm:
            best_bilstm, bilstm_tuning_table = tune_bilstm(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            bilstm_model = best_bilstm["model"]
            bilstm_history = best_bilstm["history"]
            bilstm_val_pred = best_bilstm["val_pred"]
            bilstm_config = best_bilstm["config"]
        else:
            bilstm_model, bilstm_history, bilstm_val_pred = train_bilstm(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            bilstm_config = {
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "batch_size": args.batch_size,
            }

        _ = bilstm_history
        bilstm_test_pred = bilstm_model.predict(X_test_scaled, verbose=0).ravel()

        results["BiLSTM+Attention"] = {
            "val": regression_metrics(y_val, bilstm_val_pred),
            "test": regression_metrics(y_test, bilstm_test_pred),
        }

        prediction_tables.append(
            pd.DataFrame(
                {
                    "model": "BiLSTM+Attention",
                    "split": "val",
                    "site_id": site_ids[val_mask],
                    "target_date": target_dates[val_mask].strftime("%Y-%m-%d"),
                    "actual": y_val,
                    "predicted": bilstm_val_pred,
                }
            )
        )
        prediction_tables.append(
            pd.DataFrame(
                {
                    "model": "BiLSTM+Attention",
                    "split": "test",
                    "site_id": site_ids[test_mask],
                    "target_date": target_dates[test_mask].strftime("%Y-%m-%d"),
                    "actual": y_test,
                    "predicted": bilstm_test_pred,
                }
            )
        )
    except Exception as exc:
        print(f"BiLSTM training skipped: {exc}")

    print("\n" + "=" * 80)
    print("TASK 2 MODEL COMPARISON")
    print("=" * 80)

    if not results:
        raise RuntimeError("No models were trained successfully.")

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(
            f"  Validation -> RMSE: {metrics['val']['RMSE']:.4f}, MAE: {metrics['val']['MAE']:.4f}, R2: {metrics['val']['R2']:.4f}"
        )
        print(
            f"  Test       -> RMSE: {metrics['test']['RMSE']:.4f}, MAE: {metrics['test']['MAE']:.4f}, R2: {metrics['test']['R2']:.4f}"
        )

    best_model = choose_better_model(results)
    print(f"\nBest model (lower test RMSE/MAE): {best_model}")

    # Save comparison outputs.
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "best_model": best_model,
        "results": results,
        "assumption": "Input window t-6..t-1 predicts target at t+3 (4 months ahead of last observed point).",
    }
    if args.tune_bilstm:
        metrics_payload["bilstm_best_config"] = bilstm_config
        metrics_payload["bilstm_tuning_table"] = (
            bilstm_tuning_table.to_dict(orient="records") if bilstm_tuning_table is not None else []
        )
    args.metrics_output.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if prediction_tables:
        predictions_df = pd.concat(prediction_tables, ignore_index=True)
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(args.predictions_output, index=False)
        print(f"Saved predictions: {args.predictions_output}")

    if "bilstm_model" in locals():
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        bilstm_model.save(args.model_output)
        print(f"Saved model: {args.model_output}")

    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
