"""Train and compare Task 2 forecasting models for Horizon 9 months.

Horizon-specific defaults:
  - Lookback : 18 months
  - BiLSTM   : 96 → 48 units (bidirectional)
  - Dense    : 32 → 1 (two-layer head)
  - Epochs   : 90
  - Batch    : 48
  - ES pat.  : 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from prepare_task2_data import load_prepared_data


# ---------------------------------------------------------------------------
# Horizon-specific constants
# ---------------------------------------------------------------------------
HORIZON = 9
DEFAULT_LOOKBACK = 18
DEFAULT_LSTM_UNITS_1 = 96
DEFAULT_LSTM_UNITS_2 = 48
DEFAULT_DENSE_UNITS = [32]       # two-layer head: Dense(32) → Dense(1)
DEFAULT_DROPOUT = 0.2
DEFAULT_EPOCHS = 90
DEFAULT_BATCH = 48
DEFAULT_PATIENCE = 15


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def willmott_d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_obs = np.mean(y_true)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum(
        (np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2
    )
    return 1.0 if denominator == 0.0 else float(1.0 - numerator / denominator)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "d":    willmott_d(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def flatten_lag_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten 3-D sequence array to 2-D for XGBoost."""
    return X.reshape(X.shape[0], -1)


def scale_sequences_from_train(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train only; transform val and test."""
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    train_2d = X_train.reshape(-1, n_features)
    val_2d   = X_val.reshape(-1, n_features)
    test_2d  = X_test.reshape(-1, n_features)

    train_scaled = scaler.fit_transform(train_2d).reshape(X_train.shape).astype(np.float32)
    val_scaled   = scaler.transform(val_2d).reshape(X_val.shape).astype(np.float32)
    test_scaled  = scaler.transform(test_2d).reshape(X_test.shape).astype(np.float32)

    return train_scaled, val_scaled, test_scaled, scaler


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[object, np.ndarray]:
    try:
        import xgboost as xgb
    except Exception as exc:
        raise ImportError("Install XGBoost: pip install xgboost") from exc

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
    return model, model.predict(X_val)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def build_lstm_model(input_shape: Tuple[int, int], learning_rate: float = 1e-3):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except Exception as exc:
        raise ImportError("Install TensorFlow: pip install tensorflow") from exc

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
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
    patience: int,
):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("Install TensorFlow: pip install tensorflow") from exc

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    val_pred = model.predict(X_val, verbose=0).ravel()
    return model, history, val_pred


# ---------------------------------------------------------------------------
# BiLSTM + Attention  (H9-tuned defaults)
# ---------------------------------------------------------------------------

def _compute_trend(tensor):
    """OLS slope per feature over the time dimension."""
    import tensorflow as tf
    timesteps = tf.cast(tf.shape(tensor)[1], tf.float32)
    t = tf.range(timesteps, dtype=tf.float32)
    t_centered = t - tf.reduce_mean(t)
    t_centered = tf.reshape(t_centered, (1, -1, 1))
    x_centered = tensor - tf.reduce_mean(tensor, axis=1, keepdims=True)
    numerator   = tf.reduce_sum(t_centered * x_centered, axis=1)
    denominator = tf.reduce_sum(t_centered ** 2)
    return numerator / denominator


def build_bilstm_attention_model(
    input_shape: Tuple[int, int],
    lstm_units_1: int = DEFAULT_LSTM_UNITS_1,
    lstm_units_2: int = DEFAULT_LSTM_UNITS_2,
    dense_units:  List[int] = None,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = 1e-3,
):
    """BiLSTM with temporal attention + rolling statistics context.

    dense_units: list of hidden sizes before the final Dense(1).
                 E.g. [32] → Dense(32, relu) → Dense(1).
    """
    if dense_units is None:
        dense_units = DEFAULT_DENSE_UNITS

    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except Exception as exc:
        raise ImportError("Install TensorFlow: pip install tensorflow") from exc

    inputs = layers.Input(shape=input_shape)

    # Bidirectional encoder
    x = layers.Bidirectional(layers.LSTM(lstm_units_1, return_sequences=True))(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units_2, return_sequences=True))(x)

    # Temporal attention
    query   = layers.Lambda(lambda t: t[:, -1:, :], name="attention_query")(x)
    context = layers.Attention(use_scale=True, name="temporal_attention")([query, x])
    context = layers.Flatten(name="attention_context_flatten")(context)
    query   = layers.Flatten(name="attention_query_flatten")(query)
    attn_out = layers.Concatenate(name="attention_concatenate")([query, context])

    # Rolling statistics context
    mean_feat  = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1),      name="rolling_mean")(inputs)
    std_feat   = layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1),  name="rolling_std")(inputs)
    trend_feat = layers.Lambda(_compute_trend,                            name="rolling_trend")(inputs)

    x = layers.Concatenate(name="context_concatenate")([attn_out, mean_feat, std_feat, trend_feat])
    x = layers.Dropout(dropout)(x)

    # Two-layer dense head
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="bilstm_attention_h9")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_bilstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("Install TensorFlow: pip install tensorflow") from exc

    model = build_bilstm_attention_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    val_pred = model.predict(X_val, verbose=0).ravel()
    return model, history, val_pred


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def choose_better_model(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    available = list(results.keys())
    if not available:
        return "No model available"
    return min(available, key=lambda m: (results[m]["test"]["RMSE"], results[m]["test"]["MAE"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Task 2 models for Horizon 9 months."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("task2_processed/task2_processed_data_l18_h9.npz"),
        help="Pre-built .npz produced by prepare_task2_data.py (lookback=18, horizon=9).",
    )
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--patience",   type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-lstm",    action="store_true")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("task2_output/task2_model_metrics_l18_h9.json"),
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("task2_output/task2_model_predictions_l18_h9.csv"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("bilstm_attention_results/bilstm_attention_model_h9.keras"),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print(f"[H9] Loading data from: {args.data}")
    payload = load_prepared_data(args.data)

    X_raw        = payload["X_raw"].astype(np.float32)
    y            = payload["y"].astype(np.float32)
    train_mask   = payload["train_mask"].astype(bool)
    val_mask     = payload["val_mask"].astype(bool)
    test_mask    = payload["test_mask"].astype(bool)
    target_dates = pd.to_datetime(payload["target_dates"].astype(str))
    site_ids     = payload["site_ids"].astype(str)

    for split, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            raise ValueError(f"[H9] {split} split is empty — check .npz generation.")

    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    X_val_raw,   y_val   = X_raw[val_mask],   y[val_mask]
    X_test_raw,  y_test  = X_raw[test_mask],  y[test_mask]

    X_train, X_val, X_test, _ = scale_sequences_from_train(
        X_train_raw, X_val_raw, X_test_raw
    )

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    prediction_tables: List[pd.DataFrame] = []

    def _record(model_name: str, split: str, mask: np.ndarray,
                actual: np.ndarray, predicted: np.ndarray) -> None:
        prediction_tables.append(pd.DataFrame({
            "model":       model_name,
            "split":       split,
            "site_id":     site_ids[mask],
            "target_date": target_dates[mask].strftime("%Y-%m-%d"),
            "actual":      actual,
            "predicted":   predicted,
        }))

    # ---- XGBoost -----------------------------------------------------------
    if not args.skip_xgboost:
        print("\nTraining XGBoost...")
        try:
            xgb_model, xgb_val_pred = train_xgboost(
                flatten_lag_sequences(X_train_raw), y_train,
                flatten_lag_sequences(X_val_raw),   y_val,
                seed=args.seed,
            )
            xgb_test_pred = xgb_model.predict(flatten_lag_sequences(X_test_raw))
            results["XGBoost"] = {
                "val":  regression_metrics(y_val,  xgb_val_pred),
                "test": regression_metrics(y_test, xgb_test_pred),
            }
            _record("XGBoost", "val",  val_mask,  y_val,  xgb_val_pred)
            _record("XGBoost", "test", test_mask, y_test, xgb_test_pred)
        except Exception as exc:
            print(f"XGBoost skipped: {exc}")

    # ---- LSTM --------------------------------------------------------------
    if not args.skip_lstm:
        print("\nTraining LSTM...")
        try:
            lstm_model, _, lstm_val_pred = train_lstm(
                X_train, y_train, X_val, y_val,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            )
            lstm_test_pred = lstm_model.predict(X_test, verbose=0).ravel()
            results["LSTM"] = {
                "val":  regression_metrics(y_val,  lstm_val_pred),
                "test": regression_metrics(y_test, lstm_test_pred),
            }
            _record("LSTM", "val",  val_mask,  y_val,  lstm_val_pred)
            _record("LSTM", "test", test_mask, y_test, lstm_test_pred)
        except Exception as exc:
            print(f"LSTM skipped: {exc}")

    # ---- BiLSTM + Attention ------------------------------------------------
    print("\nTraining BiLSTM+Attention (H9: 96→48 units, dense [32])...")
    try:
        bilstm_model, _, bilstm_val_pred = train_bilstm(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        )
        bilstm_test_pred = bilstm_model.predict(X_test, verbose=0).ravel()
        results["BiLSTM+Attention"] = {
            "val":  regression_metrics(y_val,  bilstm_val_pred),
            "test": regression_metrics(y_test, bilstm_test_pred),
        }
        _record("BiLSTM+Attention", "val",  val_mask,  y_val,  bilstm_val_pred)
        _record("BiLSTM+Attention", "test", test_mask, y_test, bilstm_test_pred)
    except Exception as exc:
        print(f"BiLSTM skipped: {exc}")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 2 MODEL COMPARISON  [Horizon = 9 months | Lookback = 18 months]")
    print("=" * 80)

    if not results:
        raise RuntimeError("No models were trained successfully.")

    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        print(f"  Validation -> RMSE: {metrics['val']['RMSE']:.4f}, MAE: {metrics['val']['MAE']:.4f}, "
              f"R2: {metrics['val']['R2']:.4f}, d: {metrics['val']['d']:.4f}")
        print(f"  Test       -> RMSE: {metrics['test']['RMSE']:.4f}, MAE: {metrics['test']['MAE']:.4f}, "
              f"R2: {metrics['test']['R2']:.4f}, d: {metrics['test']['d']:.4f}")

    best_model = choose_better_model(results)
    print(f"\nBest model (lower test RMSE/MAE): {best_model}")

    # ---- Save outputs ------------------------------------------------------
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps({
            "horizon_months": HORIZON,
            "lookback_months": DEFAULT_LOOKBACK,
            "best_model": best_model,
            "results": results,
            "config": {
                "lstm_units_1": DEFAULT_LSTM_UNITS_1,
                "lstm_units_2": DEFAULT_LSTM_UNITS_2,
                "dense_units":  DEFAULT_DENSE_UNITS,
                "dropout":      DEFAULT_DROPOUT,
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "patience":     args.patience,
            },
        }, indent=2),
        encoding="utf-8",
    )

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
