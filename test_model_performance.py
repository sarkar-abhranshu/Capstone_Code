#!/usr/bin/env python3
"""
test_model_performance.py
─────────────────────────────────────────────────────────────────────────────
Evaluates held-out test-split performance for the Bangalore Soil Fertility
Forecasting Capstone (GEE satellite data, 2019–2024, ~805 spatial sites).

  Task 1 — retrains XGBoost from scratch (same hyperparams as xgboost.ipynb)
            on csv/X_train_clean.csv, evaluates on csv/X_test_clean.csv
  Task 2 — re-scores saved predictions from task2_output/ (no Keras needed)

Metrics reported: MAE · RMSE · R² · MAPE · Willmott's d
Report saved to : test_report/test_metrics_report.csv
─────────────────────────────────────────────────────────────────────────────
Dependencies: pandas, numpy, scikit-learn, xgboost  (no TensorFlow)
"""

import os
import glob
import re
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

X_TRAIN_PATH = os.path.join("csv", "X_train_clean.csv")
Y_TRAIN_PATH = os.path.join("csv", "y_train.csv")
X_VAL_PATH   = os.path.join("csv", "X_val_clean.csv")
Y_VAL_PATH   = os.path.join("csv", "y_val.csv")
X_TEST_PATH  = os.path.join("csv", "X_test_clean.csv")
Y_TEST_PATH  = os.path.join("csv", "y_test.csv")
TASK2_GLOB   = os.path.join("task2_output", "task2_model_predictions_l*_h*.csv")
REPORT_DIR   = "test_report"
REPORT_PATH  = os.path.join(REPORT_DIR, "test_metrics_report.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """MAPE matching xgboost.ipynb formula: mean(|actual - pred| / (actual + eps)) * 100"""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def _willmott_d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    obs_mean    = np.mean(y_true)
    numerator   = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((np.abs(y_pred - obs_mean) + np.abs(y_true - obs_mean)) ** 2)
    return 1.0 if denominator == 0.0 else float(1.0 - numerator / denominator)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "Model / Config": label,
        "N":          len(y_true),
        "MAE":        round(float(mean_absolute_error(y_true, y_pred)), 6),
        "RMSE":       round(_rmse(y_true, y_pred), 6),
        "R2":         round(float(r2_score(y_true, y_pred)), 6),
        "MAPE (%)":   round(_mape(y_true, y_pred), 4),
        "Willmott_d": round(_willmott_d(y_true, y_pred), 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def _cell(val, col: str, width: int) -> str:
    if isinstance(val, float):
        s = f"{val:.4f}" if col == "MAPE (%)" else f"{val:.6f}"
        return s.rjust(width)
    if isinstance(val, int):
        return str(val).rjust(width)
    return str(val).ljust(width)


def print_section(title: str, rows: list) -> None:
    if not rows:
        print(f"\n  [no data for: {title}]\n")
        return

    cols = list(rows[0].keys())
    widths = {}
    for c in cols:
        values = [_cell(r[c], c, 0).strip() for r in rows]
        widths[c] = max(len(c), max(len(v) for v in values))
    widths[cols[0]] = max(widths[cols[0]], 48)

    divider  = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    header   = "|" + "|".join(f" {c:^{widths[c]}} " for c in cols) + "|"
    hdivider = "+" + "+".join("=" * (widths[c] + 2) for c in cols) + "+"
    title_width = sum(widths[c] + 3 for c in cols) + 1

    print()
    print("=" * title_width)
    print(f"  {title}")
    print("=" * title_width)
    print(divider)
    print(header)
    print(hdivider)
    for r in rows:
        print("|" + "|".join(f" {_cell(r[c], c, widths[c])} " for c in cols) + "|")
    print(divider)


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("  Soil Fertility Forecasting — Test Performance Report")
print("  Bangalore, Karnataka · GEE Data · 2019–2024 · ~805 Sites")
print("─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — retrain XGBoost (same hyperparams as xgboost.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Task 1] Loading data …")

def _align(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Match index lengths exactly as the notebook does."""
    if len(X) != len(y):
        idx = X.index.intersection(y.index)
        return X.loc[idx].copy(), y.loc[idx].copy()
    return X, y

X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
X_val   = pd.read_csv(X_VAL_PATH)
y_val   = pd.read_csv(Y_VAL_PATH).squeeze()
X_test  = pd.read_csv(X_TEST_PATH)
y_test  = pd.read_csv(Y_TEST_PATH).squeeze()

X_train, y_train = _align(X_train, y_train)
X_val,   y_val   = _align(X_val,   y_val)
X_test,  y_test  = _align(X_test,  y_test)

print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
print("[Task 1] Retraining XGBoost (n_estimators=200, same config as notebook) …")

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test).astype(np.float64)
y_test_np  = y_test.to_numpy(dtype=np.float64)

task1_rows = [compute_metrics(y_test_np, y_pred_xgb, "XGBoost")]
print(f"  ✓  Done  (n={len(y_test_np):,})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — re-score from saved prediction CSVs (no model reload needed)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Task 2] Scanning task2_output/ for prediction CSVs …")

pred_files = sorted(glob.glob(TASK2_GLOB))
if not pred_files:
    raise FileNotFoundError(f"No prediction CSVs matched: {TASK2_GLOB}")

_fname_re = re.compile(r"task2_model_predictions_l(\d+)_h(\d+)\.csv$")
_MODEL_ORDER = {"XGBoost": 0, "LSTM": 1, "BiLSTM+Attention": 2}

task2_raw = []

for fpath in pred_files:
    m = _fname_re.search(os.path.basename(fpath))
    if not m:
        print(f"  ⚠  Skipping unrecognised filename: {fpath}")
        continue
    lookback = int(m.group(1))
    horizon  = int(m.group(2))

    df = pd.read_csv(fpath)
    required = {"model", "split", "actual", "predicted"}
    if not required.issubset(df.columns):
        print(f"  ⚠  Missing columns in {fpath}: {required - set(df.columns)}")
        continue

    df_test = df[df["split"] == "test"].copy()
    if df_test.empty:
        print(f"  ⚠  No 'test' split rows in {fpath}")
        continue

    for model_name in sorted(df_test["model"].unique()):
        sub = df_test[df_test["model"] == model_name]
        row = compute_metrics(
            sub["actual"].to_numpy(dtype=np.float64),
            sub["predicted"].to_numpy(dtype=np.float64),
            label=model_name,
        )
        task2_raw.append((model_name, lookback, horizon, row))

task2_raw.sort(key=lambda t: (_MODEL_ORDER.get(t[0], 99), t[1], t[2]))

print(f"  ✓  {len(pred_files)} file(s) → {len(task2_raw)} model × config combination(s)")

task2_rows = []
for model_name, lookback, horizon, row in task2_raw:
    display_row = dict(row)
    display_row["Model / Config"] = f"{model_name:<22} L={lookback:>2}  H={horizon:>2}"
    task2_rows.append(display_row)

# ─────────────────────────────────────────────────────────────────────────────
# Print tables
# ─────────────────────────────────────────────────────────────────────────────

print_section(
    "TASK 1 — Static Regression (XGBoost)  |  held-out test split",
    task1_rows,
)
print_section(
    "TASK 2 — Sequential Forecasting (XGBoost / LSTM / BiLSTM+Attention)  |  test split",
    task2_rows,
)

# ─────────────────────────────────────────────────────────────────────────────
# Save to CSV
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(REPORT_DIR, exist_ok=True)

save_cols_map = {
    "Model / Config": "Model_Config",
    "N":              "N",
    "MAE":            "MAE",
    "RMSE":           "RMSE",
    "R2":             "R2",
    "MAPE (%)":       "MAPE_pct",
    "Willmott_d":     "Willmott_d",
}

all_save_rows = (
    [{"Task": "Task1", "Lookback": None, "Horizon": None, **r} for r in task1_rows]
    + [{"Task": "Task2", "Lookback": lb, "Horizon": hz, **r}
       for model_name, lb, hz, r in task2_raw]
)

out_df = pd.DataFrame(all_save_rows).rename(columns=save_cols_map)
col_order = ["Task", "Lookback", "Horizon"] + list(save_cols_map.values())
out_df = out_df[[c for c in col_order if c in out_df.columns]]
out_df.to_csv(REPORT_PATH, index=False)

print(f"\n✓  Report saved → {REPORT_PATH}")
print(f"   Rows: {len(out_df)}  |  Columns: {out_df.columns.tolist()}\n")
