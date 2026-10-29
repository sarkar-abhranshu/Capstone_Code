#!/usr/bin/env python3
"""
validate_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Sanity-checks the preprocessed data pipeline for the Bangalore Soil Fertility
Forecasting Capstone (GEE satellite data, 2019–2024, ~805 spatial sites).

Checks performed
────────────────
  1. No date overlap between static train / val / test splits
     └─ Evidence: csv/{train,val,test}_metadata.csv  +
                  task2_output/task2_model_predictions_l12_h6.csv (split col)
  2. Each X file has exactly 19 features (no columns added/dropped silently)
  3. Each X file has zero NaN values (no silent imputation holes)
  4. FertilityIndex target (y files) is fully within [0, 1] — confirms
     min-max normalisation was applied
  5. Splits are strictly chronologically ordered:
     max(train_date) < min(val_date) < min(test_date)

Prints PASS / FAIL with a short reason for each check.
─────────────────────────────────────────────────────────────────────────────
Dependencies: pandas, numpy  (no TensorFlow, no sklearn required here)
"""

import os
import sys

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_META  = os.path.join("csv", "train_metadata.csv")
VAL_META    = os.path.join("csv", "val_metadata.csv")
TEST_META   = os.path.join("csv", "test_metadata.csv")

X_TRAIN     = os.path.join("csv", "X_train_clean.csv")
X_VAL       = os.path.join("csv", "X_val_clean.csv")
X_TEST      = os.path.join("csv", "X_test_clean.csv")

Y_TRAIN     = os.path.join("csv", "y_train.csv")
Y_VAL       = os.path.join("csv", "y_val.csv")
Y_TEST      = os.path.join("csv", "y_test.csv")

TASK2_PREDS = os.path.join("task2_output", "task2_model_predictions_l12_h6.csv")

EXPECTED_FEATURES = 19

# ─────────────────────────────────────────────────────────────────────────────
# Result accumulator
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (check_label, passed, reason)

def _record(label: str, passed: bool, reason: str) -> None:
    """Store and immediately print a check result."""
    status = "PASS" if passed else "FAIL"
    colour = "\033[92m" if passed else "\033[91m"   # green / red (ANSI)
    reset  = "\033[0m"
    # Fallback for terminals without colour
    try:
        print(f"  [{colour}{status}{reset}]  {label}")
    except Exception:
        print(f"  [{status}]  {label}")
    if not passed:
        print(f"         ↳  {reason}")
    else:
        print(f"         ↳  {reason}")
    _results.append((label, passed, reason))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe CSV load with a clear error on missing file
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  Pipeline Validation — Bangalore Soil Fertility Forecasting")
print("  GEE Satellite Data · 2019–2024 · ~805 Spatial Sites")
print("=" * 68)
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — No date overlap between train / val / test
# ─────────────────────────────────────────────────────────────────────────────

print("CHECK 1: No date overlap between train / val / test")
print("─" * 68)

# 1a — Static metadata dates
try:
    train_meta = _load(TRAIN_META)
    val_meta   = _load(VAL_META)
    test_meta  = _load(TEST_META)

    train_dates = set(pd.to_datetime(train_meta["date"]).dt.normalize())
    val_dates   = set(pd.to_datetime(val_meta["date"]).dt.normalize())
    test_dates  = set(pd.to_datetime(test_meta["date"]).dt.normalize())

    tv_overlap  = train_dates & val_dates
    tt_overlap  = train_dates & test_dates
    vt_overlap  = val_dates   & test_dates

    all_clear = not (tv_overlap or tt_overlap or vt_overlap)

    if all_clear:
        reason_1a = (
            f"train {sorted(train_dates)[0].date()} – {sorted(train_dates)[-1].date()}  |  "
            f"val {sorted(val_dates)[0].date()} – {sorted(val_dates)[-1].date()}  |  "
            f"test {sorted(test_dates)[0].date()} – {sorted(test_dates)[-1].date()}  — "
            "zero pairwise date intersections"
        )
    else:
        parts = []
        if tv_overlap:
            parts.append(f"train∩val={len(tv_overlap)} dates")
        if tt_overlap:
            parts.append(f"train∩test={len(tt_overlap)} dates")
        if vt_overlap:
            parts.append(f"val∩test={len(vt_overlap)} dates")
        reason_1a = "Overlapping dates found: " + ", ".join(parts)

    _record("1a  Static splits — no date overlap (metadata CSVs)", all_clear, reason_1a)

except Exception as exc:
    _record("1a  Static splits — no date overlap (metadata CSVs)", False, str(exc))
    all_clear = False


# 1b — Task 2 prediction CSV: val and test windows must not overlap
try:
    t2 = _load(TASK2_PREDS)
    t2["target_date"] = pd.to_datetime(t2["target_date"])

    t2_val_dates  = set(t2.loc[t2["split"] == "val",  "target_date"].dt.normalize())
    t2_test_dates = set(t2.loc[t2["split"] == "test", "target_date"].dt.normalize())

    t2_overlap = t2_val_dates & t2_test_dates
    t2_ok      = len(t2_overlap) == 0

    if t2_ok:
        reason_1b = (
            f"Task-2 val target range  "
            f"{min(t2_val_dates).date()} – {max(t2_val_dates).date()}  |  "
            f"test target range  "
            f"{min(t2_test_dates).date()} – {max(t2_test_dates).date()}  — "
            "no intersection"
        )
    else:
        reason_1b = (
            f"{len(t2_overlap)} overlapping target date(s) between Task-2 "
            f"val and test splits — potential leakage in sequential windows"
        )

    _record(
        "1b  Task-2 sequential splits — no prediction-date overlap "
        f"({os.path.basename(TASK2_PREDS)})",
        t2_ok,
        reason_1b,
    )

except Exception as exc:
    _record(
        "1b  Task-2 sequential splits — no prediction-date overlap",
        False,
        str(exc),
    )

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Each X file has exactly EXPECTED_FEATURES features
# ─────────────────────────────────────────────────────────────────────────────

print("CHECK 2: X files have exactly 19 features")
print("─" * 68)

x_frames: dict[str, pd.DataFrame] = {}

for name, path in [("X_train", X_TRAIN), ("X_val", X_VAL), ("X_test", X_TEST)]:
    try:
        df = _load(path)
        x_frames[name] = df
        n_feat = df.shape[1]
        ok = n_feat == EXPECTED_FEATURES
        reason = (
            f"{n_feat} features  (shape {df.shape})  — "
            + ("correct" if ok else f"expected {EXPECTED_FEATURES}")
        )
        _record(f"2   {name} feature count", ok, reason)
    except Exception as exc:
        _record(f"2   {name} feature count", False, str(exc))

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Zero NaN values in all X files
# ─────────────────────────────────────────────────────────────────────────────

print("CHECK 3: X files have zero NaN values")
print("─" * 68)

for name, df in x_frames.items():
    nan_count = int(df.isna().sum().sum())
    ok = nan_count == 0
    reason = f"{nan_count} NaN(s) across {df.shape[0]:,} rows × {df.shape[1]} features"
    _record(f"3   {name} NaN count", ok, reason)

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — FertilityIndex (y files) fully in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

print("CHECK 4: FertilityIndex targets in [0, 1]  (min-max normalised)")
print("─" * 68)

for name, path in [("y_train", Y_TRAIN), ("y_val", Y_VAL), ("y_test", Y_TEST)]:
    try:
        ser = _load(path, header=None).squeeze("columns").astype(float)
        lo, hi = float(ser.min()), float(ser.max())
        ok = (lo >= 0.0) and (hi <= 1.0)
        reason = (
            f"min={lo:.6f}  max={hi:.6f}  n={len(ser):,}  — "
            + ("within [0, 1]" if ok else "OUT OF [0, 1] RANGE")
        )
        _record(f"4   {name} value range", ok, reason)
    except Exception as exc:
        _record(f"4   {name} value range", False, str(exc))

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Splits are strictly chronologically ordered
# ─────────────────────────────────────────────────────────────────────────────

print("CHECK 5: Chronological ordering  train < val < test")
print("─" * 68)

try:
    # Re-use already-loaded metadata if available
    if "train_dates" not in dir() or not train_dates:
        train_meta = _load(TRAIN_META)
        val_meta   = _load(VAL_META)
        test_meta  = _load(TEST_META)
        train_dates = set(pd.to_datetime(train_meta["date"]).dt.normalize())
        val_dates   = set(pd.to_datetime(val_meta["date"]).dt.normalize())
        test_dates  = set(pd.to_datetime(test_meta["date"]).dt.normalize())

    max_train = max(train_dates)
    min_val   = min(val_dates)
    max_val   = max(val_dates)
    min_test  = min(test_dates)

    train_before_val  = max_train < min_val
    val_before_test   = max_val   < min_test

    ok = train_before_val and val_before_test

    reason = (
        f"max(train)={max_train.date()}  min(val)={min_val.date()}  "
        f"max(val)={max_val.date()}  min(test)={min_test.date()}  — "
    )
    if ok:
        reason += "strictly ordered"
    else:
        parts = []
        if not train_before_val:
            parts.append(
                f"train extends past val start "
                f"({max_train.date()} >= {min_val.date()})"
            )
        if not val_before_test:
            parts.append(
                f"val extends past test start "
                f"({max_val.date()} >= {min_test.date()})"
            )
        reason += "ORDER VIOLATED — " + "; ".join(parts)

    _record("5   Chronological split ordering", ok, reason)

except Exception as exc:
    _record("5   Chronological split ordering", False, str(exc))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("  SUMMARY")
print("=" * 68)

n_pass = sum(1 for _, p, _ in _results if p)
n_fail = sum(1 for _, p, _ in _results if not p)

for label, passed, _ in _results:
    tick = "✓" if passed else "✗"
    print(f"  {tick}  {label}")

print()
print(f"  Passed: {n_pass} / {len(_results)}   Failed: {n_fail} / {len(_results)}")

if n_fail == 0:
    print("\n  ✅  All pipeline checks passed — no evidence of data leakage.")
else:
    print(f"\n  ❌  {n_fail} check(s) failed — review items marked FAIL above.")

print("=" * 68)
print()

sys.exit(0 if n_fail == 0 else 1)
