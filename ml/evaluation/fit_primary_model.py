"""Fit and save the primary LogisticRegression model for SciGraph AI 2.0.

Fits M1_LogisticRegression on train split, calibrates temperature on val,
saves primary_model.joblib + primary_model_meta.json to ml/gnn/checkpoints/.
Evaluates on test split and prints test metrics.

Usage:
    python ml/evaluation/fit_primary_model.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, classification_report
)

# ---- repo root on path -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features.builder import (
    build_paper_features_batch,
    FEATURE_NAMES,
    TOTAL_FEATURE_DIM,
)

CHECKPOINTS_DIR = REPO_ROOT / "ml" / "gnn" / "checkpoints"
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


# ---- helpers -----------------------------------------------------------------

def fit_temperature_nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Finds optimal temperature T minimising NLL on validation logits."""

    def nll(T: float) -> float:
        scaled = logits / max(float(T), 1e-4)
        exp_z = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        prob_true = probs[np.arange(len(y_true)), y_true]
        return float(-np.mean(np.log(np.clip(prob_true, 1e-12, 1.0))))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Applies temperature scaling and softmax."""
    scaled = logits / max(float(T), 1e-4)
    exp_z = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(np.float32)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            ece += np.abs(
                np.mean(confidences[mask]) - np.mean(accuracies[mask])
            ) * np.mean(mask)
    return float(ece)


# ---- main procedure ----------------------------------------------------------

def fit_and_save_primary_model() -> dict:
    """End-to-end: load data -> build features -> fit -> calibrate -> save."""
    print("=" * 70)
    print("SciGraph AI 2.0 — Fitting Primary Model (M1_LogisticRegression)")
    print("=" * 70)
    t0 = time.time()

    # 1. Load data splits
    print("\n[1/6] Loading parquet splits ...")
    df_train = pd.read_parquet(DATA_DIR / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # 2. Build features
    print("\n[2/6] Building 47-dimensional features (batch mode) ...")
    X_train_raw, feat_names = build_paper_features_batch(df_train)
    X_val_raw, _ = build_paper_features_batch(df_val)
    X_test_raw, _ = build_paper_features_batch(df_test)
    feature_dim = X_train_raw.shape[1]
    print(f"  Feature dims: {feature_dim}  |  Names: {feat_names[:5]} ...")

    y_train = df_train["impact_class"].values.astype(int)
    y_val = df_val["impact_class"].values.astype(int)
    y_test = df_test["impact_class"].values.astype(int)

    # 3. Fit StandardScaler on TRAIN only
    print("\n[3/6] Fitting StandardScaler on train split only ...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 4. Fit LogisticRegression (C tuned on VAL)
    print("\n[4/6] Fitting LogisticRegression (C tuned on val) ...")
    best_c = 1.0
    best_val_f1 = -1.0
    for c_cand in [0.01, 0.1, 1.0, 10.0]:
        clf_probe = LogisticRegression(
            C=c_cand,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            multi_class="auto",
        )
        clf_probe.fit(X_train, y_train)
        vf1 = f1_score(y_val, clf_probe.predict(X_val), average="macro")
        print(f"  C={c_cand:6.3f} -> val_macro_f1={vf1:.4f}")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_c = c_cand

    print(f"  Best C={best_c}  val_macro_f1={best_val_f1:.4f}")
    lr = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        multi_class="auto",
    )
    lr.fit(X_train, y_train)

    # Val metrics
    val_logits = lr.decision_function(X_val)
    val_probs_raw = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
    val_probs_raw /= val_probs_raw.sum(axis=1, keepdims=True)
    val_macro_f1 = float(f1_score(y_val, np.argmax(val_probs_raw, axis=1), average="macro"))
    val_auroc = float(roc_auc_score(y_val, val_probs_raw, multi_class="ovr"))
    print(f"  Val macro_f1={val_macro_f1:.4f}  auroc={val_auroc:.4f}")

    # 5. Temperature calibration on VAL
    print("\n[5/6] Calibrating temperature on val logits ...")
    T = fit_temperature_nll(val_logits, y_val)
    cal_val_probs = apply_temperature(val_logits, T)
    val_ece = compute_ece(cal_val_probs, y_val)
    print(f"  Optimal T={T:.4f}  calibrated_val_ECE={val_ece:.4f}")

    # 6. Test evaluation
    print("\n[6/6] Evaluating on test split ...")
    test_logits = lr.decision_function(X_test)
    test_probs = apply_temperature(test_logits, T)
    test_preds = np.argmax(test_probs, axis=1)
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_micro_f1 = float(accuracy_score(y_test, test_preds))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    test_auroc = float(roc_auc_score(y_test, test_probs, multi_class="ovr"))
    test_ece = compute_ece(test_probs, y_test)

    print(f"\n  Test Results (M1_LogisticRegression):")
    print(f"    macro_f1   = {test_macro_f1:.4f}")
    print(f"    micro_f1   = {test_micro_f1:.4f}")
    print(f"    weighted_f1= {test_weighted_f1:.4f}")
    print(f"    auroc      = {test_auroc:.4f}")
    print(f"    ece        = {test_ece:.4f}")
    print(f"\n  Per-class report (test):")
    print(classification_report(y_test, test_preds, target_names=["Low", "Medium", "High"]))

    # Save primary_model.joblib
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "model": lr,
        "scaler": scaler,
        "temperature": T,
        "feature_names": feat_names,
        "feature_dims": feature_dim,
        "best_c": best_c,
        "fitted_at": timestamp,
        "val_macro_f1": round(val_macro_f1, 4),
        "val_auroc": round(val_auroc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_auroc": round(test_auroc, 4),
    }
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINTS_DIR / "primary_model.joblib"
    joblib.dump(payload, model_path)
    print(f"\n  Saved: {model_path}")

    # Save primary_model_meta.json
    meta = {
        "model_name": "M1_LogisticRegression",
        "val_macro_f1": round(val_macro_f1, 4),
        "val_auroc": round(val_auroc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_micro_f1": round(test_micro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "test_auroc": round(test_auroc, 4),
        "test_ece": round(test_ece, 4),
        "temperature": round(T, 4),
        "best_c": best_c,
        "fitted_at": timestamp,
        "feature_dims": feature_dim,
        "feature_names": feat_names,
        "description": (
            "Logistic regression with temperature-calibrated probabilities. "
            "Best validation Macro-F1 across M0-M5 benchmark. "
            "The GNN (M5_HeteroGraphSAGE) is available as experimental but "
            "does not outperform logistic regression on the test set."
        ),
        "training_split": "train (pub_year <= 2016, 10,500 papers)",
        "val_split": "val (pub_year = 2017, 1,500 papers)",
        "test_split": "test (pub_year 2018-2020, 4,500 papers)",
        "cutoff_offset": "T_cutoff = pub_year + 2",
        "duration_seconds": round(time.time() - t0, 2),
    }
    meta_path = CHECKPOINTS_DIR / "primary_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")
    print(f"\n  Total time: {meta['duration_seconds']}s")
    print("=" * 70)
    return meta


if __name__ == "__main__":
    fit_and_save_primary_model()
