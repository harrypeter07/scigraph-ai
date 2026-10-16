"""Export held-out predictions from the primary served model (M1_LogisticRegression).

Generates:
  reports/heldout_predictions.parquet — per-paper predictions for val+test
  reports/heldout_predictions_summary.json — evaluation summary with lift, 
      spearman, confusion matrix, etc.

Usage:
    python ml/evaluation/export_heldout.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

# ---- repo root on path -------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features.builder import build_paper_features_batch, FEATURE_NAMES
from ml.evaluation.fit_primary_model import (
    fit_and_save_primary_model,
    apply_temperature,
    compute_ece,
)

CHECKPOINTS_DIR = REPO_ROOT / "ml" / "gnn" / "checkpoints"
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


# ---- helpers -----------------------------------------------------------------

def compute_empirical_percentile(
    citations_at_cutoff: float,
    field_name: str,
    pub_year: int,
    reference_df: pd.DataFrame,
) -> float:
    """Rank a paper's citations_at_cutoff against its cohort in train+val.

    Returns percentile 0-100. If cohort is empty, returns 50.0.
    """
    cohort = reference_df[
        (reference_df["primary_field_name"] == field_name)
        & (reference_df["publication_year"] == pub_year)
    ]["citations_at_cutoff"].dropna().values

    if len(cohort) == 0:
        # Fallback: same field across all years
        cohort = reference_df[
            reference_df["primary_field_name"] == field_name
        ]["citations_at_cutoff"].dropna().values

    if len(cohort) == 0:
        # Ultimate fallback: global
        cohort = reference_df["citations_at_cutoff"].dropna().values

    if len(cohort) == 0:
        return 50.0

    # fraction of cohort papers <= this paper's citations
    rank = float(np.mean(cohort <= citations_at_cutoff)) * 100.0
    return round(rank, 2)


def load_or_fit_primary_model() -> dict:
    """Load primary_model.joblib; fit it if missing."""
    model_path = CHECKPOINTS_DIR / "primary_model.joblib"
    if not model_path.exists():
        print("[INFO] primary_model.joblib not found — running fit_primary_model ...")
        fit_and_save_primary_model()
    payload = joblib.load(model_path)
    return payload


# ---- main procedure ----------------------------------------------------------

def export_heldout_predictions() -> dict:
    """Build predictions for val+test and write output files."""
    print("=" * 70)
    print("SciGraph AI 2.0 — Exporting Held-Out Predictions")
    print("=" * 70)

    # 1. Load primary model
    print("\n[1/5] Loading primary model ...")
    payload = load_or_fit_primary_model()
    lr = payload["model"]
    scaler = payload["scaler"]
    T = payload["temperature"]
    print(f"  model={payload.get('model_name', 'M1_LogisticRegression')}  "
          f"T={T:.4f}  fitted_at={payload.get('fitted_at', 'unknown')}")

    # 2. Load data
    print("\n[2/5] Loading parquet splits ...")
    df_train = pd.read_parquet(DATA_DIR / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")
    print(f"  train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}")

    # Reference corpus for empirical percentile (train+val)
    df_reference = pd.concat([df_train, df_val], ignore_index=True)
    train_work_ids = set(df_train["work_id"].astype(str))
    val_work_ids = set(df_val["work_id"].astype(str))
    test_work_ids = set(df_test["work_id"].astype(str))

    # 3. Build features for val+test
    print("\n[3/5] Building features ...")
    X_val_raw, _ = build_paper_features_batch(df_val)
    X_test_raw, _ = build_paper_features_batch(df_test)

    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Get calibrated probabilities
    val_logits = lr.decision_function(X_val)
    test_logits = lr.decision_function(X_test)
    val_probs = apply_temperature(val_logits, T)
    test_probs = apply_temperature(test_logits, T)
    val_preds = np.argmax(val_probs, axis=1)
    test_preds = np.argmax(test_probs, axis=1)

    # 4. Build output DataFrame
    print("\n[4/5] Computing empirical cohort percentiles ...")

    def build_output_df(df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray, split: str) -> pd.DataFrame:
        rows = []
        for i, row in enumerate(df.itertuples(index=False)):
            work_id = str(row.work_id)
            cit = float(row.citations_at_cutoff) if pd.notnull(row.citations_at_cutoff) else 0.0
            field = str(row.primary_field_name) if pd.notnull(row.primary_field_name) else ""
            pub_year = int(row.publication_year) if pd.notnull(row.publication_year) else 0

            actual_pct = compute_empirical_percentile(cit, field, pub_year, df_reference)

            rows.append({
                "work_id": work_id,
                "title": str(row.title) if pd.notnull(row.title) else "",
                "primary_field_name": field,
                "pub_year": pub_year,
                "split": split,
                "citations_at_cutoff": cit,
                "pred_class": int(preds[i]),
                "p_low": round(float(probs[i, 0]), 5),
                "p_med": round(float(probs[i, 1]), 5),
                "p_high": round(float(probs[i, 2]), 5),
                "actual_window": float(row.future_citations_5y) if pd.notnull(row.future_citations_5y) else None,
                "actual_class": int(row.impact_class) if pd.notnull(row.impact_class) else None,
                "actual_percentile": actual_pct,
            })
        return pd.DataFrame(rows)

    df_val_out = build_output_df(df_val, val_probs, val_preds, "val")
    df_test_out = build_output_df(df_test, test_probs, test_preds, "test")
    df_out = pd.concat([df_val_out, df_test_out], ignore_index=True)

    # Save parquet
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = REPORTS_DIR / "heldout_predictions.parquet"
    df_out.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}  ({len(df_out):,} rows)")

    # 5. Compute summary metrics on TEST split
    print("\n[5/5] Computing summary metrics on test split ...")
    y_test = df_test["impact_class"].values.astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds).tolist()

    # Per-class report
    cr = classification_report(
        y_test, test_preds,
        target_names=["Low(<50th)", "Medium(50-90th)", "High(>=90th)"],
        output_dict=True,
    )

    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_accuracy = float(accuracy_score(y_test, test_preds))
    test_auroc = float(roc_auc_score(y_test, test_probs, multi_class="ovr"))
    test_ece = compute_ece(test_probs, y_test)

    # Lift@top10%: of papers predicted High, what % were truly High?
    # "Truly High" = actual_class == 2 (≥90th pct cohort)
    predicted_high_mask = test_preds == 2
    n_predicted_high = int(np.sum(predicted_high_mask))
    if n_predicted_high > 0:
        n_truly_high_of_predicted = int(np.sum(y_test[predicted_high_mask] == 2))
        lift_top10_precision = float(n_truly_high_of_predicted / n_predicted_high)
    else:
        lift_top10_precision = 0.0

    # True prevalence of High class
    true_high_rate = float(np.mean(y_test == 2))
    lift = float(lift_top10_precision / true_high_rate) if true_high_rate > 0 else 0.0
    baseline_lift = float(true_high_rate)  # random guessing = prevalence

    # Spearman correlation between pred_class and actual_class
    spearman_rho, spearman_p = spearmanr(test_preds, y_test)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "M1_LogisticRegression",
        "split_sizes": {
            "val": len(df_val),
            "test": len(df_test),
        },
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "macro_f1": round(test_macro_f1, 4),
            "auroc_ovr": round(test_auroc, 4),
            "ece": round(test_ece, 4),
        },
        "test_confusion_matrix": {
            "labels": ["Low(<50th)", "Medium(50-90th)", "High(>=90th)"],
            "matrix": cm,
        },
        "test_per_class": {
            cls: {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
            for cls, metrics in cr.items()
            if cls not in ("accuracy", "macro avg", "weighted avg")
        },
        "test_macro_avg": {k: round(v, 4) for k, v in cr.get("macro avg", {}).items() if isinstance(v, float)},
        "test_weighted_avg": {k: round(v, 4) for k, v in cr.get("weighted avg", {}).items() if isinstance(v, float)},
        "lift_analysis": {
            "n_predicted_high": n_predicted_high,
            "n_truly_high_of_predicted": n_truly_high_of_predicted if n_predicted_high > 0 else 0,
            "lift_precision_at_predicted_high": round(lift_top10_precision, 4),
            "true_high_rate_in_test": round(true_high_rate, 4),
            "lift_ratio": round(lift, 4),
            "baseline_lift": round(baseline_lift, 4),
            "description": (
                "lift_precision_at_predicted_high = precision of High predictions. "
                "lift_ratio = lift_precision / true_high_rate. "
                "baseline_lift = true_high_rate (random guessing at correct base rate)."
            ),
        },
        "spearman_correlation": {
            "rho": round(float(spearman_rho), 4),
            "p_value": round(float(spearman_p), 6),
            "interpretation": "Spearman r between pred_class (0/1/2) and actual_class (0/1/2)",
        },
        "temperature": round(T, 4),
    }

    summary_path = REPORTS_DIR / "heldout_predictions_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Print key numbers
    print("\n" + "=" * 70)
    print("HELD-OUT PREDICTION SUMMARY (TEST SPLIT):")
    print(f"  accuracy   = {test_accuracy:.4f}")
    print(f"  macro_f1   = {test_macro_f1:.4f}")
    print(f"  auroc      = {test_auroc:.4f}")
    print(f"  ece        = {test_ece:.4f}")
    print(f"  lift@PredHigh precision = {lift_top10_precision:.4f}")
    print(f"  lift ratio (vs base {baseline_lift:.4f}) = {lift:.4f}x")
    print(f"  spearman_rho = {spearman_rho:.4f}  (p={spearman_p:.4e})")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    export_heldout_predictions()
