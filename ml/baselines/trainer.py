"""Baseline Model Trainer for Tabular Data (Majority Class, Logistic Regression & Gradient Boosting GBDT).

Evaluates models independently on temporal test split and outputs separate reports per model.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from ml.features.extractor import extract_tabular_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineTrainer")


def compute_metrics_with_absent_class_handling(preds: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
    """Compute accuracy fraction, macro-F1, and per-class precision/recall/F1 handling absent classes."""
    correct_count = int((preds == y_true).sum())
    total_count = len(y_true)
    acc_fraction = f"{correct_count}/{total_count}"
    acc_val = float(correct_count / total_count)

    per_class = {}
    for c in [0, 1, 2]:
        n_true = int((y_true == c).sum())
        if n_true == 0:
            per_class[f"class_{c}"] = {
                "sample_count": 0,
                "precision": "undefined — 0 samples in this split",
                "recall": "undefined — 0 samples in this split",
                "f1": "undefined — 0 samples in this split"
            }
        else:
            tp = int(((preds == c) & (y_true == c)).sum())
            fp = int(((preds == c) & (y_true != c)).sum())
            fn = int(((preds != c) & (y_true == c)).sum())

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            per_class[f"class_{c}"] = {
                "sample_count": n_true,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            }

    # Macro-F1 across active classes (0 and 1)
    active_f1s = [v["f1"] for k, v in per_class.items() if isinstance(v["f1"], float)]
    macro_f1 = float(np.mean(active_f1s)) if active_f1s else 0.0

    return {
        "test_correct_count": correct_count,
        "test_total_count": total_count,
        "accuracy_fraction": acc_fraction,
        "accuracy": acc_val,
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class,
        "predictions": preds.tolist(),
        "ground_truth": y_true.tolist()
    }


class TabularBaselineTrainer:
    """Trainer for baseline tabular machine learning models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_and_evaluate(
        self,
        train_path: str = "data/processed/train_temporal.parquet",
        test_path: str = "data/processed/test_temporal.parquet"
    ) -> Dict[str, Any]:
        """Train baseline models independently on train_path and evaluate on test_path."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing required dataset split files: {train_path} or {test_path}")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        X_train, y_train, _ = extract_tabular_features(train_df)
        X_test, y_test, _ = extract_tabular_features(test_df)

        # 1. Majority-Class Baseline (Predict most frequent class in training set)
        train_classes, counts = np.unique(y_train, return_counts=True)
        majority_class = int(train_classes[np.argmax(counts)])
        preds_majority = np.full(len(y_test), majority_class, dtype=int)
        majority_metrics = compute_metrics_with_absent_class_handling(preds_majority, y_test)
        majority_metrics["model_name"] = "MajorityClass_Baseline"
        majority_metrics["most_frequent_training_class"] = majority_class
        majority_metrics["note"] = "Predicts most frequent training class (Class 1, Medium) for all test samples."

        # 2. Logistic Regression Model
        clf_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf_lr.fit(X_train, y_train)
        preds_lr = clf_lr.predict(X_test)
        lr_metrics = compute_metrics_with_absent_class_handling(preds_lr, y_test)
        lr_metrics["model_name"] = "LogisticRegression"
        lr_metrics["beats_majority_baseline"] = bool(lr_metrics["accuracy"] > majority_metrics["accuracy"])

        # 3. Gradient Boosting Classifier (GBDT)
        clf_gb = GradientBoostingClassifier(random_state=self.random_state)
        clf_gb.fit(X_train, y_train)
        preds_gb = clf_gb.predict(X_test)
        gb_metrics = compute_metrics_with_absent_class_handling(preds_gb, y_test)
        gb_metrics["model_name"] = "GradientBoostingClassifier"
        gb_metrics["beats_majority_baseline"] = bool(gb_metrics["accuracy"] > majority_metrics["accuracy"])

        logger.info(f"MajorityClass Baseline Test Accuracy: {majority_metrics['accuracy_fraction']} ({majority_metrics['accuracy']*100:.1f}%) | Preds: {preds_majority.tolist()}")
        logger.info(f"LogisticRegression Test Accuracy: {lr_metrics['accuracy_fraction']} ({lr_metrics['accuracy']*100:.1f}%) | Preds: {preds_lr.tolist()}")
        logger.info(f"GradientBoosting Test Accuracy: {gb_metrics['accuracy_fraction']} ({gb_metrics['accuracy']*100:.1f}%) | Preds: {preds_gb.tolist()}")

        # Save separate artifacts
        os.makedirs("reports", exist_ok=True)

        with open("reports/baseline_majority_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Majority Class Baseline Report\n\n")
            f.write("```json\n" + json.dumps(majority_metrics, indent=2) + "\n```\n")

        with open("reports/baseline_logreg_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Logistic Regression Baseline Report\n\n")
            f.write("```json\n" + json.dumps(lr_metrics, indent=2) + "\n```\n")

        with open("reports/baseline_gbdt_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Gradient Boosting (GBDT) Baseline Report\n\n")
            f.write("```json\n" + json.dumps(gb_metrics, indent=2) + "\n```\n")

        # Clean legacy file if exists
        if os.path.exists("reports/baseline_xgboost_report.md"):
            try:
                os.remove("reports/baseline_xgboost_report.md")
            except Exception:
                pass

        return {
            "MajorityClass_Baseline": majority_metrics,
            "LogisticRegression": lr_metrics,
            "GradientBoosting": gb_metrics
        }


if __name__ == "__main__":
    trainer = TabularBaselineTrainer()
    trainer.train_and_evaluate()
