"""Baseline Model Trainer for Tabular Data (Logistic Regression & XGBoost/GBDT).

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


class TabularBaselineTrainer:
    """Trainer for baseline tabular machine learning models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_and_evaluate(
        self,
        train_path: str = "data/processed/train_temporal.parquet",
        test_path: str = "data/processed/test_temporal.parquet"
    ) -> Dict[str, Any]:
        """Train Logistic Regression and Gradient Boosting models independently on train_path and evaluate on test_path."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing required dataset split files: {train_path} or {test_path}")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        X_train, y_train, _ = extract_tabular_features(train_df)
        X_test, y_test, _ = extract_tabular_features(test_df)

        # 1. Logistic Regression Model
        clf_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf_lr.fit(X_train, y_train)
        preds_lr = clf_lr.predict(X_test)
        correct_lr = int((preds_lr == y_test).sum())
        acc_lr_fraction = f"{correct_lr}/{len(y_test)}"

        lr_metrics = {
            "model_name": "LogisticRegression",
            "test_correct_count": correct_lr,
            "test_total_count": len(y_test),
            "accuracy_fraction": acc_lr_fraction,
            "accuracy": float(correct_lr / len(y_test)),
            "predictions": preds_lr.tolist(),
            "ground_truth": y_test.tolist()
        }

        # 2. Gradient Boosting Classifier (XGBoost fallback)
        clf_gb = GradientBoostingClassifier(random_state=self.random_state)
        clf_gb.fit(X_train, y_train)
        preds_gb = clf_gb.predict(X_test)
        correct_gb = int((preds_gb == y_test).sum())
        acc_gb_fraction = f"{correct_gb}/{len(y_test)}"

        gb_metrics = {
            "model_name": "GradientBoostingClassifier",
            "test_correct_count": correct_gb,
            "test_total_count": len(y_test),
            "accuracy_fraction": acc_gb_fraction,
            "accuracy": float(correct_gb / len(y_test)),
            "predictions": preds_gb.tolist(),
            "ground_truth": y_test.tolist()
        }

        logger.info(f"LogisticRegression Test Accuracy: {acc_lr_fraction} ({lr_metrics['accuracy']*100:.1f}%) | Preds: {preds_lr.tolist()}")
        logger.info(f"GradientBoosting Test Accuracy: {acc_gb_fraction} ({gb_metrics['accuracy']*100:.1f}%) | Preds: {preds_gb.tolist()}")

        # Save separate artifacts
        os.makedirs("reports", exist_ok=True)

        with open("reports/baseline_logreg_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Logistic Regression Baseline Report\n\n")
            f.write("```json\n" + json.dumps(lr_metrics, indent=2) + "\n```\n")

        with open("reports/baseline_xgboost_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Gradient Boosting / XGBoost Baseline Report\n\n")
            f.write("```json\n" + json.dumps(gb_metrics, indent=2) + "\n```\n")

        return {"LogisticRegression": lr_metrics, "GradientBoosting": gb_metrics}


if __name__ == "__main__":
    trainer = TabularBaselineTrainer()
    trainer.train_and_evaluate()
