"""Baseline Model Trainer for Tabular Data (Majority Class, Logistic Regression & Gradient Boosting GBDT).

Evaluates models independently on temporal test split and outputs baseline-anchored reports per model.
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
from ml.evaluation.baseline_anchor import evaluate_model_with_baseline_anchor, compute_majority_class_baseline

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
        """Train baseline models independently on train_path and evaluate on test_path."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing required dataset split files: {train_path} or {test_path}")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        X_train, y_train, _ = extract_tabular_features(train_df)
        X_test, y_test, _ = extract_tabular_features(test_df)

        # 1. Majority-Class Baseline (Anchor)
        majority_stats = compute_majority_class_baseline(y_train, y_test)

        # 2. Logistic Regression Model
        clf_lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf_lr.fit(X_train, y_train)
        preds_lr = clf_lr.predict(X_test)
        lr_anchored = evaluate_model_with_baseline_anchor("LogisticRegression", preds_lr, y_test, y_train)

        # 3. Gradient Boosting Classifier (GBDT)
        clf_gb = GradientBoostingClassifier(random_state=self.random_state)
        clf_gb.fit(X_train, y_train)
        preds_gb = clf_gb.predict(X_test)
        gb_anchored = evaluate_model_with_baseline_anchor("GradientBoostingClassifier", preds_gb, y_test, y_train)

        logger.info(f"MajorityClass Baseline Test Accuracy: {majority_stats['accuracy_fraction']} ({majority_stats['accuracy_percentage']})")
        logger.info(f"LogisticRegression: {lr_anchored['evaluation_summary']['accuracy_fraction']} | Verdict: {lr_anchored['baseline_anchor']['plain_language_verdict']}")
        logger.info(f"GradientBoosting: {gb_anchored['evaluation_summary']['accuracy_fraction']} | Verdict: {gb_anchored['baseline_anchor']['plain_language_verdict']}")

        # Save separate artifacts
        os.makedirs("reports", exist_ok=True)

        with open("reports/baseline_majority_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Majority Class Baseline Report\n\n")
            f.write("```json\n" + json.dumps(majority_stats, indent=2) + "\n```\n")

        with open("reports/baseline_logreg_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Logistic Regression Baseline-Anchored Report\n\n")
            f.write("```json\n" + json.dumps(lr_anchored, indent=2) + "\n```\n")

        with open("reports/baseline_gbdt_report.md", "w", encoding="utf-8") as f:
            f.write("# SciGraph AI — Gradient Boosting (GBDT) Baseline-Anchored Report\n\n")
            f.write("```json\n" + json.dumps(gb_anchored, indent=2) + "\n```\n")

        return {
            "MajorityClass_Baseline": majority_stats,
            "LogisticRegression": lr_anchored,
            "GradientBoosting": gb_anchored
        }


if __name__ == "__main__":
    trainer = TabularBaselineTrainer()
    trainer.train_and_evaluate()
