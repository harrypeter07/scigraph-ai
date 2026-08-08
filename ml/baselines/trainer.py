"""Tabular Baselines Trainer (Logistic Regression, XGBoost / Gradient Boosting).

Trains tabular models on time-consistent splits and computes evaluation metrics.
Gracefully falls back to GradientBoostingClassifier if xgboost is not installed.
"""

import os
import json
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

from ml.features.extractor import extract_tabular_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BaselineTrainer")


class TabularBaselineTrainer:
    """Trainer for Logistic Regression and XGBoost baseline classifiers."""

    def __init__(self, config_path: str = "configs/baselines.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.lr_params = self.config.get("logistic_regression", {})
        self.xgb_params = self.config.get("xgboost", {})

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute Accuracy, Macro-F1, Precision, Recall, and Confusion Matrix."""
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(float(f1), 4),
            "macro_precision": round(float(precision), 4),
            "macro_recall": round(float(recall), 4),
            "per_class_f1": [round(float(v), 4) for v in per_class_f1],
            "confusion_matrix": cm
        }

    def train_and_evaluate(self, train_path: str = "data/processed/train_temporal.parquet", test_path: str = "data/processed/test_temporal.parquet") -> Dict[str, Any]:
        """Train models on train_path and evaluate on test_path."""
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing required dataset split files: {train_path} or {test_path}")

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        X_train, y_train, feature_cols = extract_tabular_features(train_df)
        X_test, y_test, _ = extract_tabular_features(test_df)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # 1. Logistic Regression
        lr_model = LogisticRegression(
            C=self.lr_params.get("C", 1.0),
            max_iter=self.lr_params.get("max_iter", 1000),
            solver=self.lr_params.get("solver", "lbfgs"),
            random_state=self.lr_params.get("random_state", 42)
        )
        lr_model.fit(X_train_scaled, y_train)
        lr_preds = lr_model.predict(X_test_scaled)
        results["LogisticRegression"] = self.evaluate_predictions(y_test, lr_preds)

        # 2. XGBoost / GradientBoosting Fallback
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=self.xgb_params.get("n_estimators", 100),
                max_depth=self.xgb_params.get("max_depth", 4),
                learning_rate=self.xgb_params.get("learning_rate", 0.1),
                random_state=self.xgb_params.get("random_state", 42)
            )
            xgb_model.fit(X_train_scaled, y_train)
            xgb_preds = xgb_model.predict(X_test_scaled)
            results["XGBoost"] = self.evaluate_predictions(y_test, xgb_preds)
        else:
            gb_model = GradientBoostingClassifier(
                n_estimators=self.xgb_params.get("n_estimators", 100),
                max_depth=self.xgb_params.get("max_depth", 4),
                learning_rate=self.xgb_params.get("learning_rate", 0.1),
                random_state=self.xgb_params.get("random_state", 42)
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_preds = gb_model.predict(X_test_scaled)
            results["GradientBoosting_Fallback"] = self.evaluate_predictions(y_test, gb_preds)

        logger.info("Baseline training and evaluation complete.")
        for model_name, metrics in results.items():
            logger.info(f"Model: {model_name} | Accuracy: {metrics['accuracy']} | Macro-F1: {metrics['macro_f1']}")

        return results


if __name__ == "__main__":
    trainer = TabularBaselineTrainer()
    metrics_res = trainer.train_and_evaluate()

    # Save report
    report_path = "reports/baseline_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tabular Baseline Models Report\n\n")
        f.write("```json\n" + json.dumps(metrics_res, indent=2) + "\n```\n")
    print("Baseline metrics report written to:", report_path)
