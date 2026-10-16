"""Phase 6 Unit Tests - Tabular Baselines & Metric Computations."""

import pytest
import numpy as np
import pandas as pd
from ml.features.extractor import extract_tabular_features
from ml.baselines.trainer import TabularBaselineTrainer


def test_tabular_feature_extraction():
    """Test feature matrix extraction from snapshotted DataFrame."""
    df = pd.DataFrame([
        {
            "id": "W1",
            "publication_year": 2018,
            "title": "A Sample Neural Network Paper",
            "abstract_text": "We present a novel neural architecture.",
            "historical_citation_count_at_cutoff": 10,
            "referenced_works_count": 25,
            "impact_label": 1
        }
    ])

    X, y, feature_cols = extract_tabular_features(df)

    assert X.shape == (1, 5)
    assert y.shape == (1,)
    assert y[0] == 1
    assert "historical_citation_count_at_cutoff" in feature_cols


def test_baseline_trainer_mock_execution(tmp_path):
    """Test end-to-end training and evaluation of Logistic Regression & XGBoost on mock Parquets."""
    train_records = []
    test_records = []

    for i in range(20):
        train_records.append({
            "id": f"W_tr_{i}",
            "publication_year": 2017,
            "title": f"Train paper {i}",
            "abstract_text": "Sample abstract content",
            "historical_citation_count_at_cutoff": i * 2,
            "referenced_works_count": 10 + i,
            "impact_label": i % 3
        })

    for i in range(10):
        test_records.append({
            "id": f"W_te_{i}",
            "publication_year": 2020,
            "title": f"Test paper {i}",
            "abstract_text": "Sample abstract content",
            "historical_citation_count_at_cutoff": i * 3,
            "referenced_works_count": 12 + i,
            "impact_label": i % 3
        })

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_file = tmp_path / "train_temporal.parquet"
    test_file = tmp_path / "test_temporal.parquet"

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)

    trainer = TabularBaselineTrainer()
    metrics = trainer.train_and_evaluate(train_path=str(train_file), test_path=str(test_file))

    assert "LogisticRegression" in metrics
    assert "XGBoost" in metrics or "GradientBoosting" in metrics or "GradientBoosting_Fallback" in metrics
    tree_key = [k for k in metrics.keys() if k != "LogisticRegression"][0]
    assert "accuracy" in metrics[tree_key] or "accuracy_fraction" in metrics[tree_key]
