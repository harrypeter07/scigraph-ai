"""Phase 4 Unit Tests - Cohort-Normalized Labeling & Class Distribution Sanity."""

import pytest
import pandas as pd
from ml.labels.labeler import CohortImpactLabeler


def test_cohort_labeler_mock_dataset():
    """Verify cohort percentile rank assignment and 3-class distribution."""
    # Create 10 mock papers in same cohort with varying citation trajectories
    records = []
    for i in range(10):
        records.append({
            "id": f"W{i}",
            "publication_year": 2018,
            "primary_topic_id": "T001",
            "delta_citations_5y": i * 10  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
        })

    df = pd.DataFrame(records)
    labeler = CohortImpactLabeler()
    labeled_df = labeler.label_dataset(df)

    assert "impact_cohort_percentile" in labeled_df.columns
    assert "impact_label" in labeled_df.columns

    # Verify percentiles are bounded between 0 and 100
    assert (labeled_df["impact_cohort_percentile"] >= 0).all()
    assert (labeled_df["impact_cohort_percentile"] <= 100).all()

    # Highest paper (90 citations) should be Class 2 (High)
    high_paper = labeled_df[labeled_df["id"] == "W9"].iloc[0]
    assert high_paper["impact_label"] == 2

    # Lowest paper (0 citations) should be Class 0 (Low)
    low_paper = labeled_df[labeled_df["id"] == "W0"].iloc[0]
    assert low_paper["impact_label"] == 0


def test_labeler_config_thresholds():
    """Test custom percentile threshold assignment."""
    labeler = CohortImpactLabeler()
    assert labeler.assign_class_from_percentile(30.0) == 0  # Low
    assert labeler.assign_class_from_percentile(75.0) == 1  # Medium
    assert labeler.assign_class_from_percentile(95.0) == 2  # High
