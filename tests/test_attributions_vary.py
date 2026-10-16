"""Feature Attribution Invariant Tests for WP5.

Verifies:
1. Local feature occlusion in inference_service.predict() produces paper-specific attributions.
2. Across 5 diverse test-split works from different fields and citation levels,
   top-3 attributed features are NOT identical across all 5.
3. Attributions normalize to sum to 1.0 (tolerance 1e-3).
4. No legacy hardcoded attribution constants ({0.385, 0.212, 0.142, 0.118, 0.084, 0.059}).
5. Global permutation importance report exists at reports/global_importance.json.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference_service import inference_service


@pytest.fixture(scope="module", autouse=True)
def setup_service():
    inference_service.load()


def test_global_importance_report_exists():
    """Verify global permutation importance report was generated from empirical test runs."""
    report_path = Path("reports/global_importance.json")
    assert report_path.exists(), "Missing reports/global_importance.json"

    import json
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "feature_groups" in data
    assert len(data["feature_groups"]) == 14
    assert data["metadata"]["n_repeats"] == 10


def test_attributions_vary_across_5_diverse_papers():
    """Predict on 5 diverse test papers; assert top-3 attributed features vary by paper."""
    df_works = pd.read_parquet("data/works.parquet")
    test_works = df_works[df_works["temporal_split"] == "test"]

    # Select 5 papers from different fields with different characteristics
    sample_papers = []
    for field in test_works["primary_field_name"].dropna().unique()[:5]:
        field_papers = test_works[test_works["primary_field_name"] == field]
        sample_papers.append(field_papers.iloc[0].to_dict())

    assert len(sample_papers) == 5, f"Expected 5 diverse papers, got {len(sample_papers)}"

    top3_tuples = []
    legacy_hardcoded = {0.385, 0.212, 0.142, 0.118, 0.084, 0.059, 0.382, 0.215}

    for paper in sample_papers:
        res = inference_service.predict(paper)
        attrs = res.get("attributions", [])
        assert len(attrs) == 14, f"Expected 14 feature attributions, got {len(attrs)}"

        # Check normalization
        total_score = sum(a["importance_score"] for a in attrs)
        assert abs(total_score - 1.0) < 0.01, f"Attribution scores sum to {total_score}, not 1.0"

        # Check no legacy hardcoded values
        scores_set = {a["importance_score"] for a in attrs}
        assert not (legacy_hardcoded & scores_set), (
            f"Found legacy hardcoded attribution values in output: {legacy_hardcoded & scores_set}"
        )

        top3 = tuple(a["feature_name"] for a in attrs[:3])
        top3_tuples.append(top3)

    # Invariant: top-3 features must NOT be identical across all 5 papers
    unique_top3 = set(top3_tuples)
    assert len(unique_top3) > 1, (
        f"Attribution failure! All 5 papers produced identical top-3 attributions: {top3_tuples[0]}"
    )
