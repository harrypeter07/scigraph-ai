"""Phase 3 Unit Tests - Temporal Leakage Auditing & Snapshot Verification."""

import pytest
import pandas as pd
from ml.temporal.snapshotter import compute_temporal_citations, snapshot_paper_features


def test_compute_temporal_citations_strict_cutoff():
    """Assert historical citations only accumulate years <= pub_year (T_cutoff)."""
    pub_year = 2017
    counts_by_year = [
        {"year": 2015, "cited_by_count": 5},
        {"year": 2016, "cited_by_count": 10},
        {"year": 2017, "cited_by_count": 15}, # Cutoff year
        {"year": 2018, "cited_by_count": 50}, # Future (Y+1)
        {"year": 2019, "cited_by_count": 60}, # Future (Y+2)
        {"year": 2020, "cited_by_count": 70}, # Future (Y+3)
        {"year": 2021, "cited_by_count": 80}, # Future (Y+4)
        {"year": 2022, "cited_by_count": 90}, # Future (Y+5)
        {"year": 2023, "cited_by_count": 100} # Beyond horizon (Y+6)
    ]

    hist_c, fut_c = compute_temporal_citations(counts_by_year, pub_year=pub_year, horizon_years=5)

    # Historical at cutoff must be 5 + 10 + 15 = 30
    assert hist_c == 30

    # 5-year future trajectory must be 50 + 60 + 70 + 80 + 90 = 350
    assert fut_c == 350


def test_snapshot_paper_features_dataframe():
    """Verify DataFrame temporal snapshotting applies correct column additions and leakage bounds."""
    df = pd.DataFrame([
        {
            "id": "W1",
            "publication_year": 2018,
            "counts_by_year": [
                {"year": 2018, "cited_by_count": 4},
                {"year": 2019, "cited_by_count": 20},
                {"year": 2020, "cited_by_count": 30}
            ]
        }
    ])

    snap_df = snapshot_paper_features(df, horizon_years=5)

    assert "historical_citation_count_at_cutoff" in snap_df.columns
    assert "delta_citations_5y" in snap_df.columns

    row = snap_df.iloc[0]
    assert row["historical_citation_count_at_cutoff"] == 4
    assert row["delta_citations_5y"] == 50
