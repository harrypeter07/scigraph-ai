"""Phase 5 Unit Tests - Temporal Splitting & Split Integrity Assertions."""

import pytest
import pandas as pd
from ml.temporal.splitter import TemporalDatasetSplitter


def test_time_consistent_split_integrity():
    """Assert train, val, and test year boundaries do not overlap and paper IDs are disjoint."""
    records = []
    for yr in range(2012, 2022):
        for idx in range(5):
            records.append({
                "id": f"W_{yr}_{idx}",
                "publication_year": yr,
                "title": f"Paper {yr} {idx}"
            })

    df = pd.DataFrame(records)
    splitter = TemporalDatasetSplitter()
    train_df, val_df, test_df = splitter.split_time_consistent(df)

    # Check publication year boundaries
    assert train_df["publication_year"].max() <= 2018
    assert (val_df["publication_year"] == 2019).all()
    assert test_df["publication_year"].min() >= 2020

    # Check ID set disjointness
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_naive_random_split():
    """Verify naive random split proportions and disjoint ID sets."""
    records = [{"id": f"W_{i}", "publication_year": 2018} for i in range(100)]
    df = pd.DataFrame(records)

    splitter = TemporalDatasetSplitter()
    train_df, val_df, test_df = splitter.split_naive_random(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    test_ids = set(test_df["id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
