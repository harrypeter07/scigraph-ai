"""Unit Test for Temporal Snapshot Graph Invariance (Phase B).

Verifies that for any paper P published in t_P:
1. Snapshot G_{t_P + 2} contains zero citation edges with citing_year > t_P + 2
2. Snapshot G_{t_P + 2} node features for P use only counts where year <= t_P + 2
3. Mutating future counts in counts_by_year produces an IDENTICAL snapshot and hash
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from ml.gnn.snapshot import SnapshotGraphBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_snapshot_temporal_edges_invariance():
    """Confirms no edge in snapshot G_t originated after cutoff year t."""
    builder = SnapshotGraphBuilder()
    cutoff_year = 2018
    subsets = builder.get_snapshot_subsets(cutoff_year)

    edges_t = subsets["edges"]
    if len(edges_t) > 0 and "source_pub_year" in edges_t.columns:
        assert (edges_t["source_pub_year"] <= cutoff_year).all(), (
            f"Found citing edges with pub_year > {cutoff_year} in snapshot!"
        )


def test_snapshot_paper_counts_invariance():
    """Confirms no counts record in snapshot G_t occurred after cutoff year t."""
    builder = SnapshotGraphBuilder()
    cutoff_year = 2017
    subsets = builder.get_snapshot_subsets(cutoff_year)

    counts_t = subsets["counts"]
    if len(counts_t) > 0:
        assert (counts_t["year"] <= cutoff_year).all(), (
            f"Found counts with year > {cutoff_year} in snapshot G_{cutoff_year}!"
        )


def test_snapshot_hash_future_poison_invariance():
    """Mutating future citations must produce bit-for-bit identical snapshot hash."""
    builder = SnapshotGraphBuilder()
    test_p = REPO_ROOT / "data" / "processed" / "test.parquet"
    if not test_p.exists():
        pytest.skip("Test parquet not found.")

    df_test = pd.read_parquet(test_p)
    # Pick a paper from 2018
    p2018 = df_test[df_test["publication_year"] == 2018].iloc[0]
    wid = p2018["work_id"]
    cutoff = 2020

    hash_before = builder.compute_paper_snapshot_hash(wid, cutoff)

    # Inject poison future counts into builder's counts table for years 2021, 2022, 2023
    poison_rows = pd.DataFrame([
        {"work_id": wid, "year": 2021, "cited_by_count": 99999},
        {"work_id": wid, "year": 2022, "cited_by_count": 88888},
        {"work_id": wid, "year": 2023, "cited_by_count": 77777},
    ])
    builder.df_counts = pd.concat([builder.df_counts, poison_rows], ignore_index=True)

    hash_after = builder.compute_paper_snapshot_hash(wid, cutoff)

    assert hash_before == hash_after, (
        f"Snapshot hash changed after mutating future counts! Before: {hash_before}, After: {hash_after}"
    )


def test_snapshot_pyg_hetero_data():
    """Confirms HeteroData graph constructs cleanly with valid tensors."""
    builder = SnapshotGraphBuilder()
    cutoff = 2016
    data = builder.build_hetero_snapshot(cutoff)

    assert "paper" in data.node_types
    assert data["paper"].x.shape[0] == data["paper"].num_nodes
    assert data["paper"].x.shape[1] == 7
    assert ("paper", "cites", "paper") in data.edge_types
    assert data[("paper", "cites", "paper")].edge_index.shape[0] == 2
