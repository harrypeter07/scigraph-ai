"""Temporal Cutoff & Leakage Invariant Tests for WP2.

Verifies:
1. Future-poison test: post-cutoff counts and cumulative metrics produce byte-identical features.
2. Edge-time audit: 100% of citation edges in the HeteroData graph strictly satisfy
   citing_pub_year <= referenced_cutoff_year.
3. Label-shuffle sanity check: LightGBM trained on randomly permuted train labels yields
   val macro-F1 < 0.40 (near theoretical chance ~0.33 for 3 balanced classes).
4. Training split truncation guard: 100% of papers in the train split have publication_year >= 2012.
"""

import sys
import json
import copy
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features.builder import build_paper_features, build_paper_features_batch
from ml.gnn.graph_builder import build_heterogeneous_graph


def test_future_poison_invariance():
    """Mutating post-cutoff citations and contemporary metrics produces byte-identical features."""
    cache_path = Path("data/cache/openalex/work_W2194775991.json")
    assert cache_path.exists(), f"Cached work missing at {cache_path}"

    with open(cache_path, "r", encoding="utf-8") as f:
        clean_work = json.load(f)

    clean_feats, _ = build_paper_features(clean_work, cutoff_offset=2)

    # Poison post-cutoff counts and contemporary metrics
    poisoned_work = copy.deepcopy(clean_work)
    pub_year = int(poisoned_work.get("publication_year", 2016))
    cutoff_year = pub_year + 2

    for entry in poisoned_work.get("counts_by_year", []):
        if entry.get("year", 0) > cutoff_year:
            entry["cited_by_count"] = entry.get("cited_by_count", 0) * 1000

    poisoned_work["cited_by_count"] = 999_999_999
    poisoned_work["fwci"] = 888_888.0
    poisoned_work["citation_normalized_percentile"] = {"value": 999.0}

    poisoned_feats, _ = build_paper_features(poisoned_work, cutoff_offset=2)

    # Must be byte-identical
    assert np.array_equal(clean_feats, poisoned_feats), (
        f"Temporal leakage detected! Max difference: {np.max(np.abs(clean_feats - poisoned_feats))}"
    )


def test_citation_edges_strictly_as_of_cutoff():
    """Assert all internal citation edges in the graph satisfy citing_pub_year <= referenced_cutoff."""
    df_works = pd.read_parquet("data/works.parquet")
    pub_years = df_works["publication_year"].values
    cutoff_years = df_works["cutoff_year"].values

    graph = build_heterogeneous_graph()
    edge_index = graph["paper", "cites", "paper"].edge_index.numpy()

    cites_src = edge_index[0]  # citing papers
    cites_dst = edge_index[1]  # referenced papers

    citing_pub_years = pub_years[cites_src]
    referenced_cutoff_years = cutoff_years[cites_dst]

    leaked = citing_pub_years > referenced_cutoff_years
    leaked_count = int(leaked.sum())

    assert leaked_count == 0, (
        f"Found {leaked_count} citation edges violating temporal cutoff (citing_year > referenced_cutoff)!"
    )


def test_label_shuffle_sanity():
    """Assert LightGBM trained on randomly shuffled labels achieves val macro-F1 < 0.40."""
    df_works = pd.read_parquet("data/works.parquet")
    train_mask = (df_works["publication_year"] >= 2012) & (df_works["publication_year"] <= 2016)
    val_mask = df_works["publication_year"] == 2017

    df_train = df_works[train_mask].reset_index(drop=True)
    df_val = df_works[val_mask].reset_index(drop=True)

    X_train, _ = build_paper_features_batch(df_train)
    y_train = df_train["impact_class"].values.astype(int)

    X_val, _ = build_paper_features_batch(df_val)
    y_val = df_val["impact_class"].values.astype(int)

    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y_train)

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    clf.fit(X_train, y_shuffled)
    preds = clf.predict(X_val)

    macro_f1 = float(f1_score(y_val, preds, average="macro"))
    assert macro_f1 < 0.40, f"Label leakage detected! Shuffled val macro-F1 {macro_f1:.4f} >= 0.40"


def test_no_truncated_works_in_train_mask():
    """Assert train split excludes 2010-2011 works to guarantee zero citation count truncation."""
    df_works = pd.read_parquet("data/works.parquet")
    graph = build_heterogeneous_graph()

    train_mask = graph["paper"].train_mask.numpy()
    train_years = df_works.loc[train_mask, "publication_year"].values

    assert int(train_years.min()) >= 2012, (
        f"Train split contains works published before 2012: min year is {train_years.min()}"
    )
    assert int(train_years.max()) <= 2016, (
        f"Train split contains works published after 2016: max year is {train_years.max()}"
    )
    assert train_mask.sum() == 7500, f"Expected 7,500 train works (2012-2016), got {train_mask.sum()}"
