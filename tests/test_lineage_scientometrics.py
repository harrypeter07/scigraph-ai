"""Tests for Scientometric Lineage Engine & CD Disruption Index (WP8).

Verifies:
1. Exact analytical correctness of CD index on a 5-node toy graph.
2. Inflow citations are strictly restricted to years <= cutoff_year across 10 sample papers.
3. Unknown or malformed papers handle gracefully without throwing 500 or raising exceptions.
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from ml.scientometrics.lineage_engine import (
    compute_cd_index,
    ScientometricLineageEngine,
    lineage_engine
)
from api.config import settings


def test_cd_index_analytical_toy_graph():
    """Verifies CD index against an analytical 5-node citation topology.
    
    Topology:
      - Focal paper: F
      - Reference paper cited by F: R (edge: F -> R)
      - Citers:
        - C1: cites F only (not R) -> set_i
        - C2: cites F and R -> set_j
        - C3: cites R only (not F) -> set_k
    
    Analytical Expectation:
      n_i = 1, n_j = 1, n_k = 1
      CD = (1 - 1) / (1 + 1 + 1) = 0.0
    """
    edges_toy = [
        ("F", "R"),     # F cites R
        ("C1", "F"),    # C1 cites F only
        ("C2", "F"),    # C2 cites F
        ("C2", "R"),    # C2 also cites R
        ("C3", "R"),    # C3 cites R only
    ]
    res = compute_cd_index(focal_id="F", reference_ids=["R"], citation_edges=edges_toy)
    assert res["n_i"] == 1
    assert res["n_j"] == 1
    assert res["n_k"] == 1
    assert res["cd_disruption_index"] == 0.0
    assert res["total_citers"] == 3

    # Asymmetric test case: highly disruptive innovation
    # C1, C2 cite F only. C3 cites both F and R. No paper cites R only.
    # n_i = 2, n_j = 1, n_k = 0 -> CD = (2 - 1) / (2 + 1 + 0) = 1/3 = 0.3333
    edges_disruptive = [
        ("F", "R"),
        ("C1", "F"),
        ("C2", "F"),
        ("C3", "F"),
        ("C3", "R"),
    ]
    res_disruptive = compute_cd_index(focal_id="F", reference_ids=["R"], citation_edges=edges_disruptive)
    assert res_disruptive["n_i"] == 2
    assert res_disruptive["n_j"] == 1
    assert res_disruptive["n_k"] == 0
    assert pytest.approx(res_disruptive["cd_disruption_index"], 0.001) == 0.3333

    # Consolidation case:
    # C1 cites both F and R (n_j = 1). C2 cites R only (n_k = 1). None cites F only (n_i = 0).
    # n_i = 0, n_j = 1, n_k = 1 -> CD = (0 - 1) / (0 + 1 + 1) = -0.5
    edges_consolidating = [
        ("F", "R"),
        ("C1", "F"),
        ("C1", "R"),
        ("C2", "R"),
    ]
    res_cons = compute_cd_index(focal_id="F", reference_ids=["R"], citation_edges=edges_consolidating)
    assert res_cons["n_i"] == 0
    assert res_cons["n_j"] == 1
    assert res_cons["n_k"] == 1
    assert res_cons["cd_disruption_index"] == -0.5


def test_inflow_citations_cutoff_temporal_constraint():
    """Verifies that for 10 corpus works, inflow citations at cutoff equal the sum of counts_by_year <= cutoff."""
    cache_dir = settings.DATA_DIR / "cache" / "openalex"
    cached_files = sorted(list(cache_dir.glob("work_*.json")))
    
    test_works = []
    for fpath in cached_files[:10]:
        with open(fpath, "r", encoding="utf-8") as f:
            test_works.append(json.load(f))

    # Add additional works from works.parquet with mock counts_by_year to reach 10 papers
    if len(test_works) < 10:
        df_works = pd.read_parquet(settings.DATA_DIR / "works.parquet", columns=["work_id", "title", "publication_year", "citations_at_cutoff"])
        for _, row in df_works.head(10 - len(test_works)).iterrows():
            py = int(row["publication_year"])
            # synthesize multi-year counts_by_year including years before, at, and after cutoff
            c_by_y = [
                {"year": py, "cited_by_count": 5},
                {"year": py + 1, "cited_by_count": 12},
                {"year": py + 2, "cited_by_count": 25},   # cutoff year
                {"year": py + 3, "cited_by_count": 50},   # future (leaked if unconstrained)
                {"year": py + 4, "cited_by_count": 100},  # future
            ]
            test_works.append({
                "id": f"https://openalex.org/{row['work_id']}",
                "title": row["title"],
                "publication_year": py,
                "counts_by_year": c_by_y,
                "referenced_works": []
            })

    assert len(test_works) == 10, f"Expected 10 works, got {len(test_works)}"

    for work in test_works:
        pub_year = work.get("publication_year", 2016)
        cutoff_year = pub_year + 2
        counts_by_year = work.get("counts_by_year", [])

        # Compute manual sum of citations accrued strictly up to cutoff_year
        expected_inflow = sum(
            c.get("cited_by_count", 0)
            for c in counts_by_year
            if isinstance(c, dict) and c.get("year") is not None and c.get("year") <= cutoff_year
        )

        res = lineage_engine.compute(work, cutoff_year=cutoff_year)

        assert res.inflow_count == expected_inflow, (
            f"Paper {work.get('id')} inflow mismatch: expected {expected_inflow}, got {res.inflow_count}"
        )
        assert res.cutoff_year_applied == cutoff_year
        assert res.inflow_temporal_scope == "as_of_cutoff"


def test_unknown_or_malformed_paper_graceful_handling():
    """Verifies that null, empty, or un-indexed papers return valid null-safe lineage models without 500s."""
    # 1. Empty dict
    res_empty = lineage_engine.compute({})
    assert res_empty.inflow_count == 0
    assert res_empty.outflow_count == 0
    assert res_empty.inflow_outflow_ratio is None
    assert res_empty.cd_disruption_index is None
    assert res_empty.cd_disruption_index_status in ["unknown_work", "insufficient_reference_subgraph"]

    # 2. None input
    res_none = lineage_engine.compute(None)
    assert res_none.inflow_count == 0
    assert res_none.outflow_count == 0

    # 3. Completely fictitious work
    fake_work = {
        "id": "https://openalex.org/W999999999999",
        "title": "Non-existent Fictional Quantum Gravity Proof",
        "publication_year": 2020,
        "referenced_works": ["https://openalex.org/W111111111111"],
        "counts_by_year": [{"year": 2021, "cited_by_count": 5}]
    }
    res_fake = lineage_engine.compute(fake_work, cutoff_year=2022)
    assert res_fake.inflow_count == 5
    assert res_fake.outflow_count == 1
    assert res_fake.cd_disruption_index is None
    assert res_fake.cd_disruption_index_status == "insufficient_reference_subgraph"
