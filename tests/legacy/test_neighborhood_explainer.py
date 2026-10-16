"""Phase 19 Explainability Test: Receptive Field Neighborhood Explainer."""

import pytest
from ml.explainability.neighborhood_explainer import ReceptiveFieldExplainer


def test_neighborhood_explainer_initialization():
    """Verify explainer initializes with real dataset and model."""
    explainer = ReceptiveFieldExplainer()
    assert explainer.papers_df is not None
    assert len(explainer.papers_df) > 0


def test_receptive_field_temporal_cutoff_boundary():
    """LEAKAGE-CRITICAL TEST: Verify all extracted neighbor nodes and edges
    are strictly timestamped at or before T_cutoff = publication_year.
    """
    explainer = ReceptiveFieldExplainer()
    # Test on ResNet W2194775991 (pub_year=2016)
    res = explainer.explain_paper_neighborhood("W2194775991")

    assert res["paper_id"] == "https://openalex.org/W2194775991"
    pub_year = res["publication_year"]
    t_cutoff = res["t_cutoff_boundary"]
    assert pub_year == 2016
    assert t_cutoff == 2016

    nodes = res["receptive_field"]["nodes"]
    edges = res["receptive_field"]["edges"]

    assert len(nodes) > 0
    assert len(edges) > 0

    # 1. Assert NO node has timestamp > t_cutoff
    for node in nodes:
        node_ts = node.get("timestamp")
        if node_ts is not None:
            assert node_ts <= t_cutoff, f"Leakage detected: node {node['id']} has timestamp {node_ts} > {t_cutoff}"

    # 2. Assert NO edge has timestamp > t_cutoff
    for edge in edges:
        edge_ts = edge.get("timestamp")
        if edge_ts is not None:
            assert edge_ts <= t_cutoff, f"Leakage detected: edge {edge['id']} has timestamp {edge_ts} > {t_cutoff}"

    # 3. Assert feature attributions sum sensibly and are non-empty
    attributions = res["feature_attributions"]
    assert len(attributions) == 4
    total_attr_weight = sum(a["weight"] for a in attributions)
    assert total_attr_weight > 0.8
