"""Scientometrics and Academic Lineage Unit Tests.

Validates the 4 teacher-suggested core scientometrics axes:
1. Quartiles (Q1, Q2, Q3, Q4 journal/conference prestige tiers)
2. Paid vs Unpaid Access Economics (Reader financial costs & paywalls)
3. Open Access (Gold, Green, Hybrid, Bronze, Closed & OA citation boost)
4. Bidirectional Citation Lineage (Precursor Inflow ⇄ Target ⇄ Downstream Outflow)
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.routers.paper import compute_venue_quartile, compute_access_economics

client = TestClient(app)


def test_compute_venue_quartile_mapping():
    """Verify venue quartile heuristic correctly assigns Q1, Q2, Q3, and Preprint tiers."""
    # Q1 Elite Conference
    cvpr_source = {
        "display_name": "IEEE Conference on Computer Vision and Pattern Recognition",
        "host_organization_name": "IEEE Computer Society",
        "issn_l": "1063-6919",
        "type": "conference"
    }
    q1_res = compute_venue_quartile(cvpr_source)
    assert q1_res["quartile"] == "Q1"
    assert q1_res["is_q1"] is True
    assert q1_res["prestige_score"] >= 0.90
    assert "TIER-1" in q1_res["quartile_badge"]

    # Q1 Nature / Cell / Lancet
    nature_source = {
        "display_name": "Nature Communications",
        "host_organization_name": "Springer Nature",
        "issn_l": "2041-1723",
        "type": "journal"
    }
    nature_res = compute_venue_quartile(nature_source)
    assert nature_res["quartile"] == "Q1"
    assert nature_res["is_q1"] is True

    # Preprint Repository (e.g. arXiv)
    arxiv_source = {
        "display_name": "arXiv.org",
        "host_organization_name": "Cornell University",
        "type": "repository"
    }
    arxiv_res = compute_venue_quartile(arxiv_source)
    assert arxiv_res["quartile"] == "Preprint"
    assert arxiv_res["is_q1"] is False
    assert "PREPRINT" in arxiv_res["quartile_badge"]


def test_compute_access_economics_paid_vs_unpaid():
    """Verify access economics differentiates paid paywalled content from free open access."""
    # Paid / Closed paywalled publication
    closed_oa = {"is_oa": False, "oa_status": "closed"}
    closed_loc = {"is_oa": False, "landing_page_url": "https://doi.org/10.1000/182"}
    paid_res = compute_access_economics(closed_oa, closed_loc)
    
    assert paid_res["is_paid"] is True
    assert paid_res["is_open_access"] is False
    assert "$35.00+" in paid_res["reader_fee"]
    assert paid_res["oa_citation_boost_pct"] == 0.0
    assert "Paid" in paid_res["access_type"]

    # Unpaid / Gold open access publication
    gold_oa = {"is_oa": True, "oa_status": "gold"}
    gold_loc = {"is_oa": True, "pdf_url": "https://nature.com/article.pdf"}
    unpaid_res = compute_access_economics(gold_oa, gold_loc)
    
    assert unpaid_res["is_paid"] is False
    assert unpaid_res["is_open_access"] is True
    assert "$0.00" in unpaid_res["reader_fee"]
    assert unpaid_res["oa_citation_boost_pct"] == 28.4
    assert "Unpaid" in unpaid_res["access_type"]

    # Green OA repository publication
    green_oa = {"is_oa": True, "oa_status": "green"}
    green_loc = {"is_oa": True, "pdf_url": "https://arxiv.org/pdf/1512.03385"}
    green_res = compute_access_economics(green_oa, green_loc)
    assert green_res["oa_citation_boost_pct"] == 31.2


def test_predict_endpoint_scientometrics_payload():
    """Verify /api/v1/papers/predict/ payload includes all 4 concrete scientometrics modules."""
    # Test with landmark Deep Residual Learning paper
    response = client.get("/api/v1/papers/predict/W2194775991")
    assert response.status_code == 200
    data = response.json()

    # 1. Quartile Metrics
    assert "venue_quartile" in data
    vq = data["venue_quartile"]
    assert vq is not None
    assert vq["quartile"] in ["Q1", "Q2", "Q3", "Q4", "Preprint"]
    assert "quartile_badge" in vq
    assert "prestige_score" in vq

    # 2. Access Economics
    assert "access_economics" in data
    econ = data["access_economics"]
    assert econ is not None
    assert "access_type" in econ
    assert "reader_fee" in econ
    assert "oa_citation_boost_pct" in econ
    assert isinstance(econ["is_paid"], bool)

    # 3. Bidirectional Lineage
    assert "bidirectional_lineage" in data
    lineage = data["bidirectional_lineage"]
    assert lineage is not None
    assert "precursor_inflow_count" in lineage
    assert "downstream_outflow_count" in lineage
    assert "knowledge_flow_ratio" in lineage
    assert lineage["knowledge_flow_ratio"] > 0
    assert "top_downstream_works" in lineage
    assert len(lineage["top_downstream_works"]) > 0

    # 4. GNN Subgraph with new scientometric node types and edge relations
    subgraph = data["explanation"]["subgraph"]
    node_types = {n.get("type") for n in subgraph if "type" in n}
    edge_relations = {e.get("relation") for e in subgraph if "relation" in e}

    assert "paper" in node_types
    assert "venue" in node_types
    assert ("citation_inflow" in node_types) or ("citation" in node_types)
    assert "citation_outflow" in node_types

    assert "PUBLISHED_IN" in edge_relations
    assert "CITED_BY_DOWNSTREAM" in edge_relations
    assert ("CITES_PRECURSOR" in edge_relations) or ("CITES" in edge_relations)

    # 5. Explanations feature weights include the 4 axes
    features = [f["feature"] for f in data["explanation"]["top_contributing_features"]]
    assert "venue_quartile_prestige" in features
    assert "open_access_dissemination_boost" in features
    assert "bidirectional_lineage_momentum" in features
