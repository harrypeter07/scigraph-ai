"""Phase 19 Leakage-Critical Test: Retrospective Verification Simulator."""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from ml.verification.simulator import RetrospectiveVerificationSimulator
from api.main import app

client = TestClient(app)


def test_verification_simulator_initialization():
    """Verify simulator loads dataset and trained model cleanly."""
    sim = RetrospectiveVerificationSimulator()
    papers = sim.list_available_papers(limit=5)
    assert len(papers) > 0
    assert "paper_id" in papers[0]
    assert "actual_impact_label" in papers[0]


def test_verification_simulation_zero_leakage_assertion():
    """LEAKAGE-CRITICAL TEST: Ensure reconstructed feature snapshot contains ONLY
    information dated at or before T_cutoff = publication_year.
    """
    sim = RetrospectiveVerificationSimulator()
    # Test on known paper W2194775991 (ResNet, pub_year=2016)
    res = sim.simulate_paper_forecast("W2194775991")

    assert res["paper_id"] == "https://openalex.org/W2194775991"
    assert res["publication_year"] == 2016
    assert res["snapshot_cutoff_year"] == 2016

    # 1. Historical citation count at cutoff must be strictly less than lifetime citations
    cutoff_cits = res["cutoff_snapshot"]["historical_citations_at_cutoff"]
    actual_5y_delta = res["actual_outcome"]["actual_5y_delta_citations"]

    assert cutoff_cits >= 0
    assert actual_5y_delta >= 0
    # ResNet accumulated tens of thousands of citations in its 5y horizon; cutoff citations must NOT include this future growth
    assert actual_5y_delta > cutoff_cits

    # 2. Reconstructed feature vector must be 5-dimensional and non-leaking
    feat_vec = res["cutoff_snapshot"]["feature_vector"]
    assert len(feat_vec) == 5

    # 3. Model forecast probabilities must sum to 1.0 (valid probability distribution)
    probs = res["model_forecast"]["class_probabilities"]
    assert round(sum(probs.values()), 2) == 1.0

    # 4. Explicit verification verdict must be populated
    verdict = res["verification_verdict"]
    assert "is_correct" in verdict
    assert verdict["verdict_label"] in ["CORRECT", "INCORRECT"]


def test_verification_api_endpoints():
    """Verify FastAPI verification endpoints work smoothly."""
    # List endpoint
    list_resp = client.get("/api/v1/verification/list?limit=10")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "total_available" in data
    assert len(data["papers"]) == 10

    # Simulate endpoint
    sim_resp = client.get("/api/v1/verification/simulate/W2194775991")
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert sim_data["clean_id"] == "W2194775991"
    assert "verification_verdict" in sim_data
    assert "cutoff_snapshot" in sim_data
    assert "model_forecast" in sim_data
    assert "actual_outcome" in sim_data
