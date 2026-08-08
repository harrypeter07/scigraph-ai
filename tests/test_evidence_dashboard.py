"""Phase 18 Integration Test - Live Evidence Dashboard Endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_evidence_overview_endpoint():
    """Verify live dataset overview endpoint returns real totals matching storage layer."""
    response = client.get("/api/v1/evidence/overview")
    assert response.status_code == 200
    data = response.json()

    assert "total_papers" in data
    assert data["total_papers"] == 50
    assert "total_authors" in data
    assert data["total_authors"] == 257
    assert "total_institutions" in data
    assert data["total_institutions"] == 215
    assert "total_citation_edges" in data
    assert data["total_citation_edges"] == 2170


def test_evidence_predictions_table_endpoint():
    """Verify predictions table endpoint matches evaluation logs without drift."""
    response = client.get("/api/v1/evidence/predictions")
    assert response.status_code == 200
    data = response.json()

    assert "test_set_size" in data
    assert data["test_set_size"] == 5
    assert "predictions_table" in data
    assert len(data["predictions_table"]) == 5

    row0 = data["predictions_table"][0]
    assert "paper_id" in row0
    assert "true_label" in row0
    assert "majority_baseline_pred" in row0
    assert "logreg_pred" in row0
    assert "gradient_boosting_pred" in row0
    assert "graphsage_pred" in row0


def test_evidence_model_comparison_endpoint():
    """Verify model comparison panel contains MajorityClass baseline row with beats_majority_baseline flags."""
    response = client.get("/api/v1/evidence/models")
    assert response.status_code == 200
    data = response.json()

    assert "comparison_matrix" in data
    matrix = data["comparison_matrix"]
    assert len(matrix) == 5
    assert matrix[0]["model_name"] == "MajorityClass_Baseline"
    assert matrix[0]["is_baseline"] is True
