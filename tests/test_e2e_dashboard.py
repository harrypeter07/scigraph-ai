"""Phase 14 Integration Test - End-to-End Dashboard & API Integration."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_e2e_dashboard_serving():
    """Verify HTML web dashboard is served at root /."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SciGraph AI" in response.text


def test_e2e_paper_prediction_flow():
    """Verify paper prediction endpoint returns structured probabilities and explanation."""
    response = client.get("/api/v1/papers/predict/W2194775991")
    assert response.status_code == 200
    data = response.json()

    assert "predicted_impact_class" in data
    assert "class_probabilities" in data
    assert "explanation" in data
    assert "top_contributing_features" in data["explanation"]
