"""Phase 12 Unit Tests - FastAPI Endpoint Tests."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check route."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stats_endpoint():
    """Test dataset stats API route."""
    response = client.get("/api/v1/papers/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_papers" in data
    assert "total_authors" in data
