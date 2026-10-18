"""Unit and integration tests for Topic Feasibility & Research Success Rate Explorer."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_topic_presets():
    response = client.get("/api/v1/topic/presets")
    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert len(data["presets"]) >= 5
    assert all("title" in p and "field" in p for p in data["presets"])


def test_topic_explore_mode_m():
    response = client.get("/api/v1/topic/explore?query=deep+learning&mode=M")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "deep learning"
    assert data["mode"] == "M"
    assert "topic_info" in data
    assert "feasibility_metrics" in data
    assert 0 <= data["feasibility_metrics"]["success_rate_pct"] <= 100
    assert "pillars" in data["feasibility_metrics"]
    assert "top_venues" in data
    assert "seminal_papers" in data
    assert "recent_papers" in data
    assert "gemini_analysis" in data
    assert len(data["gemini_analysis"]["novel_research_gaps"]) >= 2


def test_topic_explore_validation():
    response = client.get("/api/v1/topic/explore?query=")
    assert response.status_code in [400, 422]
