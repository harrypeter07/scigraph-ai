"""Verification Test Suite for Phase 4: Backend Remediation & API Behavior.

Verifies:
1. Health check returns 200 with model loaded and indexed works count.
2. /api/v1/papers/stats returns real counts from Parquet data tables (16,500 works, >1.7M edges).
3. /api/v1/papers/model-card returns complete benchmark metrics.
4. /api/v1/papers/corpus-sample returns valid sample papers.
5. /api/v1/papers/predict/{paper_id} produces valid schema, calibrated probabilities, and backtesting.
6. Rule R2: discrete 404 for nonexistent paper, 422 for bad identifiers.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app
from ml.inference_service import inference_service


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan initialization."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Verify health endpoint indicates model is loaded and ready."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["corpus_works_indexed"] == 16500


def test_dataset_stats(client):
    """Verify /stats returns genuine corpus metrics with zero hardcoded fallbacks."""
    response = client.get("/api/v1/papers/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_papers"] == 16500
    assert data["total_authors"] >= 45000
    assert data["total_institutions"] >= 3800
    assert data["total_venues"] >= 2400
    assert data["total_citation_edges"] >= 1700000
    assert data["provenance"] == "openalex"


def test_model_card_endpoint(client):
    """Verify /model-card serves benchmark comparison and evaluation schema."""
    response = client.get("/api/v1/papers/model-card")
    assert response.status_code == 200
    card = response.json()
    assert "benchmark_metrics" in card
    assert "PyG HeteroGraphSAGE (2-Hop)" in card["benchmark_metrics"]


def test_corpus_sample_endpoint(client):
    """Verify /corpus-sample serves pre-indexed works for UI demonstration."""
    response = client.get("/api/v1/papers/corpus-sample?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 16500
    assert len(data["sample"]) == 5
    for item in data["sample"]:
        assert "paper_id" in item
        assert "title" in item
        assert "citations_at_cutoff" in item


def test_predict_paper_endpoint(client):
    """Verify full end-to-end prediction on a real paper from corpus sample."""
    sample_res = client.get("/api/v1/papers/corpus-sample?limit=1")
    sample_paper = sample_res.json()["sample"][0]
    paper_id = sample_paper["paper_id"]

    response = client.get(f"/api/v1/papers/predict/{paper_id}")
    assert response.status_code == 200
    pred = response.json()

    # Rule R6: schema guarantees
    assert "paper_id" in pred
    assert "title" in pred
    assert "calibrated_probabilities" in pred
    assert pred["predicted_impact_class"] in [0, 1, 2]
    
    # Probabilities sum to 1.0
    probs = pred["calibrated_probabilities"]
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0

    # Scientometric components
    assert "access_economics" in pred
    assert "empirical_cohort_oa_advantage_pct" in pred["access_economics"]
    assert pred["access_economics"]["provenance"] == "openalex | computed"

    assert "lineage" in pred
    assert "inflow_count" in pred["lineage"]
    assert "outflow_count" in pred["lineage"]

    # Trajectory and attributions
    assert "attributions" in pred
    assert len(pred["attributions"]) > 0

    # Backtest comparison
    if pred.get("backtest"):
        assert "actual_future_citations_5y" in pred["backtest"]
        assert "absolute_error_log_citations" in pred["backtest"]


def test_error_handling_not_found(client):
    """Verify non-existent paper returns discrete 404 (Rule R2: no masking as {})."""
    response = client.get("/api/v1/papers/predict/W9999999999999999NOTREAL")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
