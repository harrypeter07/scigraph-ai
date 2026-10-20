"""UI metrics must match measured artifacts and the actually loaded bundle."""
import json
from pathlib import Path
from types import SimpleNamespace
from api.forecast_evidence import get_forecast_evidence

ROOT = Path(__file__).resolve().parents[1]


def service(checksum=None, loaded=True, graph_weight=0):
    return SimpleNamespace(forecast_bundle={"model_name": "hist_leaves=31_weight=None"} if loaded else None,
                           forecast_bundle_sha256=checksum,
                           graph_predictor=SimpleNamespace(graph={"graph_weight": graph_weight}))


def test_published_metrics_equal_saved_evaluation():
    manifest = json.loads((ROOT / "web/data/forecast_evidence.json").read_text())
    report = json.loads((ROOT / "reports/expanded_forecast/evaluation.json").read_text())
    assert manifest["primary"] == report["strict_temporal_test"]
    assert manifest["primary"]["n"] == 5933
    assert manifest["dataset"]["train"] == 29312


def test_matching_loaded_snapshot_is_verified():
    checksum = json.loads((ROOT / "web/data/forecast_evidence.json").read_text())["forecast_bundle_sha256"]
    result = get_forecast_evidence(ROOT, ROOT / "ml/gnn/checkpoints", service(checksum))
    assert result["serving"]["verified"] is True


def test_different_or_unloaded_model_cannot_claim_published_accuracy():
    for state in [service("different"), service(None), service("different", loaded=False)]:
        result = get_forecast_evidence(ROOT, ROOT / "ml/gnn/checkpoints", state)
        assert result["serving"]["verified"] is False


def test_graph_changing_the_forecast_invalidates_tabular_manifest():
    checksum = json.loads((ROOT / "web/data/forecast_evidence.json").read_text())["forecast_bundle_sha256"]
    result = get_forecast_evidence(ROOT, ROOT / "ml/gnn/checkpoints", service(checksum, graph_weight=.5))
    assert result["serving"]["verified"] is False
