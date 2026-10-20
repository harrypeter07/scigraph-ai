"""Read-only published evaluation, with separate runtime verification."""
import json


def get_forecast_evidence(base_dir, checkpoint_dir, service):
    evidence = json.loads((base_dir / "web/data/forecast_evidence.json").read_text(encoding="utf-8"))
    loaded = getattr(service, "forecast_bundle", None)
    checksum = getattr(service, "forecast_bundle_sha256", None) if loaded is not None else None
    verified = bool(checksum and checksum == evidence["forecast_bundle_sha256"])
    graph = getattr(service, "graph_predictor", None)
    if graph is not None and graph.graph.get("graph_weight", 0) > 0:
        verified = False  # This manifest evaluates the selected tabular predictor.
    evidence["serving"] = {
        "verified": verified,
        "status": "matching_evaluated_bundle" if verified else "different_or_unavailable_bundle",
        "model_name": loaded.get("model_name") if loaded else "legacy_or_unavailable",
        "graph_context_available": getattr(service, "graph_predictor", None) is not None,
    }
    return evidence
