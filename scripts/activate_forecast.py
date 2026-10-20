"""Install a reviewed local forecast bundle, backing up existing artifacts.

Does not deploy to Railway. Graph sidecar is optional and requires PyTorch.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def activate(bundle_path, graph_path=None, checkpoint_dir=ROOT / "ml/gnn/checkpoints"):
    bundle = joblib.load(bundle_path)
    if bundle.get("schema_version") != 3:
        raise ValueError("Unsupported model schema")
    report = json.loads((bundle_path.parent / "evaluation.json").read_text())
    graph = joblib.load(graph_path) if graph_path else None
    if graph and graph["tabular_sha256"] != hashlib.sha256(bundle_path.read_bytes()).hexdigest():
        raise ValueError("Graph bundle and tabular bundle do not match")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backup = checkpoint_dir / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for name in ["forecast_v3.joblib", "citation_forecast.joblib", "FORECAST_CARD.json"]:
        existing = checkpoint_dir / name
        if existing.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, backup / name)
    shutil.copy2(bundle_path, checkpoint_dir / "forecast_v3.joblib")
    if graph_path:
        shutil.copy2(graph_path, checkpoint_dir / "citation_forecast.joblib")
    elif (checkpoint_dir / "citation_forecast.joblib").exists():
        (checkpoint_dir / "citation_forecast.joblib").unlink()
    uses_graph = bool(graph and graph["graph_weight"] > 0)
    card = dict(model_name="CitationSAGE_validation_selected_fusion" if uses_graph else bundle["model_name"],
                architecture="One-hop relational citation GraphSAGE with validation-selected tabular fusion" if uses_graph else "Early-trajectory tabular classifier; citation graph is experimental context",
                feature_contract=bundle["features"].version,
                data_audit=bundle["audit"],
                evaluation=graph["report"] if graph else report["development_test"],
                strict_temporal_evaluation=report.get("strict_temporal_test"),
                evaluation_status="Development benchmark; no claim of fresh confirmation or 80% accuracy",
                limitations=["Citation impact does not establish scientific quality", "Sample-relative cohort labels", "New cohorts and sparse graphs require caution"],
                installed_at=datetime.now(timezone.utc).isoformat())
    (checkpoint_dir / "FORECAST_CARD.json").write_text(json.dumps(card, indent=2))
    print("Installed local model:", card["model_name"])
    print("Checkpoint directory:", checkpoint_dir)
    if backup.exists():
        print("Previous local artifacts backed up:", backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--graph", type=Path)
    args = parser.parse_args()
    activate(args.bundle, args.graph)
