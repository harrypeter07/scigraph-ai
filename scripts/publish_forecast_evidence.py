"""Publish a small UI evidence manifest from saved evaluation artifacts."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def publish():
    report = json.loads((ROOT / "reports/expanded_forecast/evaluation.json").read_text())
    graph = json.loads((ROOT / "reports/expanded_citation_enriched/graph_evaluation.json").read_text())
    strict = report["strict_temporal_test"]
    baseline_accuracy = strict["majority_baseline_accuracy"]
    import joblib
    bundle_path = ROOT / "ml/gnn/checkpoints/forecast_v3.joblib"
    bundle = joblib.load(bundle_path)
    examples = []
    for name in ["Computer Science", "Medicine", "Engineering"]:
        rows = [r for r in bundle["evaluation_reference"] if r["publication_year"] == 2020 and r["primary_field_name"] == name]
        row = sorted(rows, key=lambda r: r["work_id"])[0]
        examples.append({key: row[key] for key in ["work_id", "title", "publication_year", "primary_field_name"]})
    manifest = {
        "schema_version": 1,
        "evaluated_date": "2026-10-20",
        "model_name": report["selected_model"],
        "model_display_name": "Calibrated gradient boosting",
        "forecast_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "primary": strict,
        "retrospective_pooled": {key: report["development_test"][key] for key in ["accuracy", "macro_f1", "accuracy_ci95"]},
        "baseline": {"accuracy": baseline_accuracy, "macro_f1": 2*baseline_accuracy/(1+baseline_accuracy)/3},
        "gnn": {"accuracy": graph["strict_temporal_gnn"]["accuracy"], "macro_f1": graph["strict_temporal_gnn"]["macro_f1"],
                "selected_weight": graph["selected_graph_weight"], "coverage_pooled": graph["graph_coverage"]["development_test"],
                "architecture": "One-hop citation GraphSAGE", "seeds": len(graph["seeds"])},
        "dataset": {"total": sum(report["audit"]["sizes"].values()), **report["audit"]["sizes"],
                    "sampling": "Seeded random samples across five fields, 2012–2020. Original-corpus IDs excluded."},
        "limitations": [
            "Retrospective OpenAlex snapshots; prospective external validation remains outstanding.",
            "Percentile tiers describe sampled field/year peers, not scientific quality or population-wide discipline percentiles.",
            "Training and calibration labels mature by 2022; earlier forecast dates are flagged retrospective.",
            "Accuracy varies by field. A single paper's probability is not the test-set accuracy."
        ],
        "examples": examples,
        "source_reports": ["reports/expanded_forecast/evaluation.json", "reports/expanded_citation_enriched/graph_evaluation.json"]
    }
    destination = ROOT / "web/data/forecast_evidence.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)
    return manifest


if __name__ == "__main__":
    publish()
