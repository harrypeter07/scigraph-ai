"""Evaluate real citation message passing and validation-selected late fusion.

Requires the tabular bundle from train_forecast.py. Test remains a development
benchmark. Run this on Colab GPU for larger corpora and multiple seeds.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_forecast import load_records, metrics
from ml.features.forecast import probabilities
from ml.gnn.citation_forecast import CitationNeighborhoods, make_model


def train(data_dir, bundle_path, output_dir, epochs=100, seeds=(42, 43, 44), without_edges=False):
    torch.set_num_threads(4)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(data_dir)
    groups = [[r for r in records if lo <= int(r["publication_year"]) <= hi and r.get("impact_class") in (0, 1, 2)]
              for lo, hi in [(2012, 2016), (2017, 2017), (2018, 2020)]]
    train_rows, validation, test_rows = groups
    selection, calibration = train_test_split(validation, test_size=.5, random_state=42, stratify=[r["impact_class"] for r in validation])
    rows = train_rows + selection + calibration + test_rows
    lengths = list(map(len, [train_rows, selection, calibration, test_rows]))
    boundaries = np.cumsum([0]+lengths)
    slices = [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    bundle = joblib.load(bundle_path)
    edges = pd.read_parquet(data_dir / "citation_edges.parquet")
    graph_records = records
    if (data_dir / "neighbor_works.parquet").exists():
        extra = pd.read_parquet(data_dir / "neighbor_works.parquet").to_dict("records")
        history_map = {}
        from ml.features.forecast import clean_id
        for item in pd.read_parquet(data_dir / "neighbor_counts_by_year.parquet").to_dict("records"):
            history_map.setdefault(clean_id(item["work_id"]), []).append({"year": int(item["year"]), "cited_by_count": int(item["cited_by_count"])})
        for item in extra:
            item["counts_by_year"] = history_map.get(clean_id(item["work_id"]), [])
        graph_records = records + extra
        edges = pd.concat([edges, pd.read_parquet(data_dir / "neighbor_edges.parquet")]).drop_duplicates()
    neighborhoods = CitationNeighborhoods(graph_records, edges[["citing_work_id", "referenced_work_id"]].itertuples(index=False, name=None))
    graph_rows = [neighborhoods.build(r) for r in rows]
    neighbors = np.stack([r[0] for r in graph_rows])
    masks = np.stack([r[1] for r in graph_rows])
    scaler = StandardScaler().fit(bundle["features"].transform(train_rows))
    roots = scaler.transform(bundle["features"].transform(rows)).astype(np.float32)
    # Scaling is fitted ONLY on training neighbors and valid neighbor slots.
    valid_training = neighbors[slices[0]][masks[slices[0]].astype(bool)]
    if not len(valid_training):
        raise ValueError("No actual in-corpus training citation neighbors; acquire graph coverage first")
    neighbor_scaler = StandardScaler().fit(valid_training)
    neighbors = neighbor_scaler.transform(neighbors.reshape(-1, 8)).reshape(neighbors.shape).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, n, m = [torch.tensor(a, device=device) for a in [roots, neighbors, masks]]
    if without_edges:
        m = torch.zeros_like(m)
    y = torch.tensor([int(r["impact_class"]) for r in rows], device=device)
    selected_states, seed_results, predictions, ablations = [], [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        generator = np.random.default_rng(seed)
        model = make_model(roots.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        best_score, patience = -1, 0
        for epoch in range(epochs):
            model.train()
            permutation = generator.permutation(lengths[0])
            for start in range(0, lengths[0], 256):
                idx = torch.tensor(permutation[start:start+256], device=device)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x[idx], n[idx], m[idx]), y[idx])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(x[slices[1]], n[slices[1]], m[slices[1]]), dim=1).cpu().numpy()
            score = metrics(y[slices[1]].cpu().numpy(), p)["accuracy"]
            if score > best_score:
                best_score, patience = score, 0
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            else:
                patience += 1
            if patience >= 15:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            predictions.append(torch.softmax(model(x, n, m), dim=1).cpu().numpy())
            ablations.append(torch.softmax(model(x, n, torch.zeros_like(m)), dim=1).cpu().numpy())
        selected_states.append({k: v.cpu() for k, v in best_state.items()})
        seed_results.append(dict(seed=seed, selection_accuracy=best_score, epoch=best_epoch))
        print(seed_results[-1], flush=True)
    graph_p = np.mean(predictions, axis=0)
    no_edges_p = np.mean(ablations, axis=0)
    tab_p = probabilities(bundle, rows)
    ys = y.cpu().numpy()
    weights = [0., .1, .25, .5, .75, 1.]
    scores = [(metrics(ys[slices[1]], (1-w)*tab_p[slices[1]]+w*graph_p[slices[1]])["accuracy"], w) for w in weights]
    # Prefer the simpler tabular model on ties.
    weight = sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]
    combined = (1-weight)*tab_p + weight*graph_p
    logits = np.log(np.clip(combined[slices[2]], 1e-12, 1))
    temperature = 1.0 if weight == 0 else float(minimize_scalar(lambda t: log_loss(ys[slices[2]], softmax(logits/t, axis=1), labels=[0, 1, 2]), bounds=(.2, 5), method="bounded").x)
    combined = softmax(np.log(np.clip(combined, 1e-12, 1))/temperature, axis=1)
    no_edges_combined = softmax(np.log(np.clip((1-weight)*tab_p+weight*no_edges_p, 1e-12, 1))/temperature, axis=1)
    graph_correct = combined[slices[3]].argmax(axis=1) == ys[slices[3]]
    baseline_correct = tab_p[slices[3]].argmax(axis=1) == ys[slices[3]]
    difference = graph_correct.astype(float) - baseline_correct.astype(float)
    rng = np.random.default_rng(42)
    lift_ci = np.quantile([difference[rng.integers(0, len(difference), len(difference))].mean() for _ in range(1000)], [.025, .975]).tolist()
    report = dict(status="development benchmark; not fresh confirmation", seeds=seed_results,
                  without_edges_control=without_edges,
                  accuracy_lift_over_tabular=float(difference.mean()), paired_bootstrap_lift_ci95=lift_ci,
                  selection_weights=scores, selected_graph_weight=weight, fusion_temperature=temperature,
                  development_tabular=metrics(ys[slices[3]], tab_p[slices[3]]),
                  development_gnn=metrics(ys[slices[3]], graph_p[slices[3]]),
                  development_gnn_removed_edges=metrics(ys[slices[3]], no_edges_p[slices[3]]),
                  development_fusion=metrics(ys[slices[3]], combined[slices[3]]),
                  development_removed_edges=metrics(ys[slices[3]], no_edges_combined[slices[3]]),
                  graph_coverage={name: float((masks[sl].sum(axis=(1, 2))>0).mean()) for name, sl in zip(["train", "selection", "calibration", "development_test"], slices)},
                  architecture="one-hop incoming/outgoing relational GraphSAGE; target-cutoff features; no self-loop substitutes")
    strict_mask = np.array([2018 <= int(r["publication_year"]) <= 2020 and int(r["publication_year"])+2 >= bundle.get("label_availability_year", 2022) for r in rows])
    if strict_mask.any():
        report["strict_temporal_gnn"] = metrics(ys[strict_mask], graph_p[strict_mask])
        report["strict_temporal_fusion"] = metrics(ys[strict_mask], combined[strict_mask])
        report["strict_temporal_n"] = int(strict_mask.sum())
    import hashlib
    joblib.dump(dict(states=selected_states, root_scaler=scaler, neighbor_scaler=neighbor_scaler,
                     neighborhoods=neighborhoods, graph_weight=weight, temperature=temperature,
                     report=report, root_dim=roots.shape[1], tabular_sha256=hashlib.sha256(bundle_path.read_bytes()).hexdigest()), output_dir / "citation_gnn_experiment.joblib")
    (output_dir / "graph_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"graph_weight": weight, "fusion_accuracy": report["development_fusion"]["accuracy"], "graph_coverage": report["graph_coverage"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--bundle", type=Path, default=ROOT / "reports/forecast_v3_cohort/forecast_v3.joblib")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/citation_forecast")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--without-edges", action="store_true")
    args = parser.parse_args()
    train(args.data_dir, args.bundle, args.output_dir, args.epochs, args.seeds, args.without_edges)
