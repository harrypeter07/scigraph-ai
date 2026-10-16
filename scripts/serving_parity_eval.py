"""WP3: Serving Parity Evaluation Script.

Measures the gap between transductive benchmark GNN evaluation and
deployed inference_service.predict() on 500 test-split works.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference_service import inference_service
from ml.gnn.graph_builder import build_heterogeneous_graph
from ml.gnn.models.hetero_gnn import AuthenticHeteroSAGE
import torch


def evaluate_serving_parity(n_samples: int = 500, random_state: int = 42):
    print("=" * 70)
    print("WP3: SERVING PARITY EVALUATION (500 Test Works)")
    print("=" * 70)

    # 1. Initialize inference service
    inference_service.load()
    df_works = inference_service.df_works
    assert df_works is not None, "df_works not loaded"

    test_works = df_works[df_works["temporal_split"] == "test"].sample(
        n=n_samples, random_state=random_state
    ).reset_index(drop=True)

    print(f"Loaded {len(test_works)} test-split works for serving evaluation.")
    y_true = test_works["impact_class"].values.astype(int)

    # 2. Evaluate through full transductive GNN graph (benchmark mode)
    print("\nEvaluating benchmark transductive graph mode...")
    graph = build_heterogeneous_graph()
    model = inference_service.model
    model.eval()

    with torch.no_grad():
        out_bench = model(graph.x_dict, graph.edge_index_dict)
        test_mask = graph["paper"].test_mask
        bench_probs = out_bench["probabilities"][test_mask].numpy()
        bench_preds_all = np.argmax(bench_probs, axis=1)

    # Filter to the 500 sampled indices in test split
    test_indices_all = np.where(graph["paper"].test_mask.numpy())[0]
    sample_work_ids = set(test_works["work_id"])
    id_to_sample_idx = {wid: i for i, wid in enumerate(test_works["work_id"])}

    sample_bench_preds = np.zeros(len(test_works), dtype=int)
    for i, g_idx in enumerate(test_indices_all):
        wid = df_works.iloc[g_idx]["work_id"]
        if wid in sample_work_ids:
            s_idx = id_to_sample_idx[wid]
            sample_bench_preds[s_idx] = int(bench_preds_all[i])

    bench_f1 = float(f1_score(y_true, sample_bench_preds, average="macro"))
    bench_acc = float(accuracy_score(y_true, sample_bench_preds))
    print(f"Benchmark Graph Mode (500 test works): Macro-F1 = {bench_f1:.4f}, Accuracy = {bench_acc:.4f}")

    # 3. Evaluate each paper individually through inference_service.predict()
    print("\nEvaluating serving mode (predict() per paper)...")
    serving_preds = []
    serving_contexts = []
    t0 = time.time()

    for idx, row in test_works.iterrows():
        raw_dict = row.to_dict()
        res = inference_service.predict(raw_dict)
        serving_preds.append(int(res["predicted_impact_class"]))
        serving_contexts.append(res.get("graph_context", "none"))

    elapsed = time.time() - t0
    serving_preds = np.array(serving_preds)
    serving_f1 = float(f1_score(y_true, serving_preds, average="macro"))
    serving_acc = float(accuracy_score(y_true, serving_preds))
    gap_f1 = bench_f1 - serving_f1

    print(f"Serving Mode (500 test works): Macro-F1 = {serving_f1:.4f}, Accuracy = {serving_acc:.4f} (latency: {elapsed/n_samples*1000:.1f}ms/paper)")
    print(f"Parity Gap (Benchmark F1 - Serving F1): {gap_f1:+.4f}")

    results = {
        "n_samples": n_samples,
        "benchmark_macro_f1": round(bench_f1, 4),
        "benchmark_accuracy": round(bench_acc, 4),
        "serving_macro_f1": round(serving_f1, 4),
        "serving_accuracy": round(serving_acc, 4),
        "parity_gap_macro_f1": round(gap_f1, 4),
        "context_distribution": pd.Series(serving_contexts).value_counts().to_dict(),
        "latency_ms_per_paper": round(elapsed / n_samples * 1000, 2),
    }

    out_file = Path("reports/serving_parity_gap.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport written to {out_file}")
    return results


if __name__ == "__main__":
    evaluate_serving_parity()
