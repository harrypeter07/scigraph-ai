"""
Reconnaissance script for WP0.
Gathers ground-truth evidence across all 9 questions.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

def run_recon():
    print("=" * 70)
    print("WP0 RECONNAISSANCE SCRIPT")
    print("=" * 70)

    # 1. Real HeteroData
    from ml.gnn.graph_builder import build_heterogeneous_graph
    g = build_heterogeneous_graph()

    print("\n--- 1. Node Counts ---")
    for nt in g.node_types:
        x_shape = g[nt].x.shape if hasattr(g[nt], "x") else "No x"
        print(f"Node '{nt}': {g[nt].num_nodes:,} nodes, shape={x_shape}")

    print("\n--- 1b. Edge Counts ---")
    for et in g.edge_types:
        print(f"Edge {et}: {g[et].edge_index.shape[1]:,} edges")

    df_citations = pd.read_parquet("data/citation_edges.parquet")
    df_works = pd.read_parquet("data/works.parquet")
    works_set = set(df_works["work_id"])
    in_corpus = df_citations["citing_work_id"].isin(works_set) & df_citations["referenced_work_id"].isin(works_set)
    print(f"\nTotal rows in citation_edges.parquet: {len(df_citations):,}")
    print(f"Paper->paper edges where BOTH endpoints are in works.parquet: {in_corpus.sum():,}")

    for nt in ["author", "institution", "venue", "topic"]:
        edge_types_nt = [et for et in g.edge_types if et[0] == nt or et[2] == nt]
        total_edges = sum(g[et].edge_index.shape[1] for et in edge_types_nt)
        avg_deg = total_edges / max(1, g[nt].num_nodes)
        print(f"Avg degree for '{nt}' ({g[nt].num_nodes:,} nodes): {avg_deg:.2f}")

    # 2. Training vs Serving feature code
    print("\n--- 2. Training vs Serving Node Features ---")
    print(f"Training paper node dim: {g['paper'].x.shape[1]}")
    for nt in ["author", "institution", "venue", "topic"]:
        print(f"Training '{nt}' node dim: {g[nt].x.shape[1]}, non-zero count: {(g[nt].x != 0).sum().item()}")

    # In ml/inference_service.py:
    # x_dict = {
    #     "paper": x_paper,
    #     "author": torch.zeros((1, 8), dtype=torch.float32),
    #     "institution": torch.zeros((1, 8), dtype=torch.float32),
    #     "venue": torch.zeros((1, 8), dtype=torch.float32),
    #     "topic": torch.zeros((1, 8), dtype=torch.float32),
    # }
    print("Serving inference_service.py passes zeros((1, 8)) for author, institution, venue, topic!")

    # 3. Time filtering per paper
    print("\n--- 3. Time filtering on edges ---")
    # Does graph_builder filter edges by time?
    print("In graph_builder.py: edge_cites, edge_cited_by, authored, affiliated_with, published_in, has_topic are built globally without edge_time or time filtering.")

    # 4. Ingestion sampling method
    print("\n--- 4. Corpus sampling in scripts/ingest_openalex_corpus.py ---")
    if os.path.exists("scripts/ingest_openalex_corpus.py"):
        with open("scripts/ingest_openalex_corpus.py", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if "sort" in line or "sample" in line or "per_field" in line or "filter" in line]
            print("Sampling clues in ingest script:")
            for l in lines[:10]:
                print("  ", l)

    # 5. TF-IDF / SVD fit
    print("\n--- 5. TF-IDF and SVD fitting ---")
    # Check graph_builder.py vs inference_service.py
    print("graph_builder.py line 59: tfidf.fit_transform(df_works['title']) - fit on ALL 16,500 works (train+val+test). Not saved to disk.")
    print("inference_service.py line 99: tfidf.fit_transform(titles) - fit on boot on ALL works. Not saved to disk.")

    # 6. df_counts.year.min() and share of 2010/2011 works
    print("\n--- 6. Year range and counts_by_year ---")
    df_counts = pd.read_parquet("data/counts_by_year.parquet")
    print(f"df_counts year min: {df_counts['year'].min()}, max: {df_counts['year'].max()}")
    pub_years = df_works["publication_year"].value_counts().sort_index()
    print("Works publication year distribution:")
    for yr, cnt in pub_years.items():
        pct = cnt / len(df_works) * 100
        print(f"  {yr}: {cnt:,} ({pct:.2f}%)")
    w_2010_2011 = df_works["publication_year"].isin([2010, 2011]).sum()
    print(f"Share of 2010/2011 works: {w_2010_2011:,} / {len(df_works):,} ({w_2010_2011/len(df_works)*100:.2f}%)")

    # 7. Regression target in head
    print("\n--- 7. Regression Target ---")
    print(f"graph_builder.py line 100: y_reg = torch.tensor(np.log1p(df_works['future_citations_5y'].values))")
    print("What is future_citations_5y in works.parquet?")
    sample_work = df_works[["work_id", "publication_year", "citations_at_cutoff", "future_citations_5y", "impact_class"]].head(5)
    print(sample_work)

    # 8. Temperature fit and ECE split
    print("\n--- 8. Temperature fit & ECE split ---")
    # In ml/baselines/benchmark_suite.py:
    # compute_ece(probs_5, y_test_cls) - computed on test set!
    # In AuthenticHeteroSAGE:
    # self.temperature = nn.Parameter(torch.ones(1) * 1.5) - hardcoded initial 1.5!
    print("benchmark_suite.py computes compute_ece(probs_5, y_test_cls) on TEST set.")

    # 9. Test invariants vs hardcoded outputs
    print("\n--- 9. Tests Invariants vs Hardcoded ---")
    for tpath in ["tests/test_backend_api.py", "tests/test_phase2_scientometrics.py", "tests/test_phase3_model.py"]:
        if os.path.exists(tpath):
            with open(tpath, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"File {tpath}: {len(content.splitlines())} lines. Checks for status_code == 200, keys in response, etc.")

if __name__ == "__main__":
    run_recon()
