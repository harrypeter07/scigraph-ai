"""Density Evaluation and Gate A Validation for Dense Corpus (Phase A).

Checks the 3 Gate A criteria:
1. Median degree on target papers >= 15
2. Fraction of target papers with >= 3 in-corpus neighbors >= 70%
3. Citation-reference coverage (fraction of reference list in-corpus) median >= 0.30

Outputs reports/dense_corpus_stats.json and prints full distribution table.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DENSE_DATA_DIR = REPO_ROOT / "data" / "dense"
REPORTS_DIR = REPO_ROOT / "reports"


def evaluate_dense_corpus(data_dir: Path = DENSE_DATA_DIR) -> Dict[str, Any]:
    print("=" * 85)
    print("SCIGRAPH AI 2.0 — PHASE A: DENSE CORPUS DENSITY EVALUATION & GATE A")
    print("=" * 85)

    works_path = data_dir / "works.parquet"
    edges_path = data_dir / "edges.parquet"
    raw_refs_path = data_dir / "raw_references.parquet"

    if not works_path.exists() or not edges_path.exists():
        print(f"[ERROR] Required dense corpus tables not found in {data_dir}")
        sys.exit(1)

    print("Loading parquet tables...")
    df_works = pd.read_parquet(works_path)
    df_edges = pd.read_parquet(edges_path)
    df_raw_refs = pd.read_parquet(raw_refs_path) if raw_refs_path.exists() else None

    total_works = len(df_works)
    total_edges = len(df_edges)
    print(f"Total Works: {total_works:,}")
    print(f"Total In-Corpus Edges: {total_edges:,}")

    # Target papers are works published 2012-2020 (used for modeling cohorts with 5y follow-up)
    target_mask = (df_works["publication_year"] >= 2012) & (df_works["publication_year"] <= 2020)
    df_targets = df_works[target_mask].copy()
    target_wids = set(df_targets["work_id"])
    all_wids = set(df_works["work_id"])
    print(f"Target Cohort Works (2012–2020): {len(df_targets):,}")

    # Calculate degrees on target papers
    # Out-degree: references to other in-corpus works
    # In-degree: citations received from other in-corpus works
    out_counts = df_edges.groupby("source_work_id").size().to_dict()
    in_counts = df_edges.groupby("target_work_id").size().to_dict()

    target_degrees = []
    target_in_degrees = []
    target_out_degrees = []
    target_ref_coverages = []

    # Map raw reference counts if available
    raw_ref_counts = {}
    if df_raw_refs is not None:
        raw_ref_counts = df_raw_refs.groupby("source_work_id").size().to_dict()

    for wid in df_targets["work_id"]:
        ind = in_counts.get(wid, 0)
        outd = out_counts.get(wid, 0)
        totd = ind + outd
        target_in_degrees.append(ind)
        target_out_degrees.append(outd)
        target_degrees.append(totd)

        # Coverage = in-corpus out-degree / total raw references
        total_raw = raw_ref_counts.get(wid, 0)
        if total_raw > 0:
            cov = min(1.0, outd / total_raw)
            target_ref_coverages.append(cov)
        else:
            if outd > 0:
                target_ref_coverages.append(1.0)
            else:
                target_ref_coverages.append(0.0)

    target_degrees = np.array(target_degrees)
    target_in_degrees = np.array(target_in_degrees)
    target_out_degrees = np.array(target_out_degrees)
    target_ref_coverages = np.array(target_ref_coverages)

    # Compute key statistics
    median_degree = float(np.median(target_degrees))
    mean_degree = float(np.mean(target_degrees))
    p25_degree = float(np.percentile(target_degrees, 25))
    p75_degree = float(np.percentile(target_degrees, 75))
    p90_degree = float(np.percentile(target_degrees, 90))

    median_in_degree = float(np.median(target_in_degrees))
    median_out_degree = float(np.median(target_out_degrees))

    frac_ge_3 = float(np.mean(target_degrees >= 3))
    frac_ge_5 = float(np.mean(target_degrees >= 5))
    frac_ge_10 = float(np.mean(target_degrees >= 10))

    median_ref_cov = float(np.median(target_ref_coverages))
    mean_ref_cov = float(np.mean(target_ref_coverages))

    # Gate A checks
    gate_median_deg_pass = median_degree >= 15.0
    gate_ge_3_pass = frac_ge_3 >= 0.70
    gate_ref_cov_pass = median_ref_cov >= 0.30

    gate_a_passed = gate_median_deg_pass and gate_ge_3_pass and gate_ref_cov_pass

    print("\n" + "-" * 85)
    print(f"GATE A DENSITY METRICS (Target Cohort N={len(df_targets):,}):")
    print("-" * 85)
    print(f"1. Target Median Degree:        {median_degree:.1f} (Mean: {mean_degree:.1f}, IQR: [{p25_degree:.0f}, {p75_degree:.0f}])")
    print(f"   - Target Median In-Degree:   {median_in_degree:.1f}")
    print(f"   - Target Median Out-Degree:  {median_out_degree:.1f}")
    print(f"   - Gate Requirement: >= 15.0 -> [{'PASS' if gate_median_deg_pass else 'FAIL'}]")
    print()
    print(f"2. Fraction with >= 3 Neighbors: {frac_ge_3*100:.1f}% (with >= 5: {frac_ge_5*100:.1f}%, >= 10: {frac_ge_10*100:.1f}%)")
    print(f"   - Gate Requirement: >= 70.0% -> [{'PASS' if gate_ge_3_pass else 'FAIL'}]")
    print()
    print(f"3. In-Corpus Reference Coverage: {median_ref_cov*100:.1f}% median ({mean_ref_cov*100:.1f}% mean)")
    print(f"   - Gate Requirement: >= 30.0% -> [{'PASS' if gate_ref_cov_pass else 'FAIL'}]")
    print("-" * 85)
    print(f"GATE A OVERALL VERDICT: [{'GATE A PASSED - PROCEED TO GNN' if gate_a_passed else 'GATE A FAILED - GRAPH TOO SPARSE'}]")
    print("=" * 85)

    stats = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_dir": str(data_dir),
        "total_works": total_works,
        "total_in_corpus_edges": total_edges,
        "target_works_count": len(df_targets),
        "target_degrees": {
            "median": median_degree,
            "mean": round(mean_degree, 2),
            "p25": p25_degree,
            "p75": p75_degree,
            "p90": p90_degree,
            "median_in_degree": median_in_degree,
            "median_out_degree": median_out_degree,
        },
        "neighbor_fractions": {
            "ge_1": round(float(np.mean(target_degrees >= 1)), 4),
            "ge_3": round(frac_ge_3, 4),
            "ge_5": round(frac_ge_5, 4),
            "ge_10": round(frac_ge_10, 4),
        },
        "reference_coverage": {
            "median": round(median_ref_cov, 4),
            "mean": round(mean_ref_cov, 4),
        },
        "gate_a": {
            "median_degree_ge_15": gate_median_deg_pass,
            "fraction_ge_3_ge_70pct": gate_ge_3_pass,
            "median_ref_cov_ge_30pct": gate_ref_cov_pass,
            "overall_pass": gate_a_passed,
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "dense_corpus_stats.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved Gate A density report to: {report_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate dense corpus density and Gate A")
    parser.add_argument("--data_dir", type=str, default=str(DENSE_DATA_DIR), help="Path to dense data directory")
    args = parser.parse_args()

    evaluate_dense_corpus(Path(args.data_dir))


if __name__ == "__main__":
    main()
