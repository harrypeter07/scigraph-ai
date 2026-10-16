> [!CAUTION]
> **OBSOLETE — v1 prototype (50 papers). Do not cite.**
>
> This document describes the deprecated v1 prototype (50 papers, 5 test papers, toy features) and contradicts authoritative v2 results. For authentic, verified results based on the 16,500 paper corpus and 4,500 paper test split, refer to [README.md](../../README.md), [reports/benchmark_latest.json](../../reports/benchmark_latest.json), and [reports/heldout_predictions_summary.json](../../reports/heldout_predictions_summary.json).

---

# Experimental Results & Temporal Leakage Audit Summary

> **Document Status**: Live Benchmarks Logged  
> **Primary Claim**: Standard non-temporal graph learning evaluations suffer from temporal leakage, inflating predictive performance metrics compared to time-consistent evaluation protocols.

---

## 1. Summary of Benchmark Performances

| Model Architecture | Evaluation Condition | Split Strategy | Accuracy | Macro-F1 | Provenance Log |
|---|---|---|---|---|---|
| **Logistic Regression** | Tabular Baseline | Time-Consistent ($2012-2018 / 2020-2021$) | 0.6000 | 0.3750 | `docs/EXPERIMENTS.md` |
| **Gradient Boosted Trees** | Tabular Baseline | Time-Consistent ($2012-2018 / 2020-2021$) | 0.6000 | 0.3750 | `docs/EXPERIMENTS.md` |
| **HeteroGraphSAGE (CPU Smoke)** | GNN Baseline | Time-Consistent ($2012-2018 / 2020-2021$) | 0.6800 | 0.6200 | `ml/gnn/checkpoints/` |
| **HeteroGraphSAGE (Lab GPU)** | Full Graph GNN | Time-Consistent ($2012-2018 / 2020-2021$) | *Pending Lab GPU Run* | *Pending Lab GPU Run* | `docs/HOW_TO_RUN_ON_COLLEGE_LAB.md` |

---

## 2. Key Findings & Deliverables

1. **Strict Temporal Snapshot Engine**: Successfully implemented zero-leakage snapshotting using `counts_by_year` annual trajectory parsing rather than raw aggregate counts.
2. **Heterogeneous Graph Infrastructure**: Built multi-relational PyTorch Geometric graph pipeline connecting 50 papers, 257 authors, 215 institutions, and 90 time-consistent citation edges.
3. **Lab PC Portability**: Prepared single-command GPU training workflow (`python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda`) for execution in the college GPU laboratory.
