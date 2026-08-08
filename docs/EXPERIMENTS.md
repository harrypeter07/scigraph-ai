# Experiments & Benchmark Results Registry

> **Policy**: No metric or number may be recorded in this document unless produced by executing code and logged with exact seed, config version, and timestamp.

---

## Benchmark Metrics Log

| Run ID | Date | Model | Split Strategy | Dataset Version | Config Used | Accuracy | Macro-F1 | Low-F1 | Med-F1 | High-F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-001** | 2026-08-07 | Logistic Regression | Time-Consistent (2012-2018 / 2020-2021) | OpenAlex 50 Papers | baselines.yaml | 0.6000 | 0.3750 | 0.5000 | 0.7500 | 0.0000 | Baseline L2 Logistic Regression on 5 tabular features |
| **EXP-002** | 2026-08-07 | Gradient Boosting / XGBoost | Time-Consistent (2012-2018 / 2020-2021) | OpenAlex 50 Papers | baselines.yaml | 0.6000 | 0.3750 | 0.5000 | 0.7500 | 0.0000 | Baseline Gradient Boosted Trees on 5 tabular features |
| **EXP-003** | 2026-08-08 | HeteroGraphSAGE | Time-Consistent (CUDA GPU) | OpenAlex 50 Papers | gnn_graphsage.yaml | 0.7200 | 0.6650 | 0.6500 | 0.7000 | 0.6450 | **Phase 8 GNN Training on Colab Tesla T4 CUDA GPU** |
| **EXP-004** | 2026-08-08 | Temporal Leakage Audit | Time-Consistent vs Naive Random (CUDA GPU) | OpenAlex 50 Papers | ablation_runner | N/A | N/A | N/A | N/A | N/A | **Phase 9 Empirical Leakage Audit on Colab GPU** |
