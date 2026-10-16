# legacy_archive — Obsolete and Fabricated Files

**Do not cite any numbers or code in this directory.**

These files were archived because they:
1. Contained fabricated or hardcoded numbers that contradicted the real benchmark results
2. Labelled non-GNN sklearn models as "HeteroGraphSAGE" (fraud risk)
3. Generated fake terminal screenshots with hand-typed test names that don't exist
4. Described a 50-paper n=5 toy experiment as the real evaluation
5. Were superseded by the honest benchmark in `reports/benchmark_latest.json`

## Real Results (from benchmark_latest.json, test set of 4,500 papers from 2018-2020)

| Model | Macro-F1 | AUROC |
|---|---|---|
| M0_Majority (always guess majority) | 0.2215 | 0.500 |
| M1_LogisticRegression (BEST) | 0.5244 | 0.714 |
| M2_LightGBM | 0.5020 | 0.718 |
| M3_GraphlessMLP | 0.4111 | 0.671 |
| M4_HomogeneousGNN | 0.3138 | 0.513 |
| M5_HeteroGraphSAGE | 0.3167 | 0.517 |

The GNN did NOT outperform logistic regression. This is an honest negative result.
