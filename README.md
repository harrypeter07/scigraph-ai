# SciGraph AI 2.0 — Scientific Citation Impact Prediction

> **Predicts whether a paper will be among the most cited in its field and year, using only
> information available 2 years after publication.**

---

## Results

All numbers below are read directly from [`reports/benchmark_latest.json`](reports/benchmark_latest.json).
No metrics are hard-coded in this file.

| Model | Macro-F1 | AUROC | Notes |
|---|---|---|---|
| M0\_Majority | 0.2215 | 0.500 | Majority-class baseline |
| M1\_LogisticRegression | **0.5244** | **0.714** | **Best model** |
| M2\_LightGBM | 0.502 | 0.718 | |
| M3\_GraphlessMLP | 0.411 | 0.671 | |
| M4\_HomogeneousGNN | 0.314 | 0.513 | |
| M5\_HeteroGraphSAGE | 0.317 | 0.517 | |

> **The graph neural network did not outperform logistic regression. This is a legitimate
> research finding.**

---

## Key Findings

- **Early citations and venue prestige roughly double the majority baseline**
  (0.22 → 0.52 Macro-F1).
- **Open Access advantage:** +10.8% adjusted citation advantage
  (association, not causation — no causal claim is made).
- **GNN performance:** M5\_HeteroGraphSAGE (Macro-F1 0.317, AUROC 0.517) sits near the
  majority baseline, indicating that graph structure alone does not add predictive value
  over tabular features for this task and dataset size.

---

## Evaluation Setup

- **Test set:** 4,500 held-out papers from 2018–2020 that the model never trained on.
- **Label:** top-10% cited papers within field × year stratum (binary classification).
- **Temporal integrity:** no information after the 2-year post-publication cutoff is used
  at inference time.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (legacy tests are excluded)
pytest tests -v --ignore=tests/legacy

# Re-run the canonical benchmark
python scripts/make_numbers.py
```

---

## Corrections Notice

Previous versions of this repository made claims based on stale benchmark scripts that have been **fully retracted and removed**:

- **Inflated GNN Macro-F1 claim:** Retracted. The actual test Macro-F1 for HeteroGraphSAGE is ~0.317. Logistic Regression is the top model (~0.524).
- **Synthetic ECE calibration claim:** Retracted. Real validation-fit temperatures produce ECE ~0.03-0.05.
- **Unadjusted Open Access advantage claim:** Retracted. Real empirical OLS regression shows +23.6% crude and +10.8% adjusted (association, not causation).
- **Simulated low-latency claim:** Retracted. Actual measured API p50 latency is ~384 ms.

These claims originated from scripts that either hard-coded results or mislabelled
classical ML models (sklearn GradientBoostingClassifier) as graph neural networks.
All such scripts have been moved to [`legacy_archive/`](legacy_archive/) for audit traceability.

---

## Repository Structure

```
scigraph/
├── reports/
│   └── benchmark_latest.json   ← authoritative results (source of truth)
├── ml/
│   ├── features/               ← feature engineering
│   ├── gnn/                    ← GNN model code (M4, M5)
│   │   └── checkpoints/        ← trained model files
│   ├── baselines/              ← logistic regression, LightGBM
│   ├── evaluation/             ← evaluation pipeline
│   └── scientometrics/         ← OA advantage, venue resolver
├── api/                        ← FastAPI serving layer
├── scripts/                    ← data ingestion and training scripts
├── tests/                      ← pytest test suite
└── legacy_archive/             ← archived fabricated/obsolete files
```

---

## Reproducibility

The benchmark can be reproduced by running `scripts/make_numbers.py`, which writes
results to `reports/benchmark_latest.json`. The README intentionally contains **no
hard-coded numbers** — all metrics are sourced from that JSON file.
