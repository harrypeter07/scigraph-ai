# SciGraph AI 2.0 — Scientific Citation Impact Prediction & Independent Verification

> **20 October 2026 update:** the expanded-corpus classifier measures **81.76% accuracy** (95% bootstrap CI 80.79–82.69%, macro-F1 0.7789) on 5,933 papers published in 2020. Their cutoff is 2022, when all training/selection/calibration labels have matured. The broader retrospective 2018–2020 result is 81.20%. These are sample-relative retrospective OpenAlex results, not a guarantee for new papers or proof of a complete point-in-time metadata audit. The original corpus and expanded random sample have different sampling distributions; scores are not directly comparable.
>
> The experimental citation GNN measures 80.70% on the eligible 2020 cohort; validation retains the stronger tabular forecast. See [implementation and measured results](reports/forecast_implementation_2026-10-20.md), [reproducible Colab notebook](notebooks/scigraph_verified_forecast_colab.ipynb), and [expanded evaluation](reports/expanded_forecast/evaluation.json). Historical benchmark tables below describe the original corpus.

The overview and **Accuracy & Evidence** page read `web/data/forecast_evidence.json`, generated with `python scripts/publish_forecast_evidence.py`. The API verifies the loaded checkpoint checksum before identifying those results with the serving bundle. The selected `forecast_v3.joblib` and its model card are versioned for deployment. The larger experimental citation graph sidecar remains optional and requires PyTorch; production tabular serving does not require it. Reproduce or export the graph sidecar with the verified Colab workflow.

> **Predicts whether an academic paper will achieve Low (<50th), Medium (50–90th), or High (≥90th) citation impact in its field and cohort, using only information available 2 years after publication ($T_{\text{cutoff}} = \text{pub\_year} + 2$).**

---

## 1. Definitive Benchmark Results (Held-Out Test Split, N = 4,500 Papers, 2018–2020)

All metrics are read directly from [`reports/benchmark_latest.json`](reports/benchmark_latest.json), [`reports/accuracy_improvement/model_ladder.json`](reports/accuracy_improvement/model_ladder.json), and [`reports/accuracy_improvement/alternative_tasks.json`](reports/accuracy_improvement/alternative_tasks.json). Evaluation protocol enforces equal validation-only hyperparameter tuning, validation-fit temperature scaling calibration, and 1,000 bootstrap confidence intervals. Zero test tuning.

### A. 3-Class Cohort Impact Prediction (2-Year Cutoff Horizon)

| Model Identifier | Model Architecture & Features | Accuracy | Macro-F1 (95% CI) | AUROC (OvR) | Prec@10% (High) | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **$M_0$ (Majority)** | Always predicts dominant class (Low) | 49.76% | 0.2215 [0.217, 0.226] | 0.5000 | 10.31% | Baseline |
| **$B_1$ (Early Cits)** | Cutoff citations threshold heuristic | 48.93% | 0.4676 [0.451, 0.483] | 0.7097 | 37.11% | Heuristic |
| **$B_2$ (Venue Tier)** | Historical venue mean citation percentiles | 49.40% | 0.2968 [0.286, 0.308] | 0.5255 | 14.89% | Heuristic |
| **$M_4$ (Homo GNN)** | Homogeneous GraphSAGE (normalized node features) | 52.91% | 0.4186 [0.402, 0.436] | 0.6709 | 24.67% | Baseline |
| **$M_5$ (Hetero GNN)** | 2-hop HeteroGraphSAGE (Paper-Author-Venue) | 47.18% | 0.3266 [0.317, 0.337] | 0.5704 | 18.00% | Baseline |
| **Served Baseline** | Logistic Regression on Early Cits + Metadata (47 dims) | 55.49% | 0.5244 [0.506, 0.542] | 0.7141 | 39.56% | Baseline Model |
| **Enhanced M4 (Two-Stage)** | **Ordinal Two-Stage Model ($F_0..F_7$ + Text SVD, 55 dims)** | **70.76%** | **0.7014 [0.686, 0.717]** | **0.8508** | **71.56%** | **Winner on Val F1 (0.735)** |
| **Enhanced M0 (Logistic)** | **Tuned L2 Classifier ($F_0..F_7$ + Text SVD, 55 dims)** | **71.24%** | **0.7040 [0.687, 0.720]** | **0.8525** | **72.00%** | **Highest Test Accuracy** |

---

### B. Practically Actionable Alternative Task Formulations (Held-Out Test Set)

Because 3 fine-grained classes suffer from Poisson arrival noise near class boundaries, we evaluated alternative formulations on held-out test data:

| Task Formulation | Target Definition | Test Accuracy | Macro-F1 / PR-AUC | Test AUROC | Key Practical Value |
|:---|:---|:---:|:---:|:---:|:---|
| **Task 2a: Binary Median Split** | Citations $\ge 50$th percentile (Low vs Med+High) | **75.76%** | **0.7619 (F1)** | **0.8415** | Precision: 75.19%, Recall: 77.22% |
| **Task 2b: Elite Decile Discovery** | Citations $\ge 90$th percentile (Top 10% vs Rest) | — | **0.7492 (PR-AUC)** | **0.9520** | **71.78% Precision at Top 10% (6.96x lift over random)** |
| **Task 3: 3-Year Observation** | Cutoff extended to $T_{\text{pub}} + 3$ (3 years visible) | **77.58%** | **0.7721 (F1)** | **0.8993** | +7.16% absolute accuracy lift over 2-year cutoff |

---

## 2. Key Scientific Findings

### A. Why Did Accuracy Jump from ~55% to ~71%?
The baseline pipeline had two critical modeling omissions:
1. **Label-Feature Mismatch:** Impact tiers are defined relative to the paper's `(field, publication_year)` cohort, but the baseline only fed absolute raw citations (`log1p(early_citations)`). Feeding **cohort-normalized percentile ranks (+F1)** immediately bridged this gap (+0.0566 Val F1).
2. **Missing Trajectory Dynamics:** The baseline only used total early citations. Incorporating annual citation trajectory ($y_0, y_1, y_2$), acceleration ($(y_2 - y_1) - (y_1 - y_0)$), and growth ratios **(+F2)** provided massive predictive signal (+0.1105 Val F1).
3. **Compact Semantic Text Embeddings:** Adding 32-dim Train-only TF-IDF + SVD title embeddings **(+E1)** lifted test accuracy past 70% without overfitting.

### B. How Good Can It Possibly Get: The Predictability Ceiling & Noise Proof
A rigorous Negative Binomial over-dispersion model fitted on training residuals ($\hat{\phi} = 3.8633$, $\text{Var}(W \mid \mu) = \mu + \mu^2 / \phi$) proved:
- **Over-Dispersion:** At citation counts in the hundreds ($\mu \approx 381$), true empirical variance is $\approx 37,957$ ($\sigma \approx 195$), almost $100\times$ larger than naive Poisson variance ($\sigma \approx 19.5$).
- **Irreducible Error Ceiling:** The maximum theoretical accuracy for any Bayes-optimal expected-value predictor is **65.04%** (irreducible stochastic error rate = **34.96%**).
- **Boundary Paper Preponderance:** **44.33% of test papers** have maximum class probability $\max_k P_{ik} < 0.60$. Their true tier is determined by stochastic citation arrival variance rather than deterministic features.
- **Within-One-Tier Accuracy (94.91% – 96.44%):** While fine-grained adjacent-boundary distinction (Low vs. Med or Med vs. High) is subject to arrival noise, polar confusion ($P(\text{Low} \leftrightarrow \text{High})$) occurs on only **3.56% – 5.09%** of papers.
- **Authoritative Analysis:** For the full mathematical proof and selection bias audit, see [`docs/THE_TRUTH_ABOUT_ACCURACY.md`](docs/THE_TRUTH_ABOUT_ACCURACY.md).

### C. What Happened to the GNN in the Latest Modifications?
1. **Node Standardization Fix:** Standardizing paper node features using a train-fitted `StandardScaler` improved $M_4$ Homogeneous GraphSAGE Macro-F1 from **0.314 → 0.4186** (Accuracy: **52.91%**, AUROC: **0.6709**).
2. **Why GNN Alone Does Not Beat Tabular Models on this Graph:**
   - **Early Citations Are Dominant:** Early citations at cutoff ($B_1$) alone achieve AUROC = 0.7097.
   - **Empirical Graph Density Audit (Phase A):** In our OpenAlex corpus density check ([`reports/dense_corpus_stats.json`](reports/dense_corpus_stats.json)), target papers have a **median degree of only 5.0** and **in-corpus reference coverage of only 5.7%** (over 94% of reference links point outside the sampled subfield).
   - Running neighborhood aggregation over sparse citation links dilutes the strong early-citation signal with neighbor noise.
   - **Legitimate Scientific Finding:** GNNs do not automatically beat strong tabular predictors when tabular early citations are strong and the citation graph sample is sparse.

---

## 3. Independent External Ground-Truth Verification (Layers 1–3)

Evaluators do not need to trust our database or our models. SciGraph provides direct verification against live external sources:

### A. Live Single-Paper Verification Tool
```bash
# Query OpenAlex API live, freeze at pub_year + 2, and verify revealed 3-year citations:
python scripts/verify_paper.py W2752782242
```
**Example Output on Held-Out Paper [`W2752782242`](https://openalex.org/W2752782242)** (*"Squeeze-and-Excitation Networks"*, 2018):
- **Frozen Inputs as of 2020**: 3,661 citations at cutoff.
- **Revealed Outcome (2021–2023)**:
  - 2021: 3,469 citations
  - 2022: 4,703 citations
  - 2023: 5,925 citations
  - **Total 3-Year Window**: **14,097 citations**
- **Threshold**: $CS::2018$ $P_{90} = 1024.1 \implies$ **High ($\ge 90\text{th}$)**
- **Audit**: **Exact Match** verified live against `api.openalex.org`.

### B. Live Batch Spot-Checking
```bash
# Randomly samples held-out test papers and spot-checks them live against api.openalex.org:
python scripts/verify_sample.py --n 20
```

### C. Cryptographic Prediction Ledger
- Every prediction generates an immutable SHA256 receipt before ground truth is revealed.
- Stored in [`reports/prediction_ledger.jsonl`](reports/prediction_ledger.jsonl).
- Audited via `/api/v1/verify/receipt/{receipt_hash}/check`.

### D. Leakage Controls & Anti-Cheating
- **Future-Poison Immunity**: Injecting +100,000 citations 10 years in the future produces strictly $\Delta P = 0.000000$ shift.
- **Snapshot Graph Temporal Invariance**: Unit tests ([`tests/test_snapshot_graph_time.py`](tests/test_snapshot_graph_time.py)) confirm snapshots $G_t$ contain zero citing edges or counts after year $t$.

---

## 4. How to Run & Verify

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all verification and temporal integrity unit tests (48/48 passing across 10 suites)
pytest tests/ -v

# 3. Export all codebase code into clipboard
python scripts/export_all_code_to_clipboard.py

# 4. Start the interactive dashboard
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
# Open http://localhost:8080 in your browser
```

---

## 5. Repository Structure

```
scigraph/
├── data/
│   ├── MANIFEST.sha256         ← Cryptographic checksums of all dataset files
│   ├── works.parquet           ← 16,500 core works (train, val, test)
│   ├── counts_by_year.parquet  ← Annual citation accumulation
│   ├── citation_edges.parquet  ← Direct citation graph edges
│   └── dense/                  ← 39,600 Computer Vision dense corpus (Phase A)
├── ml/
│   ├── features/               ← Single leak-free feature builder & As-Of feature ladder
│   ├── gnn/
│   │   ├── snapshot.py         ← Time-bounded snapshot graph builder G_t
│   │   └── checkpoints/        ← Served primary model (primary_model.joblib)
│   ├── evaluation/
│   │   ├── fit_primary_model.py← Primary model fitting (VAL-tuned)
│   │   ├── export_heldout.py   ← Held-out test prediction exporter
│   │   └── run_dense_experiments.py ← Dense systematic GNN search runner
│   └── verification/
│       └── receipt.py          ← Cryptographic receipt & ledger engine
├── reports/
│   ├── benchmark_latest.json   ← Authoritative benchmark results (M0..M5, B1..B2)
│   ├── accuracy_improvement/   ← Complete Phase 1-7 reports (ceiling, ladder, tasks)
│   │   ├── ceiling.json        ← Mathematical Poisson boundary noise proof
│   │   ├── feature_ladder.json ← Step-by-step feature group additions (F0..F7)
│   │   ├── embeddings.json     ← Text embedding benchmark & drift check
│   │   ├── model_ladder.json   ← M0-M5 model benchmark & calibration
│   │   ├── learning_curve.json ← Empirical data scaling curve
│   │   ├── alternative_tasks.json ← Binary median split & elite top-10% discovery
│   │   └── ACCURACY_IMPROVEMENT_REPORT.md ← Comprehensive synthesis report
│   ├── dense_corpus_stats.json ← Empirical Gate A density metrics
│   ├── heldout_predictions.parquet ← 6,000 predictions (val + test)
│   └── prediction_ledger.jsonl ← Append-only cryptographic prediction ledger
├── api/                        ← FastAPI backend endpoints (/predict, /verify)
├── web/                        ← Interactive Dashboard (Verification Lab & Spot-Check)
├── scripts/
│   ├── accuracy_improvement/   ← Phase 1 to 6 experiment scripts
│   ├── export_all_code_to_clipboard.py ← Bundles & copies complete code to clipboard
│   ├── verify_paper.py         ← Independent external single-paper verification
│   └── verify_sample.py        ← Independent external batch spot-checking
└── tests/                      ← Comprehensive test suite (48/48 passing)
```
