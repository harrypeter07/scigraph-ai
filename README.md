# SciGraph AI 2.0: Authentic Scientometrics & Scientific Benchmark Suite

> **Scientometric 5-Year Citation Trajectory & Impact Tier Prediction**  
> **Backend:** FastAPI + PyTorch Geometric (`HeteroConv` + `SAGEConv`) + LightGBM  
> **Dataset:** 16,500 Multi-Disciplinary OpenAlex Works (2010–2020) + SCImago + CORE Rankings  
> **Evaluation:** Strict Temporal Cutoff ($T_{\text{cutoff}} = \text{pub\_year} + 2$) with Clean Temporal Splits (Train: 2012–2016, Val: 2017, Test: 2018–2020)  
> **Numbers Single Source of Truth:** [`docs/NUMBERS.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/docs/NUMBERS.md) (All values read dynamically from `reports/*.json`)

---

## 🎯 Rigorous Scientific Remediation & Absolute Rules

SciGraph AI 2.0 was developed under strict adherence to scientific reproducibility standards:

1. **Zero Data Fabrication & Single Feature Builder (Rule F1 & WP1)**:
   - Sourced venue prestige directly from **SCImago Journal Ranks (Q1–Q4)** via ISSN-L and **CORE Conference Rankings ($A^*, A, B, C$)** via acronyms. Unknown venues strictly return `-1.0` with explicit `venue_known=0.0` missingness indicators.
   - Sourced real Article Processing Charges (APC) in USD from OpenAlex and DOAJ. Eliminated hardcoded paywall fees.
   - All paper features are generated through a single, mathematically verified module: `ml/features/builder.py` (47 dimensions, 14 feature groups). Zero fragmented feature duplication.
   - Persistent text encoder: TF-IDF(2500) + TruncatedSVD(32) fit strictly on training titles only (`ml/gnn/checkpoints/text_encoder.joblib`).
2. **Zero Temporal Leakage (Rule F2 & WP2)**:
   - Future-poison invariant: Modifying citation counts after cutoff has zero effect (`max_diff = 0.00000000`).
   - Citation graph filtered to strictly retain in-corpus edges where $\text{pub\_year}_{\text{citing}} \le T_{\text{cutoff},\text{referenced}}$. Leaked future edges (43,816 edges, 49.96%) completely eliminated.
   - Clean training split: OpenAlex citation counts start in 2012. Papers published in 2010–2011 were excluded from training to prevent early citation history truncation.
3. **Train-Serving Parity & Authentic Ego-Graphs (WP3)**:
   - Evaluated parity gap on 500 test works: Measured gap is **+0.0000 Macro-F1**.
   - Serving generates genuine multi-relational ego-graphs matching training schemas and degrees via `ml/gnn/ego_graph.py`.
4. **Empirical Cohort Backtesting (WP4)**:
   - Ground truth backtesting evaluates strictly the window $(\text{cutoff\_year}, \text{pub\_year} + 5]$. Open windows (e.g. pub $\ge 2021$) return null ground truth with clear warnings instead of fabricated labels.
   - Classification tiers determined against empirical $p50/p90$ thresholds from `data/reference/cohort_thresholds.json`.
5. **Real Attributions & Dynamic Explainability (WP5)**:
   - Local feature occlusion computed on the deployed neural network against empirical train-set baselines. Zero hardcoded attribution arrays.

---

## 📊 Standardized 5-Seed Benchmark Results (WP6)

Evaluated across 5 random seeds (0, 1, 2, 3, 4) on the held-out temporal test split (4,500 papers, 2018–2020) with 1,000 bootstrap resamples for 95% confidence intervals and paired hypothesis testing:

| Model Paradigm | Architecture / Protocol | Test Macro-F1 (Mean ± Std) | 95% Bootstrap CI | AUROC (OvR) | Calibrated ECE | p-val vs M2 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M0: Majority Class** | Majority Class Baseline | 0.2215 ± 0.0000 | [0.2172, 0.2258] | 0.5000 | 0.0006 | 1.0000 |
| **M1: Regularized LogReg** | L2 LogReg (C tuned on Val) | **0.5244 ± 0.0000** | [0.5062, 0.5421] | 0.7141 | 0.0318 | 0.0030 |
| **M2: LightGBM** | Gradient Boosted Trees (Val tuned) | 0.5020 ± 0.0000 | [0.4866, 0.5185] | **0.7182** | 0.0799 | — (ref) |
| **M3: Graphless MLP** | 2-Layer MLP (47 -> 64 -> 32 -> 3) | 0.4111 ± 0.0352 | [0.4010, 0.4349] | 0.6711 | 0.0712 | 1.0000 |
| **M4: Homogeneous SAGE** | PyG 2-Hop SAGE on Citation Graph | 0.3138 ± 0.0042 | [0.3061, 0.3257] | 0.5129 | 0.0917 | 1.0000 |
| **M5: HeteroGraphSAGE** | PyG 2-Hop HeteroConv (5 edge relations) | 0.3167 ± 0.0164 | [0.3149, 0.3343] | 0.5173 | 0.0679 | 1.0000 |

*Detailed benchmark report: [`reports/BENCHMARK_REPORT.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/BENCHMARK_REPORT.md)*  
*Traceable JSON data: [`reports/benchmark_latest.json`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/benchmark_latest.json)*

---

## 📈 Empirical Scientometric Open Access Advantage (WP7)

Evaluated via Ordinary Least Squares (OLS) with asymptotic standard errors and t-distribution 95% CIs on 16,500 multi-disciplinary works:

- **Crude Open Access Effect:** **+23.59%** citation multiplier ($b = 0.2118$, 95% CI: `[0.1850, 0.2386]`, $p < 10^{-50}$, $R^2 = 0.0143$).
- **Multivariate Adjusted Effect:** **+10.82%** citation multiplier ($b = 0.1027$, 95% CI: `[0.0775, 0.1280]`, $p < 10^{-14}$, $R^2 = 0.3091$).
- **Confounding Attenuation:** Controlling for scientific field, publication cohort, author team scale, and venue prestige attenuates the estimated advantage by **51.5%**.
- **Field-Specific Adjusted Multipliers:**
  - Biochemistry, Genetics & Molecular Biology: **+25.8%** ($p < 10^{-8}$)
  - Physics & Astronomy: **+12.6%** ($p < 0.001$)
  - Computer Science: **+6.3%** ($p = 0.042$)
  - Engineering: **+4.5%** ($p = 0.022$)
  - Medicine: **-1.9%** (not statistically significant, $p = 0.548$)

*Detailed OA report: [`reports/OA_ADVANTAGE_ANALYSIS.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/OA_ADVANTAGE_ANALYSIS.md)*

---

## ⚠️ Known System Limitations (Scientometric & Architectural)

1. **2010–2011 Truncation Bias:** OpenAlex `counts_by_year` records start in 2012. Works published in 2010–2011 (18.18% of the raw sample) have truncated early citation trajectories. To eliminate this bias, training splits strictly use cohorts 2012–2016 (7,500 works; 100% complete trajectories).
2. **Residual As-Of-Today Covariates:** Open Access status and APC fees reflect contemporary DOAJ/OpenAlex crawls rather than historical states as of publication cutoff. Ablation experiments prove their predictive impact is modest (delta = +0.0002 Macro-F1).
3. **GNN vs. Tabular Empirical Reality:** In tabular citation velocity data, LightGBM and Regularized Logistic Regression outperform multi-hop GNNs due to high predictive concentration in early citation velocity ($T_{\text{cutoff}} = \text{pub} + 2$) and graph neighborhood sparsity (50% of papers have sparse in-corpus degrees).
4. **CD Disruption Index Subgraph Truncation:** On a 16.5k-node corpus, reference citation histories ($k$) cannot be fully observed for out-of-corpus works. The engine reports `null` with status `insufficient_reference_subgraph` rather than fabricating an artificial disruption score.
5. **Train-Serving Parity:** Single-paper serving achieves +0.0000 parity gap against benchmark graph evaluation via authentic ego-graph reconstruction (`ml/gnn/ego_graph.py`).

---

## 🚀 Quickstart & Server Launch

### 1. Installation
Ensure Python 3.10+ is installed:
```powershell
pip install -r requirements-cpu.txt
```

### 2. Run Verified Test Suite
```powershell
pytest tests/ -v
```

### 3. Start Application Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### 4. Quantitative Numbers Single Source of Truth
```powershell
python scripts/make_numbers.py
```
Outputs `docs/NUMBERS.md` and `docs/NUMBERS.json`.
