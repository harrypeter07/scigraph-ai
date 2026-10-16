# SciGraph AI 2.0: Authentic Scientometrics & Heterogeneous Graph Neural Networks

> **Scientometric 5-Year Citation Trajectory & Impact Tier Prediction**  
> **Backend:** FastAPI + PyTorch Geometric (`HeteroConv` + `SAGEConv`) + LightGBM  
> **Dataset:** 16,500 Multi-Disciplinary OpenAlex Works (2010–2020) + SCImago + CORE Rankings  
> **Evaluation:** Strict Temporal Cutoff ($T_{\text{cutoff}} = \text{pub\_year} + 2$) with Held-out Temporal Split ($\ge 2018$)  

---

## 🎯 Key Remediation Pillars & Non-Negotiable Rules

SciGraph AI 2.0 underwent a complete, rigorous scientific remediation to ensure **zero data fabrication**, **zero silent failures**, **strict temporal cutoff enforcement**, and **honest benchmark comparison**:

1. **Zero Fabricated Constants (Rule R1)**:
   - Sourced venue prestige directly from **SCImago Journal Ranks (Q1–Q4)** via ISSN-L and **CORE Conference Rankings ($A^*, A, B, C$)** via acronyms. Unknown venues strictly return `null` (never defaulting to Q2 or assuming Q1).
   - Real Article Processing Charges (APC) in USD from OpenAlex and DOAJ. Eliminated hardcoded paywall fees ($35).
   - Removed salted hash topic velocity, fake author h-index formulas, and synthetic percentages.
2. **Zero Silent Failures (Rule R2)**:
   - Replaced all `except Exception: pass` and `return {}` with discrete HTTP status codes (404 Not Found, 429 Upstream Rate Limit, 502 Upstream Error, 503 Model Unavailable, 422 Bad Input).
   - Loud failure at boot if the ML checkpoint or model card is missing or corrupted.
3. **Strict Temporal Cutoff (Rule R3)**:
   - Every input feature (citations accrued, venue SJR, author activity) is computed strictly as of the cutoff year ($T_{\text{cutoff}} = \text{pub\_year} + 2$).
4. **Authentic Relational Graph Architecture**:
   - PyTorch Geometric `AuthenticHeteroSAGE` network executing 2-hop message passing across 5 node types (`paper`, `author`, `institution`, `venue`, `topic`) and 10 relational edge types.
5. **Honest Scientific Evaluation (Rule R5)**:
   - Benchmarked on 4,500 held-out temporal test papers against Log-Linear, Regularized Logistic Regression, LightGBM, and Graphless Deep MLP.

---

## 📊 Held-Out Temporal Test Set Benchmark Results

Evaluated on 4,500 papers published 2018–2020:

| Model Paradigm | Macro-F1 | Weighted-F1 | AUROC (OvR) | MAE (Log Citations) | ECE Calibration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Log-Linear Early Citations** | 0.3767 | 0.3508 | 0.6489 | 0.4607 | 0.3584 |
| **Logistic Regression (L2)** | 0.5057 | 0.5533 | 0.7180 | 0.4365 | 0.1214 |
| **LightGBM Gradient Boosted Trees** | 0.4995 | 0.5002 | 0.7179 | **0.4185** | 0.1832 |
| **Graphless Deep MLP** | 0.5026 | 0.5585 | 0.7153 | 0.6633 | 0.0618 |
| **★ PyG HeteroGraphSAGE (2-Hop)** | **0.5381** | **0.5639** | **0.7205** | 0.5088 | **0.0083** |

*Detailed benchmark report: [`reports/BENCHMARK_REPORT.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/BENCHMARK_REPORT.md)*

---

## 📂 Key Project Artifacts & Documentation

- **Audit Findings**: [`AUDIT.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/AUDIT.md) — Comprehensive line-by-line catalog of remediated legacy issues.
- **Dataset Documentation**: [`data/DATA_CARD.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/data/DATA_CARD.md) — Multi-disciplinary corpus schema, field distributions, and validation checks.
- **Empirical OA Analysis**: [`reports/OA_ADVANTAGE_ANALYSIS.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/OA_ADVANTAGE_ANALYSIS.md) — +19.3% median OA citation advantage across 16,500 works.
- **Benchmark Evaluation Report**: [`reports/BENCHMARK_REPORT.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/reports/BENCHMARK_REPORT.md) — 5-model competitive evaluation on held-out temporal data.
- **Presentation Guide**: [`PRESENTATION_GUIDE.md`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/PRESENTATION_GUIDE.md) — Defense prep with exact answers to teacher questions.

---

## 🚀 Quickstart & Server Launch

### 1. Installation
Ensure Python 3.10+ is installed:
```powershell
pip install -r requirements-cpu.txt
```

### 2. Run Verified Test Suite
```powershell
pytest tests/ --ignore=tests/legacy -v
```
*(All 16 unit, integration, and scientometric tests pass cleanly).*

### 3. Start Application Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### 4. Access Interactive Dashboard
Open your web browser at:
```
http://localhost:8080/
```
Features available in the dashboard:
- **Search by Title, DOI, or OpenAlex ID** with live autocomplete.
- **Calibrated Probability Gauges** with temperature scaling uncertainty bounds.
- **Ground-Truth Backtest Comparison** for papers $\le 2020$.
- **Citation Trajectory Timeline** visualizing historical accumulation up to cutoff.
- **Model & Data Verification Page** displaying genuine corpus counts and the full benchmark table.
- **Side-by-Side Paper Comparison** tab.
