# SciGraph AI 2.0: Scientometric Open Access Citation Advantage Analysis

**Date:** 2026-10-16 16:06:49 UTC  
**Sample Size:** 16,500 multi-disciplinary research articles  
**Outcome Metric:** $\ln(1 + \text{citations_window})$ over horizon $(\text{cutoff}, \text{pub} + 5]$  

---

## 1. Executive Summary & Core Empirical Findings

- **Crude Open Access Effect:** **+23.6%** (b = 0.2118, 95% CI: [0.1850, 0.2386], p = 1.20e-53, R2 = 0.0143).
- **Multivariate Adjusted Effect:** **+10.8%** (b = 0.1027, 95% CI: [0.0775, 0.1280], p = 1.63e-15, R2 = 0.3091).
- **Confounding Attenuation:** Controlling for field, cohort year, author count, SJR, and top-tier venue status reduces the estimated association by **51.5%**.

> [!IMPORTANT]
> **Causation vs. Association:** In accordance with established scientometric literature (e.g., Davis et al. BMJ 2008, Gaule & Maystre 2011), the remaining positive association cannot be interpreted as purely causal. High-impact research groups systematically self-select into Gold/Hybrid open access publishing.

---

## 2. Regression Specifications & Model Comparison

| Specification | Model Formulation | OA Coef ($b$) | 95% Confidence Interval | Citation Multiplier ($e^b - 1$) | $R^2$ | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Crude OLS** | $\ln(1+C) \sim \text{is_oa}$ | `0.2118` | `[0.1850, 0.2386]` | **+23.6%** | `0.0143` | `< 0.001` |
| **Adjusted OLS** | $\ln(1+C) \sim \text{is_oa} + \mathbf{X}_{\text{covariates}}$ | `0.1027` | `[0.0775, 0.1280]` | **+10.8%** | `0.3091` | `< 0.001` |

---

## 3. Disaggregated OA Subtypes (Reference = Closed)

| OA Subtype | Crude Coef ($b$) | Crude Multiplier | Adjusted Coef ($b$) | Adjusted Multiplier | Adjusted 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Gold` | `+0.3152` | +37.1% | `+0.1286` | **+13.7%** | `[0.0935, 0.1638]` |
| `Green` | `+0.1373` | +14.7% | `+0.0863` | **+9.0%** | `[0.0578, 0.1147]` |
| `Hybrid` | `+0.2851` | +33.0% | `+0.1011` | **+10.6%** | `[0.0657, 0.1365]` |
| `Bronze` | `+0.1524` | +16.5% | `+0.1000` | **+10.5%** | `[0.0647, 0.1352]` |

---

## 4. Stratified Analysis Across Scientific Fields

| Scientific Field | Sample Size | Adjusted Coef ($b$) | 95% Confidence Interval | Adjusted Advantage ($e^b - 1$) | Field $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Biochemistry, Genetics and Molecular Biology** | 3,300 | `+0.2298` | `[0.1551, 0.3045]` | **+25.8%** | `0.1955` |
| **Computer Science** | 3,300 | `+0.0615` | `[0.0023, 0.1207]` | **+6.3%** | `0.3298` |
| **Engineering** | 3,300 | `+0.0442` | `[0.0065, 0.0820]` | **+4.5%** | `0.2133` |
| **Medicine** | 3,300 | `-0.0190` | `[-0.0810, 0.0430]` | **-1.9%** | `0.2032` |
| **Physics and Astronomy** | 3,300 | `+0.1190` | `[0.0587, 0.1792]` | **+12.6%** | `0.1052` |

---

## 5. Methodological Notes & Reproducibility

- **Command to Reproduce:** `python scripts/oa_analysis.py`
- **Input Data:** `data/works.parquet` (16,500 OpenAlex papers)
- **Dynamic Consumption:** Read directly at runtime by `ml/scientometrics/access_economics.py` and served via `/api/v1/paper/predict`.
- **Zero Handcrafted Numbers:** All percentages and confidence intervals generated from empirical regression matrices.