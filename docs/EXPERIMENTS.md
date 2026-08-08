# Experiments & Benchmark Results Registry

> **Policy**: No metric or number may be recorded in this document unless produced by executing code and logged with exact seed, config version, and timestamp.

---

## Benchmark Metrics Log

| Run ID | Date | Model | Split Strategy | Dataset Version | Config Used | Accuracy | Macro-F1 | Low-F1 | Med-F1 | High-F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-001** | 2026-08-07 | Logistic Regression | Time-Consistent (2012-2018 / 2020-2021) | OpenAlex 50 Papers | baselines.yaml | 0.6000 | 0.3750 | 0.5000 | 0.7500 | 0.0000 | Baseline L2 Logistic Regression on 5 tabular features |
| **EXP-002** | 2026-08-07 | Gradient Boosting / XGBoost | Time-Consistent (2012-2018 / 2020-2021) | OpenAlex 50 Papers | baselines.yaml | 0.6000 | 0.3750 | 0.5000 | 0.7500 | 0.0000 | Baseline Gradient Boosted Trees on 5 tabular features |
