# Implemented forecast pipeline — 20 October 2026

The installed local classifier exceeds 80% on the expanded corpus. It has not been deployed to Railway. This does not establish 80% on the original corpus or on arbitrary future papers.

## Measured results

| Evaluation | N | Accuracy | Macro-F1 | Scope |
| --- | ---: | ---: | ---: | --- |
| Expanded corpus, label-maturity eligible 2020 cohort | 5,933 | **81.76%** | **0.7789** | Paper cutoff 2022; training/selection/calibration outcomes mature by 2022 |
| Expanded corpus, pooled 2018–2020 | 17,764 | 81.20% | 0.7721 | Retrospective development comparison; earlier forecast dates precede some calibration outcomes |
| Experimental citation GNN, eligible 2020 cohort | 5,933 | 80.70% | See graph report | Same label-maturity boundary; graph model not selected for the served tier |
| Experimental citation GNN, pooled expanded cohort | 17,764 | 80.54% | See graph report | Retrospective comparison |
| Repaired tabular model, original corpus | 4,500 | 71.38% | 0.7108 | Original development benchmark |
| Citation GNN ensemble, original corpus | 4,500 | 72.31% | 0.7155 | Original development benchmark; different sampling from expanded corpus |

Eligible-2020 accuracy bootstrap CI: **80.79–82.69%**. Its majority-class baseline is **53.36%**. High-impact precision is **78.79%**, recall **64.57%**. Thus a High prediction is still uncertain and the model misses some genuinely high-impact works. Overall accuracy is not uniformly above 80% within every discipline. The pooled Computer Science result is 78.32%.

These are retrospective OpenAlex snapshot benchmarks, not prospective external validation. Citation timestamps and label maturity are explicitly bounded; historical topic assignments, database backfilling, and other metadata still require a true point-in-time audit. Labels are percentiles within sampled field/year cohorts, not independently measured population-wide discipline percentiles. Average ranks handle ties; tied zero-citation works can make class sizes differ from 50/40/10.

## Data and selection

- Acquired **52,990 new unique works** through seeded random field/year sampling, without citation-ranked selection or an abstract requirement. The old corpus's IDs were excluded; measured ID overlap is zero.
- Five fields, publication years 2012–2020. Scope is scholarly works returned by the field/year query, rather than a document-type-restricted journal-article benchmark. Sampling differs from the original abstract-filtered corpus.
- Supervised training: **29,312** works published 2012–2016. Selection and calibration: disjoint **2,957 + 2,957** works published 2017. Pooled test: **17,764** works published 2018–2020.
- Annual citation histories are accepted only when their sum reconciles to the API's current citation total. Reconciled uncited works receive explicit zero-year records. Exclusions are recorded in `data/expanded_forecast/acquisition_audit.json`.
- Selection compares logistic regression, histogram gradient boosting, LightGBM, weighted/unweighted losses and a validation-selected average. The selected classifier is `hist_leaves=31_weight=None`. Temperature is fitted on the separate calibration partition.
- No threshold or model was chosen by maximizing test accuracy. Bug fixes and additional audits occurred during development, so the test reports do not claim independent confirmation.
- Validation learning curve for the selected model: 78.56% at 1,000 training examples; 79.98% at 15,000; 80.15% at 25,000; 80.49% at 29,312. These are subsample experiments on the same expanded corpus, not a controlled attribution of the entire improvement to sample size.

## Actual GNN role

`ml/gnn/citation_forecast.py` implements one-hop relational GraphSAGE with separate learned incoming/outgoing citation message functions. It uses actual neighbor papers, not citation self-loops masquerading as neighbors. Every neighbor's citation features are computed at the target paper's cutoff. There are no institutional prestige or author-causality claims.

Random supervised samples have little graph overlap: initial expanded-test coverage was **1.89%**. Fetching up to 40,000 real referenced-paper candidates increased expanded-test coverage to **32.82%** after excluding unreconciled histories and invalid publication dates. These unlabelled neighbors are not extra supervised training examples.

Three training seeds are averaged; early stopping and fusion weights use selection data only. Validation selected **graph weight 0** on the expanded corpus. The strongest served prediction therefore remains tabular. The frontend shows the experimental GNN's actual, explicitly uncalibrated probabilities and its own message-removal sensitivity, along with real sampled neighbors. It identifies that those probabilities do not replace the selected forecast.

On the original corpus, the GNN's pooled accuracy gain over the repaired tabular model was 0.93 percentage points (paired bootstrap interval +0.18 to +1.69). A separately trained root-only neural control reached 71.56%. Those original-corpus findings do not establish an improvement on the expanded sample.

## Serving and claim corrections

- Training and inference share `ForecastFeatures` and probability code. Features include early annual counts, growth/acceleration, dated field/year early-citation normalization, references and explicit missingness. Partial author metadata is disabled when training coverage is inadequate.
- Matching label outcomes are isolated in an evaluation-only reference. Forecast probability code never reads them. Known-paper backtests use the active model's own corpus; external papers do not receive a fabricated verified cohort outcome.
- Historical predictions whose cutoff predates training/calibration label maturity are flagged retrospective. Incomplete observation windows, missing peer references, sparse graph context and lower model confidence are exposed as limitations.
- Unknown graph neighborhoods remain empty. Data and bundle hashes identify inputs and reject incompatible tabular/graph pairs.
- Gemini generates advisory prose using supplied forecast facts. It cannot overwrite tiers, probabilities, model identity or citation forecasts. Embedded credential fallbacks and cross-project credential lookup were removed; use explicit project configuration. Previously distributed credential values should be rotated.
- Removed the heuristic numeric citation forecast derived from arbitrary class centers. Citation count forecasts remain null without a fitted regression estimate.
- Replaced the synthetic 42-cell “empirical” heatmap and hardcoded 28/60/12 percentages with observed annual counts. Removed guessed percentile readings, fabricated missing P90 defaults, venue probability boosts and unsupported individual-node influence claims.
- Feature and relation explanations are same-model probability sensitivities, not causal claims. Authorship position is not equated with PI status. Training membership is checked by actual ID, not inferred solely from publication year.
- Historical benchmark tables are labeled as historical; the active model card reports the new corpus and evaluation scope.

## Verification and reproducibility

- **15 tests passed**: 11 new forecast-contract tests and 4 existing feature-parity tests. They cover future-count injection, missing annual history, serialization parity, real graph neighbors, target-cutoff filtering, empty neighborhoods, isolation of outcome labels, partial author metadata, preserving numerical predictions against LLM output, training/serving graph-forward parity, and training/calibration label maturity.
- An 18-paper API/schema smoke test matched offline probabilities to four-decimal response rounding, verified matching corpus outcomes, and confirmed that no numeric citation forecast was fabricated.
- Modified frontend JavaScript passed syntax checks. Notebook code cells parse successfully. The training ZIP uses an explicit source/data whitelist, excluding `.env`, raw API dumps and embedded credential pools.
- The broader legacy temporal-test module could not collect in the new environment because it requires `torch_geometric`. The new citation model uses PyTorch directly. The entire legacy suite is not reported as passing.

Reproduce with `notebooks/scigraph_verified_forecast_colab.ipynb` and `reports/forecast_training_kit.zip`. The notebook includes tabular/GNN training, optional acquisition and graph enrichment, a separately trained no-edge control, calibration, field/class breakdowns, label-maturity checks and artifact export. It does not promise a preset score. The older V2 notebook's advertised score ranges are not evidence and it should not be used for this workflow.

Local installation uses `scripts/activate_forecast.py`; existing local artifacts are backed up. The installed artifacts are `ml/gnn/checkpoints/forecast_v3.joblib`, `citation_forecast.joblib`, and `FORECAST_CARD.json`. Graph serving requires PyTorch; scikit-learn is pinned to 1.7.2. The selected forecast checkpoint and model card are now versioned for deployment. Expanded raw data and the optional graph sidecar remain ignored by Git. No direct Railway deployment command has been run.

Machine-readable evidence: `reports/expanded_forecast/evaluation.json`, `reports/expanded_citation_enriched/graph_evaluation.json`, `reports/citation_forecast/graph_evaluation.json`, and the acquisition/neighbor audit files under `data/expanded_forecast/`.
