# Accuracy and GNN review — 20 October 2026

Scope: source inspection and saved experiment artifacts; no new training, live Railway verification, or independent reproduction of scores. Local Python lacks pandas, so dataset schema and numerical checks were not executed. Existing working-tree changes were preserved.

## Evidence

- `ml/gnn/checkpoints/primary_model_meta.json`: served 51-feature logistic regression, test micro-F1 0.4804 (equivalent to accuracy for this single-label multiclass task), macro-F1 0.4641. These are saved metrics, not a fresh evaluation.
- `reports/accuracy_improvement/model_ladder.json`: logistic regression reports 71.24% test accuracy; the validation-selected ordinal model reports 70.76%. LightGBM reports 68.38%. A more complex model has not yet demonstrated a win here.
- `ml/features/builder.py`: served features omit the experimental cohort normalization and annual trajectory features. Sharing this builder between baseline training and serving does not provide parity with the separate experimental feature pipeline.
- `ml/inference_service.py`: tier probabilities come from the primary logistic model. Optional GNN output supplies a citation regression estimate and separate attributions, not the served tier probabilities.
- `ml/gnn/ego_graph.py`: only the target is appended to `paper_feats_list`; citation edge endpoints are both zero. Actual cited/citing paper representations are absent. Global authorship counts also need target-cutoff filtering.
- `ml/gnn/models/hetero_gnn.py`: affiliation relations emitted by the ego builder are absent from both convolution layers. Institutional message-passing claims are unsupported by this implementation.
- GNN occlusion receives the primary classifier probability as its original probability. Comparing a GNN occlusion output against a different model's probability does not isolate an occlusion effect.
- Linear explanations label absolute coefficient-times-standardized-feature scores as probability changes. Logit contributions are not percentage-point probability changes.
- `web/js/graph.js` describes institutional infrastructure flow, individual author effects, and venue prestige boosts without measurements supporting those claims. `web/js/predictor.js` describes the tier decision as GNN synthesis.
- When GNN regression is unavailable, inference estimates citation counts from manually chosen class centers (0.5 times p50, midpoint, 1.25 times p90). This is a heuristic, not a validated citation regression forecast.
- `scripts/train_scigraph_v2.py` defaults absent trajectory to zeros and cohort median to 12; it reads split parquet files rather than constructing the experimental ladder from the source tables. Actual column availability needs verification. It fits no advertised multiclass calibration. Its exported model lacks the scaler expected by current serving and uses a different feature schema.
- `notebooks/scigraph_v2_colab_training.ipynb` advertises unverified performance ranges. Its name and GPU check do not establish that training uses a GPU or a larger corpus.
- `ml/features/asof.py` computes normalization within each evaluation cohort. This uses early counts rather than future labels, but serving one paper requires an equivalent, timestamped cohort reference; batch experiment performance does not establish single-paper parity.

## Recommended implementation order

1. Establish the prediction contract: citation impact, not scientific truth or quality; define whether the label counts years publication+3 through publication+5. Current code uses publication-year bins through publication+2, which is not exactly a rolling 24-month window. Define handling of tied percentile boundaries and minimum cohort size.
2. Audit completeness of annual citation records and missing fields. Unknown history must not silently mean zero. Freeze metadata and neighbor features at the target paper's cutoff, including author activity, OA status and historical venue metrics where used. Until audited, do not claim zero leakage.
3. Create one versioned feature/bundle contract for offline training and API inference: annual counts keyed by year, cumulative early citations, growth, acceleration, field normalization, missingness flags, and a persisted reference/text encoder. Reject incompatible bundles. Test identical raw-paper features and probabilities through both paths.
4. Reproduce the baseline and trajectory ladder with temporal validation, then compare logistic/ordinal models and boosted trees under equal feature availability. Try balanced and unweighted losses; balanced weights do not guarantee better raw accuracy. Fit calibration and decision thresholds only on designated validation data.
5. Reserve a fresh, sufficiently mature temporal holdout before further model selection: the existing test results have already informed development. Report accuracy, macro-F1, per-class precision/recall, confusion matrix, calibration, field/year breakdowns, majority baseline and confidence intervals. Also evaluate the API path, including out-of-corpus papers.
6. Run training-size experiments (for example 2k, 4k, 7.5k, then 15k/25k/50k if acquired) against fixed temporal validation. Expand representative fields/years and real graph neighborhoods. Repeating records or training longer does not increase information. Choose acquisition scale from the measured learning curve.
7. Repair the graph: include actual neighboring papers and target-cutoff-safe features, correct directional edges and relation schema, and train on the same neighborhood construction used at inference. Include affiliation convolution only if retrained and evaluated. A new architecture needs a new checkpoint.
8. Compare tabular-only, GNN-only and tabular-plus-GNN embeddings, plus shuffled/removed-edge controls and per-relation ablations across multiple seeds. Select any fusion weights on validation. Claim a predictive graph contribution only when the improvement is reproducible and uncertainty supports it.
9. Replace unsupported frontend text. Show the actual classifier, graph availability, graph coverage, evidence timestamps, calibrated probabilities, and missing-data limitations. Keep descriptive metadata separate from measured model effects. Compute graph ablation deltas against the same served fused model; do not call these causal effects.

## Useful user-facing insights

- Early-citation percentile against a named, dated discipline cohort, with cohort size.
- Observed annual citation trajectory and whether growth accelerates or slows.
- Verified neighboring papers with links and the relation that connects them.
- Prediction uncertainty and an insufficient-evidence outcome for sparse/missing history or unsupported fields.
- If experimentally validated: probability with versus without graph context, relation removal effects, and predictive neighbors. Label these model sensitivities, not reasons a paper will succeed.
- Clearly separate historical backtest outcomes from information available at prediction time. Keep LLM prose grounded in structured evidence; do not treat agreement between Gemini and the classifier as independent validation.

## 80% target and Colab decision

80% three-class accuracy is an experimental target, not a promised outcome. Binary accuracy, top-decile precision, AUROC and selective accuracy are different metrics and must not be substituted. If abstention reaches 80% on a subset, report its coverage and full-population performance.

Do not spend a long Colab run on the current V2 notebook yet. First repair data/feature contracts, verify missingness and leakage, and establish API parity. Then prepare one reproducible Colab notebook that audits inputs, uses temporal selection/calibration, compares the baseline and graph ablations, exports an API-compatible bundle, and reports measured results without preset success claims.

Method references: https://scikit-learn.org/stable/modules/learning_curve.html and https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html
