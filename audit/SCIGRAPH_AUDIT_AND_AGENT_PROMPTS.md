# SciGraph AI 2.0 — Audit, Score, Verification Design & Agent Prompts

Basis: static read of the full codebase manifest + the dashboard screenshot. Nothing was executed.
Where I say "verify", the agent must confirm by running code before changing anything.

---

## 1. SCORE: 41 / 100

| Area | Weight | Score | Why |
|---|---|---|---|
| Tabular ML pipeline and evaluation | 15 | 10.5 | Real temporal split, val-tuned, bootstrap CIs, honest baselines. Best part of the project. |
| Claims are true and consistent | 15 | 6 | Several hard-coded UI numbers are wrong. Tabs disagree with each other. Old v1 docs contradict v2. |
| Verification / provability | 20 | 8 | "Blind test" is cosmetic. Poison test is rigged to pass. No per-paper receipt. Truth source unclear on screen. |
| GNN (the core promise) | 15 | 3 | GNN is not used for the served class prediction. It is also badly trained and leaky. |
| UI for a non-technical person | 15 | 4.5 | Numbers without meaning, jargon, fake graph nodes, blank placeholders, no story. |
| README and docs | 10 | 4 | README is short. 15+ stale docs contradict it. PROGRESS.md contains made-up numbers. |
| Reproducibility and deploy | 5 | 2 | Data is gitignored, the served model is not committed, the corpus is not reproducible, Railway would crash at boot. |
| Tests | 5 | 2.5 | Many tests check "status 200", and the key leakage tests are vacuous. |

The honest summary: you have a decent, honestly evaluated tabular predictor (Macro-F1 about 0.52 vs 0.22 baseline). It is wrapped in a UI and docs that over-claim, and the GNN, which is your headline, does not actually drive any prediction.

---

## 2. FINDINGS (evidence-based)

### 2A. Things that are false or fabricated (fix first)

| # | Where | Problem | Truth |
|---|---|---|---|
| F1 | `web/index.html`, Anti-Cheating tab | "Training 9,000 papers", "Validation 3,000 papers" | MODEL_CARD and `test_temporal_cutoff` say train 2012–2016 = **7,500**, val 2017 = **1,500**. The 9,000 and 3,000 were invented so the total reaches 16,500. |
| F2 | UI default paper and chip "Held-Out Test Paper (W2194775991)" | W2194775991 is ResNet, published **2016**. That is in the **training** split (≤2016). | It is not held-out. A correct "High" on a training paper proves nothing. |
| F3 | UI blind-test chips (W2798606413 "Nature 2018", W2963445205 "BERT NeurIPS 2018", W2991192534 "Lancet 2019", W2971253011 "CVPR 2019") | Labels are hand-typed and unverified. The IDs may not be those papers or may not be in the corpus. | Populate chips only from the test split in `works.parquet`, with titles read from data. |
| F4 | `scripts/update_ui.py` / `web/index.html` | Lift paragraph hard-codes "93.5%", "30 out of 464", "10.3%", "39.6%", "3.84x" | The values match today's JSON, but they are typed text. If the model changes, the page lies. Your own rule says no hard-coding. |
| F5 | `reports/PROGRESS.md` WP7 | Quotes SE 0.0245, t 8.65, CI [17.70, 29.77]; CS +12.4% p=0.012; Physics +11.7%; Eng +10.2% | `oa_advantage_metrics.json` says SE 0.01368, CS **+6.34%** p=0.042, Physics **+12.63%**, Eng **+4.52%**. The log was written by hand and is wrong. |
| F6 | `docs/MAJOR_PROJECT_SEMINAR_DOSSIER.md`, PPT prompt, AUDIT_REPORT, SEMINAR_HANDBOOK, IMPLEMENTATION slides | v1 content: 50 papers, 5 test papers, 5 features, "GNN 80% accuracy", "world's first", "70% of papers leak" (no source), "Nature studies show concise titles", "26 tests" | Contradicts v2 (16,500 papers, 47 features, GNN fails). Evaluators who open `docs/` will see two projects. |
| F7 | UI header and Anti-Cheating tab "Zero Temporal Leakage" | Too strong | Known residual: OA status and APC are as-of-today. Author and institution node counts use all years. Neighbor-paper features use the neighbor's own cutoff. See 2C. |
| F8 | UI "Empirical OA Association: +6.34% median boost" | Label says "median" | It is an adjusted OLS multiplier on log-citations for the field, not a median. |
| F9 | UI note "Crude: …, Adjusted: …" | Placeholders show as "…" on the first page (screenshot) | `fetchOaMetrics()` only runs when the Benchmarks tab is opened. |
| F10 | UI "(OLS, p<0.001)" | Hard-coded for all fields | Medicine adjusted effect is -1.9%, p=0.55, not significant. |
| F11 | UI control list "pub_year, primary_field, venue_quartile" | Wrong | The regression controls are field, year, log authors, SJR, SJR-known, top-tier flag. |
| F12 | UI "Forecasted Citations in Window: 13,925" for ResNet | Comes from the GNN regression head (log-MAE ≈ 1.2–2.9 in benchmarks, bad) | ResNet really got about 90k in 2019–2021. The headline number is wrong by about 6x and looks authoritative. |

### 2B. Cosmetic proofs that prove nothing (teachers will catch these)

1. **Poison test is rigged.** For corpus papers, `_lookup_local_corpus()` puts a precomputed `citations_at_cutoff` into the record, and `builder.py` uses it before looking at `counts_by_year`. The test only edits `counts_by_year` and `cited_by_count`, so Δ=0 is guaranteed by construction. The same applies to `audit_leakage.py` Audit 1 and `test_future_poison_invariance` (verify whether `data/cache/openalex/work_W2194775991.json` contains a `citations_at_cutoff` key). There is also **no positive control**: nobody shows that changing a pre-cutoff citation does change the prediction. A model that ignores citations also passes.
2. **"Blind forecast / Reveal" is theatre.** Both endpoints call the same `predict()` on the same server. Nothing is committed before the reveal. The server already knows the answer.
3. **"Parity gap 0.0000"** (`serving_parity_gap.json`): both F1s are 0.2371, which is about the majority baseline. `context_distribution` is `{"none": 500}`, so the ego graph was empty for every paper. The file is also stale, because `predict()` now returns Logistic Regression classes, not GNN classes.
4. **Attributions are invalid.** `inference_service.predict` computes `original_prob` from **Logistic Regression** but occludes features on the **GNN** and subtracts `GNN_prob_after − LR_prob`. That measures the gap between two models, not feature importance. The global importance file shows the GNN scoring Macro-F1 0.2233 (about majority) with drops of 0.0001–0.0006, so "sjr 42.9%" is noise normalised to 100%. `NUMBERS.md` presents this as a finding.
5. **Google Scholar button is not a check.** Scholar counts are cumulative, not per year, and are not comparable to OpenAlex window counts.
6. **Edge audit narrative** ("49.96% future edges pruned") describes the GNN graph, but the served class model is a graph-less LR.

### 2C. Real data and model integrity problems

- **Model selection on the test set.** `run_all.py` picks `empirical_best_model = max(macro_f1_mean on TEST)`, then serves that model. Choose on val, report test once.
- **Two different "M1" models.** Benchmark M1: C=10, no class weights, T=1.423, F1 0.5244. Served M1 (`fit_primary_model.py`): C=0.1, balanced, T=1.5637, F1 0.5275. The UI shows T=1.423/ECE 0.0318 on one tab and τ=1.5637 on another, and 0.527 vs 0.5244 across tabs.
- **Served model trained on 10,500 papers including 2010–2011** (`primary_model_meta.json`). Those papers have truncated `citations_at_cutoff` (counts start 2012). The audit fixed the GNN and benchmark train mask to 2012–2016 but not the served model.
- **Non-as-of-cutoff graph features** (`graph_builder.py`): author and institution node features are `log1p(count over ALL authorships, all years)`. Venue = contemporary DOAJ flag. Topic features = all zeros. Neighbor-paper features use the neighbor's own cutoff (can be years after the target's cutoff), which leaks future information through message passing.
- **Venue year mismatch.** Training enrichment (`ingest_scimago_and_core.py`) uses the nearest-year SJR, while serving (`venue_resolver`) uses latest year ≤ pub year. Feature parity is not exact for some papers. Also the CVPR proceedings in the screenshot resolve to "Unranked" because the name has no "(CVPR)" acronym, and the UI prints "UNRANKED (SCIMAGO)".
- **Backtest "actual class" ≠ evaluation label.** The live backtest uses `cohort_thresholds.json` (built from ≤2017 only, so 2018–2020 fall back to field-level thresholds). The confusion matrix uses `impact_class` (percentile within the true field-year cohort). A paper can show "Medium" in one place and "High" in another.
- **Weak baseline framing.** "55% vs 50% accuracy" and "0.527 vs 0.222" compare against the majority class. The real bar is "rank by early citations only". Add it.
- **Reproducibility.** `data/*` and `primary_model.joblib` are gitignored. `inference_service.load()` silently retrains the model at boot if it is missing, which violates your own "fail loudly" rule. `railway.json` would crash because there is no data. Ingestion has no sort or seed, so a re-run gives a different corpus.
- **XSS and UX bugs.** Titles go into `innerHTML` unescaped. `query.startsWith('W')` disables title search for any title starting with W. Errors use `alert()`. The page auto-runs a heavy prediction on load.

### 2D. GNN diagnosis (why it is at AUROC ≈ 0.5 and sometimes below)

1. **The GNN is not serving the prediction.** Class probabilities come from LR. The GNN supplies only the (bad) citation-count regression and (invalid) attributions.
2. **Unscaled inputs.** In `run_all.py` M4 and M5 train on raw `graph["paper"].x`, while M1/M2/M3 get `StandardScaler`-ed features. This is a plain bug.
3. **Graph is far too sparse.** 87,709 in-corpus paper→paper edges across 16,500 papers (about 5 per paper), and only 43,893 after the temporal filter (about 2.7). The corpus is a random 300-per-field-year sample, so a paper's references and citers are almost never in it. There is nothing for message passing to pass. Dataset design is the root cause.
4. **Auxiliary nodes carry almost no signal** (topic = zeros, venue = DOAJ bit, author/inst = a leaky count).
5. **Architecture washes out the strongest feature.** Mean aggregation over sparse neighbors dilutes the paper's own early-citation signal. There is no skip or fusion with the tabular model.
6. **Training recipe is weak.** Full-batch, 80 epochs, no class weights, early stop by noisy val F1, seed instability (M5 AUROC 0.41–0.60), regression MAE up to 2.9.
7. **Ego graph for serving is fake.** `ego_graph.py` appends `(0,0)` self-loops for every citation edge ("self-loop or neighbor index") and never adds neighbor paper nodes. Raw corpus rows have no `authorships`, so context = "none".
8. **The Cytoscape graph is partly invented.** Hop-2 institutions are hard-coded "Primary Academic Affiliation" / "Research Institute"; "Ref Concept 1/2/3" are placeholders; references show "Referenced Work (W10…)". It is not what the model sees.

---

## 3. WHAT TO PRESENT vs. WHAT TO STOP SAYING

**Safe, defensible headline (all from `heldout_predictions_summary.json`):**
> "Using only what was known 2 years after publication, we rank papers into Low / Medium / High (top-10%) citation tiers *relative to their own field and year*. On 4,500 papers we never trained on, Macro-F1 is 0.53 vs 0.22 for guessing. Papers we flag as High really are top-10% 40% of the time, against a 10% base rate (3.8× better than chance). We catch 60% of the true top-10% papers while flagging only 16% of papers as High (704 of 4,500). We are still wrong a lot (60% of 'High' flags are not top-10%), and we show every error."

**Say honestly about the GNN:** "Our first GNN did not beat the tabular model. Here is the diagnosis (sparse graph, preprocessing bug, leaky node features) and here is the fixed version's result." The result of the fixed version must come from running it, never from writing it.

**Stop saying / remove:** "Zero temporal leakage" (use "leakage-tested", list known residuals), "world's first", "70% of papers leak", "GNN predicts", "2-hop network shows the model's reasoning", "5-year citations" (it is **years 3–5 after publication**), "Held-Out Test Paper" for ResNet, 9,000/3,000, "Forecasted citations" as a precise number, "independently verifiable" until the receipt flow in §4 exists.

---

## 4. THE VERIFICATION DESIGN (this is what your teachers want)

Their question: "You predict. How do I know it's right?" Answer with three layers, all visible on one page.

### Layer 1 — Retrospective time-travel on papers whose answer is already known
For any held-out paper (published 2018–2020, never in training or tuning):
1. **Frozen inputs table** (what the model saw): citations per year up to the cutoff year only, venue, authors count, OA status, references count. Each row labelled "known by {cutoff year}".
2. **Prediction receipt**: tier, probabilities, model version, data-snapshot id, UTC time, and a `sha256` of the canonical JSON of (work_id, inputs, probs, model_version). Written to an append-only `reports/prediction_ledger.jsonl` **before** the truth is shown.
3. **Reveal**: raw OpenAlex `counts_by_year` rows for each year in the window (for example 2019 / 2020 / 2021 with the three numbers and their sum), the same-field-same-year cohort (size, p50, p90, a small histogram with this paper marked), and the rule that gives the actual tier.
4. **One-click independent check**: link to the exact API URL `https://api.openalex.org/works/W…?select=id,publication_year,counts_by_year`. A teacher opens it, adds three numbers and compares. Remove the Google Scholar button, or label it "not comparable".
5. **Re-hash button** proves the stored prediction was not edited after the reveal.

### Layer 2 — Bulk proof on all 4,500 test papers
- Download `heldout_predictions.csv` with: work_id, openalex_url, api_url, title, pub_year, cutoff_year, inputs used, probs, predicted tier, window counts per year, actual tier, correct?.
- `python scripts/verify_paper.py W…` fetches live from OpenAlex, recomputes the actual tier **using only raw JSON**, and compares with the stored row. `python scripts/verify_sample.py --n 20` spot-checks 20 random rows live, printing the match rate. The UI has a "Spot-check 20 random papers against OpenAlex now" button that shows the same table.
- Scoreboard compares against **three** simple guesses: always-majority, "rank by early citations only", "venue tier only". The model must visibly beat the best of them, or you say it doesn't.

### Layer 3 — Prospective (the real "prediction" proof)
- Run the model today on recent papers (for example published 2023), write hashes to the ledger, commit and tag in git (`prereg-2026-10-17`). When the window closes, score the pre-registered predictions. Show "Pre-registered: N papers, scored when window closes in YYYY". Be honest that this proves nothing until it closes.

### Leakage proof, rewritten to be real (merged into the same page, plain language)
- **Positive control**: change a citation count **inside** the allowed window → prediction changes (shown).
- **Negative control**: change counts **after** the cutoff in the **raw source** (before feature building, through the real pipeline, not just a field the builder ignores) → prediction unchanged.
- **Label-shuffle** result shown with its meaning ("scrambling the answers drops us to chance, so the model isn't peeking").
- **Graph time audit**: for the sampled paper, "every neighbor used was published on or before {cutoff}".

---

## 5. UI REWRITE SPEC (non-technical user first)

**Principle**: every number gets (a) a plain label, (b) one sentence "what this means", (c) a "how to check" link. Jargon (T_cutoff, ΔP, OvR, ECE, τ, `(pub+3..pub+5]`) goes into a collapsible "For experts" section.

**Page order (single scroll page + 3 tabs, not 5):**
0. **Hero**: "Will this paper become one of the most cited in its field?" + 3-step strip (1. We freeze what was known 2 years after publication → 2. The model guesses → 3. We compare with what actually happened, from OpenAlex).
1. **Try a paper** (default = a random held-out paper, with a green badge "Held-out: the model never saw its outcome"; a training paper gets an amber badge "In training data, not valid evidence").
   - Result sentence: "We think this paper is **likely in the top 10%** (62% confident)."
   - Side-by-side: **We predicted** vs **What actually happened** (hidden until "Show what really happened").
   - "What the model looked at" (plain labels, real attribution only after the fix), "Citations in years 3–5" as a **range or percentile bucket**, not a fake exact number.
2. **Did it work?** (merge Verification Lab + Anti-Cheating): scoreboard in plain English with the 3 baselines, confusion matrix with sentences ("Of 464 papers that truly became top-10%, we flagged 279 as High"), the receipt flow (§4), the spot-check button, CSV download, the leakage controls.
3. **How it works and limits**: pipeline diagram, model comparison table including the **fixed GNN** with a one-line verdict each, OA analysis with the "association not causation" note, data snapshot date, known limits.
4. **Compare two papers** stays, with the same plain language.

**Specific bugs from the screenshot to fix:**
- Graph: giant overlapping labels. Use `breadthfirst` or `concentric` layout, short labels, full text in a side panel, legend filters, and hide hop-2 by default.
- **Real graph data only** (see Prompt 4). Out-of-corpus references appear as one gray stub "76 more references (not in our dataset)", never as fake nodes.
- "UNRANKED (SCIMAGO)" → "No ranking found in SCImago or CORE".
- "None reported" APC → "Not listed".
- Venue name printed twice → de-duplicate.
- Blank "Crude: …, Adjusted: …" → load OA metrics on first paint.
- Replace `alert()` with inline errors. Escape all text (`textContent`). Fix the "W-prefix" search bug. Do not auto-run a prediction on load without a clear loading state.
- Header: remove "Zero Temporal Leakage"; use "Leakage-tested → see proof".
- Add glossary tooltips, mobile layout, keyboard focus, colour-blind-safe palette (don't rely on cyan vs blue vs purple).
- Rename "5-year" everywhere to "citations in years 3–5 after publication".

---

## 6. TEST PLAN (add these; delete or rewrite the vacuous ones)

| ID | Test | Pass condition |
|---|---|---|
| T1 | Negative control through raw pipeline: mutate raw OpenAlex JSON `counts_by_year` after cutoff, rebuild features via the same path as ingestion | features byte-identical |
| T2 | Positive control: add +50 citations inside the cutoff window | `citations_at_cutoff` and prediction change |
| T3 | Snapshot-graph audit: for 200 random targets, max(time) of every node feature and edge ≤ cutoff | zero violations |
| T4 | Independent label recompute from raw `counts_by_year` + cohort percentile, compared with stored `impact_class` | 100% match |
| T5 | Live OpenAlex spot-check (20 papers, marked `network`) | ≥95% window sums match (document tolerance for OpenAlex drift) |
| T6 | Split integrity: no work_id overlap, years respect train ≤2016 (2012–2016 only), val 2017, test ≥2018, nothing from 2010–2011 in any training path including the served model | pass |
| T7 | Serve/batch parity: `predict()` vs batch for 200 test papers | max prob diff <1e-5 and 100% same class |
| T8 | HTML hard-coding scan: no numeric literal with %, ×, or decimal inside `web/` outside a whitelist | pass |
| T9 | UI e2e (Playwright): load page, run default paper, assert no visible "…", "undefined", "NaN", "null", "Author 1" | pass |
| T10 | Attribution sanity: same model for probability and occlusion; occluding all groups moves prediction toward cohort mean | pass |
| T11 | Reproducibility: run benchmark twice with the same seed | identical metrics; data checksum matches `data/MANIFEST.sha256` |
| T12 | GNN gate (report, not hard fail): val Macro-F1 of GNN-fusion vs M1 over 5 seeds | printed to `reports/gnn_gate.json` |
| T13 | Model selection uses val only | assert in code; test reads `run_all` output field `selected_on: "val"` |
| T14 | Boot fails loudly if checkpoint or data missing (no silent refit) | pass |

---

## 7. AGENT PROMPTS (paste one at a time, in order, and make the agent finish and show test output before the next)

### PROMPT 0 — Ground rules (paste at the start of every session)

```
You are fixing the SciGraph AI 2.0 repo. Rules you must follow:
1. Never invent a number. Every number shown anywhere (UI, README, docs, reports) must be read from a generated file in reports/ or data/ at runtime or at doc-generation time. If a result is not produced yet, show "not yet computed", never a guess.
2. Do not edit any result JSON by hand. Re-run the script that generates it.
3. Never choose a model, hyperparameter, threshold or temperature using the TEST split. Use VAL only. Test is reported once.
4. After every change, run: pytest tests -v --ignore=tests/legacy. Paste the real output. If a test fails, fix the code, not the test, unless the test is provably wrong, in which case explain why.
5. If something I describe does not match what you find in the code, STOP and tell me what you found. Do not silently adapt.
6. Work on one numbered task at a time. At the end of each task print: files changed, command run, real output, and what is still unverified.
7. Do not delete files. Move stale files to legacy_archive/ or docs/legacy_v1/ and add a banner saying they are obsolete.
```

### PROMPT 1 — Integrity fixes (do first, small and fast)

```
TASK 1. Fix false / fabricated content.

1a. web/index.html and scripts/update_ui.py: delete every hard-coded metric string. Specifically remove: "9,000 Papers", "3,000 Papers", "(τ = 1.5637)", "93.5%", "30 out of 464", "10.3%", "39.6%", "3.84x", "87,709", "43,893", "43,816", "49.96%", "PASSED" defaults, "0.222 baseline", "50% baseline". Replace each with an element filled from the API (add data-metric="..." attributes). Split sizes must come from a new endpoint GET /api/v1/meta/splits that counts rows of data/processed/train|val|test.parquet and the train mask used by the served model. Expected truth: train 2012-2016 = 7,500, val 2017 = 1,500, test 2018-2020 = 4,500. If the numbers differ, show what the code reports.
1b. The default paper (W2194775991, ResNet, published 2016) is in the TRAINING split. Change the default and the "Held-Out Test Paper" chip to a paper chosen from the test split at runtime via /api/v1/papers/corpus-sample. Replace the four hand-labelled chips (W2798606413, W2963445205, W2991192534, W2971253011) with chips generated from the test split with titles read from data. Add a badge on every prediction: green "Held-out test paper" if split == "test"; amber "In training/validation data - not valid evidence" otherwise. predict() already returns split and in_training_data, so use them.
1c. Call fetchOaMetrics() on first page load so the "Crude / Adjusted" placeholders are never blank. Fix the label "median boost" to "estimated citation advantage (adjusted for field, year, authors, venue prestige)". Replace the hard-coded "(OLS, p<0.001)" with the real p-value per field from reports/oa_advantage_metrics.json; if p>=0.05, show "not statistically significant". Fix the control list text to: field, year, log(authors), SJR (+known flag), top-tier venue flag.
1d. Venue label: if ranking_system is null show "No ranking found in SCImago or CORE" (not "Unranked (SCImago)"). De-duplicate venue names like "X - X". Fix ml/scientometrics/venue_resolver.py so IEEE CVPR proceedings resolve via CORE (match by acronym tokens and by normalised title containing "computer vision and pattern recognition"); add a test using the real venue_name string from the corpus.
1e. Remove from the UI header and tabs the phrases "Zero Temporal Leakage" and "Independently Verifiable". Use "Leakage-tested (see proof)". Replace "5-year" with "citations in years 3-5 after publication" everywhere in code, API field descriptions and UI text. Keep the field name future_citations_5y internally but add the alias citations_window in API output.
1f. Escape all dynamic text: use textContent or an escapeHtml() helper for titles, authors, venue names. Fix the search bug (startsWith('W')): only skip autocomplete when the input matches /^W\d+$/i or starts with "10.". Replace alert() with an inline error banner.
1g. reports/PROGRESS.md: regenerate the WP7 block from reports/oa_advantage_metrics.json (the current text has wrong SE, CI and per-field numbers). Add a header note: "Numbers below are copied from reports at the time shown; reports/*.json is authoritative."
1h. Move these to docs/legacy_v1/ with a red banner "OBSOLETE - v1 prototype (50 papers). Do not cite.": AUDIT_REPORT.md, SEMINAR_HANDBOOK.md, MAJOR_PROJECT_SEMINAR_DOSSIER.md, MAJOR_PROJECT_PPT_MASTER_PROMPT.md, IMPLEMENTATION_AND_PIPELINE_SLIDES.md, MATHEMATICAL_SPECIFICATION_AND_FLOWCHART.md, SciGraph_Mathematical_Specification.html, RESULTS.md, EXPERIMENTS.md, MODEL_CARD.md, API_SPEC.md, GNN_AND_PYG_PRIMER.md (keep the primer but fix its example to say it is illustrative). Do not leave any claim like "world's first", "70% of papers", or "GNN 80%" outside legacy_v1.

Acceptance: grep -rE "9,000|3,000 Papers|world's first|70%|Zero Temporal Leakage" web/ README.md docs/ (excluding docs/legacy_v1) returns nothing. pytest passes.
```

### PROMPT 2 — Honest model selection and one served model

```
TASK 2. One model, one set of numbers, no test-set selection.

2a. ml/evaluation/run_all.py: choose empirical_best_model using VAL Macro-F1 (compute val metrics for every model), store "selected_on": "val" in benchmark_latest.json, and report test metrics unchanged. Also record val metrics in the JSON.
2b. Unify M1: make fit_primary_model.py and run_all.py use the SAME code path and hyperparameter search (C from [0.01,0.1,1,10], class_weight decision made on val, temperature on val). Write the served model's test metrics into reports/benchmark_latest.json under the same key so the UI, README and scoreboard all show identical numbers (single source: reports/benchmark_latest.json and reports/heldout_predictions_summary.json generated from the same run).
2c. The served model must be trained on 2012-2016 only (exclude 2010 and 2011, whose citations_at_cutoff are truncated because OpenAlex counts_by_year starts in 2012). Add an assert in fit_primary_model.py that train years are within 2012..2016 and that the count is 7,500. Fix primary_model_meta.json text (it currently says 10,500).
2d. ml/inference_service.py load(): if primary_model.joblib or data is missing, FAIL LOUDLY with a clear error. Remove the silent auto-refit. Add scripts/train_all.sh (or make train) that does everything explicitly.
2e. Add two simple baselines to the benchmark and scoreboard: B1 "rank by citations at cutoff only" (map the percentile of log citations_at_cutoff within its field-year cohort to the three tiers using train-derived cutoffs) and B2 "venue tier only". Save in benchmark_latest.json. The UI scoreboard must show model vs majority vs B1 vs B2.
2f. Fix the backtest class mismatch: the "actual class" shown for a paper must equal the stored impact_class from works.parquet when the paper is in the corpus. For papers outside the corpus compute it from cohort_thresholds.json and label it "estimated using cohort thresholds from 2012-2017".
2g. Add data/MANIFEST.sha256 (checksums of every parquet) and a "data_snapshot": {"openalex_retrieved_utc": ..., "manifest_sha256": ...} block in benchmark_latest.json and in /health. Show the snapshot date in the UI footer.

Acceptance: README, UI scoreboard, Benchmarks tab and /health show the same Macro-F1 for the served model to 4 decimals. T6, T13, T14 from the test plan pass.
```

### PROMPT 3 — Verification system (the thing teachers want)

```
TASK 3. Build the verification receipt flow and real leakage controls. Read section 4 of SCIGRAPH_AUDIT_AND_AGENT_PROMPTS.md first.

3a. New module ml/verification/receipt.py:
  - build_receipt(work_id) -> dict with: work_id, openalex_url, api_url (https://api.openalex.org/works/{id}?select=id,publication_year,counts_by_year), title, pub_year, cutoff_year, split, frozen_inputs (the exact feature inputs in human-readable form, only values known by cutoff_year, with the citation counts per year <= cutoff_year), model_version, data_snapshot_id, probs, predicted_tier, created_utc, receipt_sha256 (sha256 of canonical JSON of the fields above).
  - append it to reports/prediction_ledger.jsonl (append-only; never rewrite lines).
  - build_reveal(work_id) -> window years and per-year counts (raw counts_by_year rows for years cutoff+1..pub+5), their sum, field-year cohort (size, p50, p90, the cohort values histogram bins), actual_tier, correct?, and recompute_receipt_hash() so the UI can show "receipt unchanged".
3b. Endpoints under /api/v1/verify: POST /receipt/{work_id}, GET /reveal/{work_id}, GET /receipt/{hash}/check, GET /heldout.csv (all 4,500 test rows with the columns listed in section 4 layer 2), GET /spotcheck?n=20 (picks n random test papers, fetches each LIVE from api.openalex.org, recomputes the window sum and tier from raw JSON only, and returns stored vs live side by side with match flags and match rate; handle network errors with a clear message, never fake a match).
3c. Scripts: scripts/verify_paper.py W... and scripts/verify_sample.py --n 20 that do the same independently of the web server, with their own minimal code (do NOT import inference_service for recomputing the truth). Print a table.
3d. Replace the poison test. New endpoint GET /api/v1/verify/leakage-controls/{work_id} that runs through the REAL feature pipeline (the same function used by ingestion, ml/features/builder.py with a raw OpenAlex-shaped dict that has NO precomputed citations_at_cutoff key):
  - negative control: add +100,000 citations to years after cutoff in raw counts_by_year and to cited_by_count -> features and probabilities must be identical;
  - positive control: add +50 citations in a year <= cutoff -> citations_at_cutoff and probabilities must change;
  - report both with the actual numbers. Do not write the expected delta in the page text before the test runs.
  Also fix ml/scientometrics ingestion/_lookup_local_corpus so a raw dict built from the corpus does not smuggle in citations_at_cutoff: the builder must compute it from counts_by_year when counts_by_year is present, and only use a precomputed column for parquet rows. Update scripts/audit_leakage.py Audit 1 and tests/test_temporal_cutoff.py to use the same non-vacuous procedure and to include the positive control.
3e. Keep the label-shuffle result but show it with the plain sentence in the UI: "When we scramble the answers, accuracy falls to chance (Macro-F1 x vs y), so the model is not peeking."
3f. Remove the Google Scholar "cross-check" or relabel it "not comparable (different counting)". The OpenAlex link must point to the API URL that shows counts_by_year.
3g. Tests: T1, T2, T4, T5, T9 (receipt part). Add a test that tampering with a ledger line makes /receipt/{hash}/check fail.

Acceptance: from the UI I can pick a held-out paper, click "Freeze and predict", see inputs + receipt hash, click "Show what really happened", see the three yearly numbers, their sum, the cohort thresholds, the API link, and a green "receipt unchanged". The spot-check button returns real live results.
```

### PROMPT 4 — Make the GNN real and make it work (read carefully; this is the project's core)

```
TASK 4. Rebuild the GNN so it is correct, leak-free and actually used. Do steps in order and report the real numbers after each. Do NOT claim improvement unless the validation numbers show it. Never tune on test.

4a. Diagnose first and print to reports/gnn_diagnosis.json:
  - fraction of target papers with 0, 1-2, 3-9, 10+ in-corpus neighbors (citing + cited) AS OF their own cutoff; mean/median degree; share of authors with >1 paper in corpus; share of venues with >20 papers.
  - confirm bug: run_all.py feeds raw (unscaled) paper features to M4/M5. Fix by applying the SAME StandardScaler fitted on train to the paper node matrix before training M4/M5.
  - confirm bug: ego_graph.py adds self-loops instead of neighbor nodes.

4b. Fix the data so a graph can help. Create scripts/ingest_dense_corpus.py that builds a dense citation neighborhood instead of a random sample:
  - choose ONE or TWO subfields (for example OpenAlex subfield "Artificial Intelligence" and "Computational Theory/Genetics" - pick based on what OpenAlex returns), pull ALL (or a large number of) works published 2010-2022 in that subfield, keep 'referenced_works', authorships with institution ids, primary_topic, counts_by_year;
  - also pull works that cite or are cited by your seed papers (filter=cites:W... / referenced_works) so papers have many in-corpus neighbors;
  - use a fixed sort (cited_by_count desc is NOT allowed for targets because it biases labels; use publication_date and cursor paging with a fixed seed for any subsampling) and record retrieved_utc;
  - target: median in-corpus degree >= 15 and >= 70% of target papers with >= 3 neighbors. Print the diagnosis from 4a on the new corpus. If the target is not met, say so and stop; do not proceed pretending.
  Keep the old 16,500 corpus as "v1 sample" for comparison. Splits stay temporal: train pub 2012-2016, val 2017, test 2018-2020.

4c. Time-correct graphs. Replace the single global graph with SNAPSHOT graphs, one per cutoff year t (t = pub_year + 2):
  - G_t contains only papers published <= t; paper features are computed AS OF t (citations_at_cutoff for a neighbor = citations up to year t, not up to the neighbor's own cutoff);
  - edges only if the citing paper was published <= t;
  - author node features = as-of-t stats (number of papers by the author published <= t, sum and mean of citations those papers had by t, years active); institution = same aggregates; venue = number of papers <= t and mean citations by t (+ SJR as-of value, CORE rank); topic = number of papers <= t, mean citations by t. NO all-years counts, NO DOAJ contemporary flag unless labelled as-of-today and ablated;
  - targets for G_t are papers with pub_year = t - 2;
  - train on G_2014..G_2018, validate on G_2019, test on G_2020..G_2022 snapshots.
  Add tests/test_snapshot_graph_time.py: for 200 random targets assert every node feature and edge in the target's snapshot has time <= its cutoff (T3).

4d. Also add as-of "graph-derived tabular features" to the tabular baselines (this is the fair comparison and a quick win): mean and max of cited references' citation counts at cutoff, number of in-corpus citers at cutoff, authors' prior paper count and mean prior citations as-of cutoff, venue mean citations as-of cutoff. Re-run M1 and M2 with them as M1g and M2g. Report the gain. This tells us how much signal the graph holds.

4e. Model: new class FusionHeteroSAGE in ml/gnn/models/hetero_gnn.py:
  - paper input = standardized features (47 dims + the new graph-derived ones);
  - 2 HeteroConv layers (SAGEConv with root_weight=True, aggr='sum' across relations, LayerNorm, dropout 0.3, residual connections);
  - final classifier takes concat[ h_paper_after_input_projection (skip), h_paper_after_graph ] so the strong early-citation signal cannot be washed out;
  - optionally add the tabular LR logits as an input feature (stacking) trained only on train folds (use out-of-fold LR logits to avoid leakage);
  - heads: 3-class classification (weighted cross-entropy with class weights from train) + log-citation regression (weight 0.2).
  - training: PyG NeighborLoader mini-batches with temporal sampling, 200 epochs, AdamW, lr scheduler, early stopping on VAL macro-F1 AND val NLL with patience 20, 5 seeds, save best-by-val checkpoint per seed. Fit temperature on val only.
  - run_all.py: M5 = FusionHeteroSAGE. Keep the old M5 as M5_old for the record.

4f. Honest ablation ladder, saved to reports/gnn_ablation.json and shown in the UI "How it works" page: majority, B1 early-citations-only, M1, M2, M1g, M2g, graphless MLP, homogeneous SAGE (fixed), hetero SAGE without author edges, without venue edges, without citation edges, full FusionHeteroSAGE. Each row: val and test Macro-F1, AUROC, 95% bootstrap CI, mean±std over 5 seeds. Use a paired bootstrap on test for differences vs M1g.
  Decision rule written in code: the served model is whichever has the best VAL Macro-F1. If FusionHeteroSAGE wins on VAL, serve it. If it does not, serve the tabular winner and the UI says exactly: "The GNN did not beat the tabular model on this data because <numbers from gnn_diagnosis.json>." Do not tune anything to make the GNN win.

4g. Serving: replace ml/gnn/ego_graph.py. For a corpus paper take the k-hop subgraph (k=2, torch_geometric.utils.k_hop_subgraph) from the paper's cutoff-year snapshot graph; for a non-corpus paper build neighbors from the raw OpenAlex JSON (referenced works and authors with real ids), look their features up in the snapshot tables, and report graph_context = full/partial/none honestly with counts ("12 of 79 references found in our data"). Add tests/test_serving_parity.py (T7): predict() equals batch inference for 200 test papers (max prob diff < 1e-5, 100% same class). Regenerate reports/serving_parity_gap.json from the real served model.

4h. Fix attributions: compute_local_occlusion_attributions must use the SAME model and the SAME probability for original and occluded runs. For the served model: occlusion on feature groups (replace by train mean), plus relation-level ablation for the GNN (drop all author edges / venue edges / citation edges and measure change in P(class)). Return both under "feature_groups" and "relations". Regenerate reports/global_importance.json on the served model and drop the NUMBERS.md "top 5" if the base model is near-constant. Add T10.

4i. Regression head: evaluate MAE in log space against the cohort-median baseline. If the head is not better than that baseline on val, remove "Forecasted Citations" from the UI and API, and show only the tier probabilities and "expected percentile range". Never show an exact citation count the model cannot back up.

Acceptance: reports/gnn_diagnosis.json, reports/gnn_ablation.json, reports/gnn_gate.json exist with real numbers; the UI "How it works" page shows the ablation table with one-line plain-English verdicts; T3, T7, T10, T12 pass. Final message must say plainly whether the GNN beat the tabular model on VAL (and on TEST), with the numbers.
```

### PROMPT 5 — UI rewrite for non-technical users

```
TASK 5. Rewrite web/index.html (and scripts/update_ui.py generator or delete the generator and keep index.html + web/app.js + web/styles.css as real files). Follow section 5 of SCIGRAPH_AUDIT_AND_AGENT_PROMPTS.md exactly.

Structure: Hero -> "Try a paper" -> "Did it work? (Proof)" -> "How it works and limits" -> "Compare". Three top tabs max: Try / Proof / How it works (Compare lives inside Try as a "Compare with another paper" panel).

Rules:
- Every metric card has: a plain title, the value (from API), one sentence "What this means", and a "How to check" link (to the receipt, CSV or script).
- Put technical terms inside <details><summary>For experts</summary>. Add a glossary drawer with: cutoff, tier, cohort, calibration, held-out, lift, Macro-F1, AUROC, CD index, OA advantage.
- "Try a paper": result sentence, side-by-side "We predicted" vs "What actually happened (hidden until you click)". The actual-outcome panel is the receipt reveal from Task 3 (three yearly numbers, sum, cohort thresholds, API link, receipt hash check).
- "Proof" page, in this order: (1) 3-step diagram of how we test; (2) scoreboard vs three simple guesses with sentences; (3) confusion matrix with a sentence per row, generated from the data; (4) try-one-paper blind->reveal; (5) "Check our work yourself": spot-check button, CSV download, copy-paste command for scripts/verify_paper.py; (6) leakage controls (negative + positive) with plain sentences; (7) known limits.
- Graph: real data only from a new endpoint GET /api/v1/papers/{id}/graph returning nodes and edges from the tables: real author names, real institution names (institutions.parquet), real venue, real topic, real references and citers (as-of cutoff) with titles when in corpus. Out-of-corpus references collapse into one gray stub node "N more references not in our dataset". Include coverage text "X of Y references are inside our data". Layout: concentric (target in the centre), short labels (max 18 chars), full detail in the side panel, legend as filter checkboxes, hop-2 hidden by default behind a toggle, PNG export kept. Under the graph, show "What the model used from this network" taken from the relation ablation (Task 4h) or "The served model does not use the network; here it is for context" if the served model is tabular. Never draw invented nodes.
- Accessibility and quality: escape all text, no alert(), inline errors, loading skeletons, keyboard focus, aria-labels, mobile layout at 375px, colour-blind-safe palette with shapes/icons in addition to colours, minimum 14px body text, max line length 75ch.
- No numeric literal with % or decimals in the HTML/JS source except in a whitelist file web/allowed_literals.json. Add tests T8 and T9 (Playwright).
- Footer on every page: data snapshot date, model version, git commit, link to README "How to verify".

Acceptance: take screenshots of all three tabs at 1440px and 375px and attach them. A person who has never seen the project must be able to answer, from the page alone: what is predicted, how good it is compared to guessing, where the "real" numbers come from, and how to check one paper themselves.
```

### PROMPT 6 — README and documentation

```
TASK 6. Rewrite README.md and make docs consistent. README must be generated or checked so no number is hand-typed: add scripts/make_readme_numbers.py that fills {{placeholders}} from reports/*.json and fails if a placeholder is unfilled. Sections, in this order:

1. What this is (3 plain sentences) and a screenshot.
2. The question being answered, with the exact definitions: cutoff = pub_year + 2; window = citations in years pub+3, pub+4, pub+5; tiers = Low (<50th), Medium (50-90th), High (>=90th percentile within the same field and publication year); how the cohort percentile is computed.
3. Results (auto-filled): model vs majority vs early-citations-only vs venue-only, with 95% CIs; what the model gets wrong; the GNN ablation table and the plain verdict.
4. How to verify (the most important section): Layer 1 per-paper receipt (UI steps), Layer 2 `python scripts/verify_paper.py W...` and `python scripts/verify_sample.py --n 20` and the CSV, Layer 3 the pre-registration ledger and git tag. Say exactly where the "real" numbers come from: OpenAlex `counts_by_year`, retrieved on <date>, snapshot checksum, API URL pattern. Say which data the model could not have seen and how that is enforced and tested.
5. Leakage: what is enforced, how it is tested (T1-T4, label-shuffle, snapshot-graph audit), and the known residual as-of-today fields (OA status, APC, SJR year) with their measured effect.
6. Data: how the corpus was built (dense subfield corpus, dates, filters), splits with counts read from data, where to download the snapshot (Zenodo/HF/Release link) and its checksum.
7. Models: every model M0..M5 plus B1/B2/M1g/M2g in one table, what each uses, what is served and why (decision rule from Task 4f).
8. Install and run from scratch: python version, pip install -r requirements.txt (add requirements.txt with pinned versions incl. torch, torch_geometric), data download, `make train`, `make test`, `make serve`, `make verify`, expected outputs and runtimes.
9. API reference (table of endpoints with example responses taken from the live server).
10. UI guide with a page-by-page walkthrough for non-technical readers.
11. Repo map, deployment (Railway needs data + checkpoint: document how they are fetched at build; the app must refuse to start if missing), known limitations, corrections history (keep the retraction notice), license, how to cite.
Docs: keep docs/NUMBERS.md generated. Add docs/GLOSSARY.md (plain English). Banner every file in docs/legacy_v1. Remove claims listed in section 3 of the audit file. Add docs/TEACHER_QA.md with the 10 hardest questions ("how do I know it's not cheating?", "why is the GNN worse?", "where does the real answer come from?", "is 55% good?", "what if OpenAlex changes?") each answered with a link to the exact UI step or command.
Acceptance: a new person can clone, follow the README, reproduce reports/benchmark_latest.json within tolerance, and run the verification commands without asking anything.
```

---

## 8. ORDER OF WORK AND WHAT TO EXPECT

1. Prompt 1, then 2 (about a day): removes every false statement and makes numbers consistent.
2. Prompt 3: gives you the "answer-first, then predict, then compare" proof your teachers asked for.
3. Prompt 4: GNN rebuild. This is the risky one. **It may still not beat the tabular model.** If a dense corpus plus fusion architecture does not win on VAL, the honest, defensible story is "we diagnosed why, here is the ablation proving the graph adds X or nothing." Do not let the agent hide that.
4. Prompts 5 and 6: UI and README last, because they depend on the final numbers.

Quick reality check on how far to trust the GNN narrative: the fixed hetero GNN needs a dense graph. If your final corpus still has median in-corpus degree below about 10, expect little gain whatever the architecture.
