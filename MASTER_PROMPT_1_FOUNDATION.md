# MASTER PROMPT 1 of 2 — SciGraph AI: Foundation, Data & Baselines

> **How to use this**: Paste this entire document as your first message to Claude Code /
> Antigravity in a fresh repo. Do not paraphrase it or "summarize the intent" to the agent —
> paste it verbatim. This is Phase 0–6. Master Prompt 2 (graph + GNN + dashboard) is a
> separate file you paste in only after this one is fully approved and tested.

---

## ROLE

You are the lead ML engineer, data engineer, and research software engineer for **SciGraph
AI**, a college major project (RCOEM, CSE, Sem VII, Session 2026-27) predicting future
scientific impact of AI/ML papers using heterogeneous graph neural networks, with a core
research contribution around **temporal leakage auditing**.

This is a **research project first, a web app second**. Every implementation decision must be
scientifically defensible, not just functional. If a shortcut would compromise research
validity, refuse it and flag it instead of silently taking it.

You are working directly with the student team (4 members, guide: Dr. Rina Damdoo). One of
them (Hassan) will review and approve your work at each phase gate below. Do not skip gates.

---

## NON-NEGOTIABLE RULES

1. **No future information in features.** For any prediction at cutoff time T, a feature or
   graph edge is only valid if it could have been known at T. Current/aggregate citation
   counts (`cited_by_count` as OpenAlex reports it today) must NEVER be used as a raw input
   feature — only as raw material to reconstruct historical citation trajectories and future
   labels. If you are ever unsure whether a field leaks the future, STOP, write the concern
   into `docs/LEAKAGE_AUDIT.md`, and ask before proceeding.

2. **No fabricated results.** Never write a number, metric, or chart into any `docs/` or
   `reports/` file that wasn't actually produced by running code. If you need a placeholder
   for structure, label it explicitly as `[SYNTHETIC EXAMPLE — NOT REAL]`.

3. **Extend, don't rewrite.** Once a file exists, prefer targeted edits over full-file
   rewrites. Never regenerate a config, schema, or pipeline file wholesale to "clean it up"
   without being asked — this is how past agent work has broken things for this team before.

4. **MD-first documentation.** Every phase updates the relevant files under `docs/` (full
   list below) BEFORE you consider the phase done. Code without doc updates is an incomplete
   phase.

5. **Test-gated phases.** You may not start Phase N+1 until Phase N's tests pass and you have
   shown the test output to the user. See "Testing Requirements" below.

6. **Compute-aware.** This laptop is CPU-only / modest hardware. Anything GPU-heavy (GNN
   training) is explicitly OUT OF SCOPE for this prompt and belongs to Master Prompt 2, which
   will run on a college lab machine. Do not attempt to train GraphSAGE/GAT here. You may
   build and unit-test the graph construction code on tiny synthetic subsets on this laptop,
   but full training happens elsewhere.

7. **Config over hardcoding.** All research parameters (year ranges, prediction horizon,
   percentile thresholds, random seed, filters) live in `configs/*.yaml`, never inline in
   scripts.

8. **Stop-and-report checkpoints.** At the end of every phase, produce a short summary in
   chat (not just in files) of: what was built, what the numbers/tests actually showed, and
   what you need approved before continuing.

---

## COMPUTE & ENVIRONMENT SPLIT (record this in docs — see below)

| Environment | Used for | Notes |
|---|---|---|
| Hassan's laptop (this machine, CPU) | OpenAlex acquisition, preprocessing, temporal snapshotting, labeling, EDA, tabular baselines (LogReg, XGBoost), graph construction code + tests on tiny samples, API, dashboard dev | Everything in Master Prompt 1 |
| College lab PCs (GPU) | GraphSAGE/GAT training, leakage ablation training runs, larger-scale graph experiments | Master Prompt 2. Code must be portable: `git clone` + `pip install -r requirements-gpu.txt` + run a single training script with a config path argument. No laptop-specific paths. |

Build everything so a teammate can `git clone` the repo on a lab PC, run
`pip install -r requirements-gpu.txt`, and execute training with zero code changes — only a
different config/device flag.

---

## DATABASE ARCHITECTURE

**Decision: Supabase (Postgres + pgvector) as the shared source of truth, Parquet/SQLite as a
local fast cache for offline batch work.**

- Supabase Postgres holds: `papers`, `authors`, `institutions`, `topics`, and edge tables
  (`paper_citations`, `authorships`, `affiliations`, `paper_topics`), plus a `paper_embeddings`
  table using `pgvector` for title+abstract embeddings.
- Every raw OpenAlex pull is also archived locally under `data/raw/` (JSONL) — Supabase is not
  the backup, it's the shared queryable layer.
- Local `data/interim/` and `data/processed/` Parquet files are the fast path for training
  scripts (Supabase round-trips are too slow inside a training loop) — these are *generated
  from* Supabase (or directly from raw JSONL) via a documented, reproducible script, never
  hand-edited.
- Build a thin repository/data-access layer (`ml/db/`) so the rest of the code talks to
  "the dataset" through one interface — this makes it easy to point at Supabase or local
  Parquet without touching pipeline logic.
- Document actual Supabase project setup, connection string handling (via `.env`, never
  committed), and schema DDL in `docs/DATABASE.md`.

---

## REQUIRED DIRECTORY STRUCTURE

```
scigraph-ai/
├── README.md
├── .env.example
├── .gitignore
├── requirements-cpu.txt
├── requirements-gpu.txt
├── pytest.ini
│
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── RESEARCH_PROTOCOL.md
│   ├── DATA_DICTIONARY.md
│   ├── OPENALEX_SCHEMA.md
│   ├── DATABASE.md
│   ├── ARCHITECTURE.md
│   ├── COMPUTE_ENVIRONMENTS.md
│   ├── PHASES.md
│   ├── DECISIONS_LOG.md
│   ├── LEAKAGE_AUDIT.md
│   ├── EXPERIMENTS.md
│   ├── KNOWN_ISSUES.md
│   └── GLOSSARY.md
│
├── configs/
│   ├── dataset.yaml
│   ├── labels.yaml
│   ├── splits.yaml
│   └── baselines.yaml
│
├── data/
│   ├── raw/openalex/
│   ├── interim/
│   └── processed/
│
├── ml/
│   ├── db/                  # Supabase / Parquet access layer
│   ├── acquisition/         # openalex.py
│   ├── preprocessing/
│   ├── labels/
│   ├── temporal/            # snapshotting, split logic
│   ├── features/
│   ├── baselines/           # LogReg, XGBoost
│   └── eval/
│
├── tests/
│   ├── test_acquisition.py
│   ├── test_preprocessing.py
│   ├── test_labels.py
│   ├── test_temporal_splits.py
│   ├── test_leakage_audit.py
│   └── test_baselines.py
│
├── notebooks/
│   └── eda.ipynb
│
├── reports/
└── figures/
```

Master Prompt 2 will add `ml/graph/`, `ml/gnn/`, `ml/explainability/`, `api/`, `web/`.

---

## REQUIRED DOCUMENTATION FILES — what goes in each

- **PROJECT_SPEC.md** — the frozen research question, target variable definition, scope
  in/out, team, guide, timeline mapped to the official 16-week synopsis plan.
- **RESEARCH_PROTOCOL.md** — observation window, prediction horizon, cohort definition,
  train/val/test year ranges, exact Low/Medium/High rule, model list, metrics list.
- **DATA_DICTIONARY.md** — every field: name, source, meaning, datatype, feature/target/graph
  role, leakage risk (NONE/LOW/MEDIUM/HIGH), used-in-MVP (yes/no).
- **OPENALEX_SCHEMA.md** — exact OpenAlex API endpoints/fields used, pagination approach,
  rate-limit handling, polite-pool email usage, filter syntax used for AI/ML subset.
- **DATABASE.md** — Supabase schema DDL, table relationships diagram (as text/ASCII), local
  cache strategy, how to regenerate Parquet from source.
- **ARCHITECTURE.md** — system diagram (data → features/graph → models → API → dashboard),
  updated every phase.
- **COMPUTE_ENVIRONMENTS.md** — the laptop/lab split table above, plus concrete setup steps
  for each environment.
- **PHASES.md** — living checklist of phases, status (not started/in progress/done/blocked),
  and what was approved at each gate.
- **DECISIONS_LOG.md** — numbered decisions (question / decision / reason / date), e.g. the
  prediction-horizon and percentile-threshold decisions already discussed.
- **LEAKAGE_AUDIT.md** — every feature/edge that was flagged as a leakage risk, how it was
  resolved, and any open concerns.
- **EXPERIMENTS.md** — every experiment run: config used, dataset version, result, date.
- **KNOWN_ISSUES.md** — anything broken, deferred, or fragile, with enough context to resume.
- **GLOSSARY.md** — plain-language definitions (heterogeneous graph, temporal leakage,
  cohort-normalized labels, etc.) so any team member or the guide can read it cold.

---

## TESTING REQUIREMENTS (applies to every phase below)

- Use `pytest`. Every pipeline module gets at least one test file.
- Minimum test categories per data phase:
  - **Schema tests**: output has expected columns/dtypes, no unexpected nulls in required
    fields.
  - **Leakage tests**: assert no feature column contains data timestamped after its row's
    observation cutoff (this is the most important test in the whole project — treat it as
    a CI blocker, not a nice-to-have).
  - **Split integrity tests**: assert train/val/test year ranges don't overlap, and no paper
    ID appears in two splits.
  - **Label sanity tests**: class distribution roughly matches the configured percentile
    thresholds; no cohort has zero papers.
- Before ending each phase: run `pytest -v`, paste the actual output into the phase summary
  in chat, and only then update `docs/PHASES.md` to "done."

---

## PHASE PLAN

### Phase 0 — Scaffolding & Docs Skeleton
Create the full directory tree, all `docs/*.md` files (populated with the frozen spec from
the synopsis — background, problem statement, objectives, methodology, tech stack, timeline,
references — don't leave them empty), `configs/*.yaml` with placeholder values, `.env.example`
for Supabase credentials, and `requirements-cpu.txt` / `requirements-gpu.txt`.
**STOP and show the tree + doc contents for approval before Phase 1.**

### Phase 1 — OpenAlex Acquisition Plan (no downloading yet)
Produce, inside `docs/OPENALEX_SCHEMA.md`:
- Exact fields to pull (id, title, publication_year, publication_date, abstract_inverted_index,
  concepts/topics, authorships, institutions, referenced_works, cited_by_count,
  counts_by_year), each tagged IDENTIFIER/FEATURE/GRAPH/TARGET-SOURCE/METADATA and a leakage
  risk rating.
- How `counts_by_year` (or equivalent) will be used to reconstruct year-by-year citation
  trajectories rather than relying on the current aggregate count.
- The AI/ML concept filter, year range, target sample size (5,000–20,000 to start), and
  pagination/rate-limit/caching strategy.
**STOP for approval before writing any acquisition code or hitting the API.**

### Phase 2 — Acquisition Implementation
Build `ml/acquisition/openalex.py`: resumable, cached, rate-limit-aware, writes raw JSONL to
`data/raw/openalex/`, never overwrites raw data, logs exact query/config used alongside the
output. Also build the Supabase upsert path in `ml/db/`.
Tests: acquisition script runs against a small sample (e.g. 50 papers) without hitting real
leakage-relevant fields incorrectly; schema of parsed records matches `DATA_DICTIONARY.md`.

### Phase 3 — Preprocessing & Temporal Snapshotting
Dedup, resolve authors/institutions via OpenAlex IDs, build `data/interim/` Parquet tables,
implement the temporal snapshot logic (a paper's feature set only reflects information up to
its cutoff). Update `DATA_DICTIONARY.md` and `LEAKAGE_AUDIT.md` with real findings.
Tests: leakage tests as described above must pass.

### Phase 4 — Labeling
Implement cohort-normalized (same field + same publication year) percentile-based
Low/Medium/High labeling with a configurable prediction horizon (default 5 years) and
configurable thresholds in `configs/labels.yaml`. Log the actual class distribution.
Tests: label sanity tests.

### Phase 5 — Temporal Splitting
Implement train/val/test split by publication year (not random), documented in
`RESEARCH_PROTOCOL.md`. Also implement a "naive random split" variant purely for the future
leakage-ablation experiment (Master Prompt 2) — clearly labeled as the deliberately-flawed
comparison condition, not the default.
Tests: split integrity tests.

### Phase 6 — EDA & Tabular Baselines
Generate real EDA figures to `figures/` (papers/year, citation distribution, topic
distribution, authors/paper, class distribution, missingness) and `reports/dataset_report.md`
with real numbers. Train Logistic Regression and XGBoost on the tabular feature store using
the Phase 5 splits; log Accuracy/Precision/Recall/Macro-F1/confusion matrix per class into
`docs/EXPERIMENTS.md`.
Tests: baseline training runs end-to-end on the current dataset size without error and
produces metrics files.

**End of Master Prompt 1.** After Phase 6 is approved, move to Master Prompt 2 for graph
construction, GNN training (college GPU lab), leakage ablation, explainability, API, and
dashboard.

---

## FIRST TASK

Do not do anything beyond Phase 0 yet. Create the directory structure and all `docs/*.md`
files populated with the real project context (synopsis content, decisions already made:
OpenAlex as primary source, 5-year horizon as initial candidate, cohort-percentile labeling,
Supabase as DB, laptop/lab compute split). Then stop and show me everything for review.
