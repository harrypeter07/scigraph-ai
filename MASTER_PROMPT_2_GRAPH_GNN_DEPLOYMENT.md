# MASTER PROMPT 2 of 2 — SciGraph AI: Heterogeneous Graph, GNN Training, Ablation & Dashboard

> **How to use this**: Paste this only after Master Prompt 1's Phase 0–6 are complete, tested,
> and approved. This picks up the same repo and the same rules — it does not replace them.

---

## ROLE (continued)

Same role and same non-negotiable rules as Master Prompt 1: no future-information leakage, no
fabricated results, extend-don't-rewrite, MD-first documentation, test-gated phases,
config-over-hardcoding, stop-and-report checkpoints. Re-read `docs/RESEARCH_PROTOCOL.md`,
`docs/LEAKAGE_AUDIT.md`, and `docs/DECISIONS_LOG.md` before writing any code — do not
re-derive decisions that are already frozen there.

---

## PRECONDITIONS

Confirm before starting:
- `data/processed/` contains labeled, temporally-split tabular data with passing tests.
- Baseline (LogReg, XGBoost) results exist in `docs/EXPERIMENTS.md`.
- Supabase schema for papers/authors/institutions/topics/edges is live and populated.

If any of these are missing or stale, stop and say so instead of proceeding on assumptions.

---

## COMPUTE SPLIT — THIS PROMPT SPANS BOTH ENVIRONMENTS

| Phase | Environment | Why |
|---|---|---|
| 7 (graph construction) | Laptop | I/O and graph-building from Supabase/Parquet, not GPU-bound |
| 8 (GNN training), 9 (leakage ablation) | **College lab GPU** | PyTorch Geometric training needs real GPU memory/time |
| 10 (evaluation/significance), 11 (explainability) | Either — evaluation is cheap, explainability plots can run on laptop from saved model checkpoints | |
| 12 (API), 13 (dashboard), 14 (integration tests) | Laptop | Web dev, not compute-heavy |

**Portability rule for Phase 8/9**: everything needed to train must run as
`python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda` from a fresh
`git clone`, with no laptop-specific paths, no interactive notebook dependency, and automatic
checkpoint saving every N epochs so a lab session that gets cut off can resume. Write
`docs/HOW_TO_RUN_ON_COLLEGE_LAB.md` with literal copy-paste steps: clone, create venv, install
`requirements-gpu.txt`, pull data (from Supabase or a synced data bundle), run training,
retrieve results/checkpoints back to the shared repo (e.g., via git-lfs, a small artifacts
folder, or manual copy — pick the simplest one that actually works for a shared college PC and
document it).

---

## ADDITIONAL DIRECTORY STRUCTURE

```
scigraph-ai/
├── ml/
│   ├── graph/                 # HeteroData construction from Supabase/Parquet
│   ├── gnn/
│   │   ├── models/            # graphsage.py, gat.py
│   │   ├── train.py
│   │   └── checkpoints/       # gitignored, synced manually from lab PC
│   ├── evaluation/            # metrics, significance testing, leakage ablation runner
│   └── explainability/        # attention-weight extraction, GNNExplainer wrapper
│
├── api/
│   ├── main.py                 # FastAPI
│   ├── routers/
│   └── schemas/
│
├── web/                         # Next.js + TypeScript + Tailwind
│   ├── app/
│   └── components/
│
├── tests/
│   ├── test_graph_construction.py
│   ├── test_gnn_smoke.py        # tiny synthetic graph, CPU, runs anywhere
│   ├── test_leakage_ablation.py
│   ├── test_evaluation_metrics.py
│   ├── test_api.py
│   └── test_e2e_dashboard.py
│
└── docs/  (additions)
    ├── HOW_TO_RUN_ON_COLLEGE_LAB.md
    ├── MODEL_CARD.md
    ├── API_SPEC.md
    └── RESULTS.md
```

---

## TESTING REQUIREMENTS (this prompt's phases)

- **Graph construction**: schema tests on `HeteroData` object (correct node types, correct
  edge types, no orphan nodes beyond expected rate, edge counts match Supabase edge tables).
- **GNN smoke test**: a tiny synthetic heterogeneous graph (10–20 nodes) that runs a full
  forward+backward pass on CPU in under a few seconds — this must pass on the laptop BEFORE
  any real training is attempted on the lab GPU, so broken code is caught before burning lab
  time.
- **Leakage ablation test**: assert the "naive split" and "time-consistent split" pipelines
  produce genuinely different graphs (different edge counts/statistics) — a common bug is
  building the ablation but accidentally feeding both conditions the same data.
- **Evaluation tests**: metric functions tested against known hand-computed values on a tiny
  fixture.
- **API tests**: FastAPI endpoints return correct schema on a sample request; error handling
  for unknown paper IDs.
- **E2E test**: one scripted path from "enter a paper ID" through the dashboard to a rendered
  prediction, run against a local dev server.

Every phase: run the relevant tests, paste real output in chat, then update `docs/PHASES.md`.

---

## PHASE PLAN

### Phase 7 — Heterogeneous Graph Construction (Laptop)
Build `ml/graph/build_graph.py`: constructs a PyTorch Geometric `HeteroData` object with node
types `Paper, Author, Institution, Topic` and edge types
`(Paper, cites, Paper)`, `(Author, writes, Paper)`, `(Author, affiliated_with, Institution)`,
`(Paper, has_topic, Topic)` — strictly respecting each paper's temporal cutoff (no edge to/from
a node that didn't exist yet at that cutoff). Also build the **naive** variant (random,
non-temporal edges/splits) purely for the Phase 9 ablation, clearly separated in code and
config so it can never accidentally become the default path.
Produce graph statistics (node/edge counts, degree distribution) into `figures/` and
`reports/graph_report.md`.
**STOP for approval before Phase 8 — this is the last laptop-only phase before lab GPU time is
needed, so get the graph right first.**

### Phase 8 — Baseline GNN Training (College Lab GPU)
Implement GraphSAGE and GAT with neighbour sampling in `ml/gnn/models/`, trainable via
`python -m ml.gnn.train --config <path> --device cuda`. Use identical temporal splits from
Phase 5/7. Checkpoint regularly. Log results into `docs/EXPERIMENTS.md` and `docs/RESULTS.md`
with the exact config and dataset version used — never a number without its provenance.
Run the CPU smoke test first, locally, before this ever goes to the lab machine.

### Phase 9 — Leakage Ablation (College Lab GPU)
Train the **same** GNN architecture on the naive-split graph vs. the time-consistent graph
(from Phase 7). Report the performance gap explicitly as the project's core empirical claim
("naive evaluation overestimates performance by X points"). Run across multiple seeds; report
mean ± std, and note in `docs/RESULTS.md` if/how significance testing was applied.

### Phase 10 — Full Evaluation & Feature Ablation
Compare LogReg/RF/XGBoost/GraphSAGE/GAT on identical splits with Macro-F1, per-class F1,
confusion matrices, and (if time allows) a feature-ablation table (metadata-only → +citations
→ +authors → +institutions → +topics). All numbers real, all provenance logged.

### Phase 11 — Explainability
Add attention-weight extraction and/or a GNNExplainer wrapper in `ml/explainability/`.
Produce sample explained predictions (top contributing neighbours/features) for a handful of
real papers, saved as artifacts the dashboard can render.

### Phase 12 — FastAPI Backend
Expose: paper lookup, prediction + explanation for a given paper ID, dataset/graph summary
stats. Document in `docs/API_SPEC.md`. Backend reads trained model checkpoints + Supabase, not
the raw pipeline.

### Phase 13 — Minimal Next.js Dashboard
Build only: a search/lookup box, a prediction result view (class + probabilities +
explanation), and a basic graph visualization (Cytoscape.js) for a selected paper's local
neighbourhood. No UI polish beyond functional clarity — this is a research demonstrator, not
a product.

### Phase 14 — Integration & Wrap-up
Run the full E2E test. Update `docs/MODEL_CARD.md` (what the model does/doesn't do, scope,
known limitations, dataset it was trained on) and `docs/RESULTS.md` as the final summary
document suitable for pulling directly into the seminar/report.

---

## FIRST TASK

Confirm the preconditions above are actually met by inspecting the current repo state (don't
assume). Then begin Phase 7 only, and stop for approval before touching Phase 8, since Phase 8
is the first phase that costs real lab-GPU time.
