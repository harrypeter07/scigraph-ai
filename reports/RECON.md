# WP0 Reconnaissance Report: SciGraph AI 2.0 Ground Truth Audit

**Audit Date:** 2026-10-16  
**Repository:** `scigraph-ai`  
**Execution Environment:** Windows PowerShell / Python 3.12 / PyTorch Geometric / Pandas / OpenAlex  
**Audit Purpose:** WP0 Read-only ground-truth reconnaissance establishing empirical facts and leakages before code remediation.

---

## 1. Real HeteroData Architecture & Edge Connectivity

### Command Executed
```bash
python scripts/recon_wp0.py
```

### Raw Output Excerpt
```text
--- 1. Node Counts ---
Node 'paper': 16,500 nodes, shape=torch.Size([16500, 45])
Node 'author': 104,906 nodes, shape=torch.Size([104906, 8])
Node 'institution': 7,151 nodes, shape=torch.Size([7151, 8])
Node 'venue': 2,441 nodes, shape=torch.Size([2441, 8])
Node 'topic': 1,196 nodes, shape=torch.Size([1196, 8])

--- 1b. Edge Counts ---
Edge ('paper', 'cites', 'paper'): 87,709 edges
Edge ('paper', 'cited_by', 'paper'): 87,709 edges
Edge ('author', 'authored', 'paper'): 167,202 edges
Edge ('paper', 'authored_by', 'author'): 167,202 edges
Edge ('author', 'affiliated_with', 'institution'): 150,370 edges
Edge ('institution', 'affiliates', 'author'): 150,370 edges
Edge ('paper', 'published_in', 'venue'): 16,110 edges
Edge ('venue', 'publishes', 'paper'): 16,110 edges
Edge ('paper', 'has_topic', 'topic'): 16,500 edges
Edge ('topic', 'topic_of', 'paper'): 16,500 edges

Total rows in citation_edges.parquet: 1,760,775
Paper->paper edges where BOTH endpoints are in works.parquet: 87,709
Avg degree for 'author' (104,906 nodes): 6.05
Avg degree for 'institution' (7,151 nodes): 42.06
Avg degree for 'venue' (2,441 nodes): 13.20
Avg degree for 'topic' (1,196 nodes): 27.59
```

### Findings
- **Node Counts:** `paper`: 16,500 | `author`: 104,906 | `institution`: 7,151 | `venue`: 2,441 | `topic`: 1,196. Total entities: 132,194.
- **Paper-to-Paper Internal Subgraph:** While `citation_edges.parquet` contains **1,760,775** raw citation links (mostly outgoing references pointing outside the corpus), only **87,709** directed citation edges have **both** citing and referenced works present in `works.parquet`.
- **Degrees:**
  - `author`: 6.05 edges/node
  - `institution`: 42.06 edges/node
  - `venue`: 13.20 edges/node
  - `topic`: 27.59 edges/node

---

## 2. Training vs. Serving Feature Parity

### Findings & Evidence
- **Training Graph Features (`ml/gnn/graph_builder.py` lines 107-128):**
  - `paper`: 45-dim vector (cutoff citations, pub offset, references, authors, OA status one-hot, APC paid, SJR, top tier flag, 32-dim title SVD).
  - `author`: 8-dim vector, where `col 0` is `log1p(auth_counts)` and cols 1–7 are 0.0.
  - `institution`: 8-dim vector, where `col 0` is `log1p(inst_counts)` and cols 1–7 are 0.0.
  - `venue`: 8-dim vector, where `col 0` is `is_in_doaj` and cols 1–7 are 0.0.
  - `topic`: 8-dim vector, all zeros.
- **Serving Features (`ml/inference_service.py` lines 240–258):**
  - `paper`: Reconstructed 45-dim feature vector.
  - `author`, `institution`, `venue`, `topic`: Fed as dummy tensors `torch.zeros((1, 8), dtype=torch.float32)`.
  - Edges: Fed as single self-loops `torch.tensor([[0], [0]], dtype=torch.long)`.
- **Verdict:** Severe train-serving feature disparity. Serving passes disconnected zero vectors and self-loops, completely discarding author histories, venue prestige, and real multi-hop relational neighborhoods that the GNN trained on.

---

## 3. Temporal Edge Filtering Audit

### Findings & Evidence
- In `ml/gnn/graph_builder.py`:
  - `edge_cites` and `edge_cited_by` are constructed statically across all 16,500 works regardless of publication year or cutoff year.
  - For a training paper published in 2012 with cutoff 2014, its `cited_by` edges include citations from papers published in 2015, 2016, 2018, etc.
- **Verdict:** Temporal leakage in transductive message passing. Future citing papers propagate embeddings backwards into past target papers. PyG message passing must be filtered temporally so that for every message incoming to target paper $p$, $\text{edge\_time} \le \text{cutoff\_year}(p)$.

---

## 4. Corpus Sampling Methodology

### Findings & Evidence (`scripts/ingest_openalex_corpus.py` lines 73–84, 140–165)
- OpenAlex query parameters:
  ```python
  params = {
      "filter": f"primary_topic.field.id:{field_id},publication_year:{pub_year},has_abstract:true",
      "per-page": per_page,
      "page": page,
      "select": "..."
  }
  ```
- **Sorting:** No `sort` parameter was specified. OpenAlex defaults to internal relevance/ID ordering.
- **Sampling:** Not random sampling and not sorted by citations; rather, the first 300 works per (field, publication_year) cohort returning from the OpenAlex default pagination with `has_abstract:true`.
- Total cohorts: 5 fields $\times$ 11 publication years (2010–2020) = 55 cohorts $\times$ 300 works = 16,500 works.

---

## 5. Text-Encoder Persistence & Fitting Scope

### Findings & Evidence
- In `ml/gnn/graph_builder.py` lines 59–62:
  ```python
  tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
  tfidf_matrix = tfidf.fit_transform(df_works["title"].fillna("Untitled"))
  svd = TruncatedSVD(n_components=32, random_state=42)
  title_embeddings = svd.fit_transform(tfidf_matrix)
  ```
  Fit on **ALL** 16,500 works (train + val + test), causing data snooping from val and test titles into the SVD subspace.
- In `ml/inference_service.py` lines 97–108:
  Fit on boot across the entire corpus titles dynamically instead of loading a static checkpoint.
- The fitted `tfidf` and `svd` objects were **not saved to disk**.
- **Verdict:** Violates WP1 requirements. TF-IDF + TruncatedSVD must be fit on TRAIN split only and saved to `ml/gnn/checkpoints/text_encoder.joblib`.

---

## 6. Year Range, Min Year, and Truncation in `counts_by_year`

### Findings & Evidence
```text
df_counts year min: 2012, max: 2026
Works publication year distribution:
  2010: 1,500 (9.09%)
  2011: 1,500 (9.09%)
  2012: 1,500 (9.09%)
  ...
  2020: 1,500 (9.09%)
Share of 2010/2011 works: 3,000 / 16,500 (18.18%)
```
- OpenAlex `counts_by_year` in our dataset only tracks citations back to **2012**.
- Papers published in 2010 have cutoff year 2012 ($2010 + 2$). Their citations in 2010 and 2011 are missing from `counts_by_year`.
- Papers published in 2011 have cutoff year 2013 ($2011 + 2$). Their citations in 2011 are missing from `counts_by_year`.
- Total affected works: 3,000 papers (18.18% of corpus).
- **Verdict:** Per WP2d, we should drop publication_year < 2012 and retrain with train = 2012–2016, or explicitly document this truncation. Dropping eliminates truncation bias.

---

## 7. Dual-Head Regression Target Definition

### Findings & Evidence
- In `ml/gnn/graph_builder.py` line 100:
  ```python
  data["paper"].y_reg = torch.tensor(np.log1p(df_works["future_citations_5y"].values), dtype=torch.float)
  ```
- In `scripts/ingest_openalex_corpus.py` lines 270–280:
  `future_citations_5y` was computed as citations where $\text{cutoff\_year} < \text{year} \le \text{pub\_year} + 5$.
- For $\text{cutoff\_year} = \text{pub\_year} + 2$, the interval is $(\text{pub\_year} + 2, \text{pub\_year} + 5]$, exactly years $\text{pub} + 3, \text{pub} + 4, \text{pub} + 5$.
- **Verdict:** The regression target is $\ln(1 + \text{citations\_window})$. Per Rule F9, this must be consistently labeled `citations_window` everywhere and never referred to as "5-year total citations".

---

## 8. Temperature Scaling & Calibration Split Audit

### Findings & Evidence
- In `ml/gnn/models/hetero_gnn.py` line 84:
  `self.temperature = nn.Parameter(torch.ones(1) * 1.5)` (hardcoded initialization to 1.5).
- In `ml/gnn/train.py` and `ml/baselines/benchmark_suite.py`:
  The cross-entropy loss trains against raw logits `criterion_cls(logits_train, y_train_gnn)`. The temperature parameter is never optimized or fit.
- In `ml/baselines/benchmark_suite.py` line 340:
  `ece_5 = compute_ece(probs_5, y_test_cls)` evaluates calibration on the **test set** with the uncalibrated 1.5 parameter.
- **Verdict:** Violates calibration protocol. Temperature scaling must be fit strictly on the **validation split** (minimizing NLL or ECE), and evaluated on the test set.

---

## 9. Test Suite Invariant Audit

### Findings & Evidence
- `tests/test_backend_api.py`, `tests/test_phase2_scientometrics.py`, `tests/test_phase3_model.py`:
  - Rely predominantly on checking HTTP status 200, key existence in dictionary payloads, and simple non-null / range checks (e.g. `assert data["total_papers"] == 16500`).
  - Missing tests:
    - No temporal cutoff invariant test (asserting that modifying post-cutoff citations has zero effect on features and predictions).
    - No feature parity test comparing raw OpenAlex JSON against parquet row features.
    - No test ensuring serving graph matches training graph structure.
- **Verdict:** Tests must be expanded with strict mathematical invariant assertions as mandated in WPs 1, 2, 4, 5.

---

## Conclusion & Action Plan
All 9 reconnaissance items have been empirically verified. We proceed directly to **WP1 — Single Feature Builder & Text-Encoder Persistence**.
