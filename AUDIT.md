# Comprehensive Scientometric & Architecture Audit (Phase 0)

> **Audit Date:** 2026-10-14  
> **Repository:** `scigraph-ai`  
> **Scope:** Full repository codebase, ML pipeline, FastAPI backend, Cytoscape UI, and documentation.  
> **Audit Status:** Non-destructive / Read-Only Inspection.

---

## 1. Technical Verdict on Model Architecture: Does "HeteroGraphSAGE" Use Graph Structure?

### **Verdict: NO. The model is a 5-parameter Multi-Layer Perceptron (MLP) operating on isolated vectors. It does NOT use graph structure, adjacency matrices, neighborhood message passing, edge types, or heterogeneous node types.**

### Detailed Technical Proof:

1. **Self-Loop Identity Fallback in Neighborhood Aggregation:**
   In [`ml/gnn/models/graphsage.py#L36-L41`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/ml/gnn/models/graphsage.py#L36-L41):
   ```python
   def aggregate_neighbors(self, h: torch.Tensor, adj_matrix: torch.Tensor = None) -> torch.Tensor:
       """Mean message aggregation: h_N(v) = (1 / |N(v)|) * sum_{u in N(v)} h_u"""
       if adj_matrix is None or adj_matrix.dim() < 2 or adj_matrix.shape[0] != h.shape[0]:
           return h  # Self-loop fallback when single paper is queried
   ```
2. **Missing Adjacency Call in Forward Pass:**
   In [`ml/gnn/models/graphsage.py#L43-L64`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/ml/gnn/models/graphsage.py#L43-L64):
   ```python
   def forward(self, x, edge_index_dict=None):
       ...
       # 1-Hop Message Passing + Residual
       neighbor_x = self.aggregate_neighbors(x)           # <-- Called with NO adj_matrix!
       h1_input = torch.cat([x, neighbor_x], dim=-1)     # <-- Concatenates x with x!
       h1 = F.relu(self.sage_conv1(h1_input) + self.res1(x))
       ...
       # 2-Hop Message Passing + Residual
       neighbor_h1 = self.aggregate_neighbors(h1)        # <-- Called with NO adj_matrix!
       h2_input = torch.cat([h1, neighbor_h1], dim=-1)   # <-- Concatenates h1 with h1!
   ```
   `edge_index_dict` is an unused parameter in the signature. The method concatenates the input tensor with itself, running a linear projection with residual connections.
3. **Inference Feature Vector is 5 Isolated Scalars:**
   In [`api/routers/paper.py#L632-L643`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/api/routers/paper.py#L632-L643):
   ```python
   feat_tensor = torch.tensor([[
       float(pub_year - 2012) / 10.0,
       float(hist_citations) / 100.0,
       float(title_length) / 20.0,
       float(num_authors) / 10.0,
       float(torch.log1p(torch.tensor(float(hist_citations))))
   ]], dtype=torch.float)
   logits = active_model(feat_tensor)
   ```
   Neither venue, open access status, author history, institution prestige, citation lineage edges, nor topic embeddings enter the PyTorch model.
4. **Training Script Feeds Random Gaussian Noise:**
   In [`ml/gnn/train.py#L35-L61`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/ml/gnn/train.py#L35-L61):
   `HeteroGraphBuilder().build_time_consistent_graph()` is constructed, but its return `graph_data` is discarded. The training loop trains directly on `x = torch.randn(50, in_channels)`.

---

## 2. Inventory of Antipatterns & Violations by File & Line

### A. Fabricated Scientometric Estimates (Numbers Invented from Formulas)

| File | Line(s) | Fabricated Formula / Implementation | Real Research Violation |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 302, 306, 310, 314 | `h_idx = int(min(145, max(12, int(np.sqrt(max(10, total_citations)) * 1.1))))` | Author $h$-index is calculated using the square root of the target paper's citations multiplied by arbitrary multipliers (`1.1`, `0.95`, `1.05`, `0.7`). It is not the author's real $h$-index. |
| `api/routers/paper.py` | 327 | `"estimated_works": max(5, int(h_idx * 2.8))` | Author publication count is fabricated as $2.8 \times$ the fabricated $h$-index. |
| `api/routers/paper.py` | 328 | `"total_author_cits": int(max(50, total_citations * 0.8))` | Lifetime author citations are computed as $80\%$ of the single target paper's citations. |
| `api/routers/paper.py` | 408 | `velocity_val = round(14.2 + (abs(hash(topic_name)) % 120) / 10.0, 1)` | Topic citation velocity is computed from Python's string `hash()`. Because Python hashes are salted per process (`PYTHONHASHSEED`), this changes randomly on every server reboot. |
| `api/routers/paper.py` | 481-482 | `knowledge_export_ratio = round(outflow_count / max(1, inflow_count), 1)` | Ratio divides total post-cutoff citations by references. This is not a standardized metric and leaks the label. |
| `api/routers/paper.py` | 652-657 | `w_cit = 0.30 + 0.30 * (hist_cits / (hist_cits + 50.0))` ... `w_topic = max(0.05, 1.0 - sum(w))` | "Top contributing features" are hand-coded heuristic ratios rather than model gradients, SHAP values, or permutation importances. |
| `api/routers/paper.py` | 809 | `num_authors = 3` | Hardcoded author count fallback in local dataset prediction. |

---

### B. Hardcoded Constants Presented as Live Data & Metrics

| File | Line(s) | Hardcoded Constant | Context / Impact |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 128-175 | `"$35.00+ / paper (Subscription Required)"`, `"$2,500+"`, `"+28.4%"`, `"+31.2%"`, `"+22.6%"`, `"+14.5%"` | Economic reader fees, APC estimates, and citation advantage boosts are hardcoded string literals across OA profiles. |
| `api/routers/paper.py` | 776-779 | `n_papers = 50`, `n_authors = 257`, `n_inst = 215`, `n_cit = 2170` | Default fallbacks for `/api/v1/papers/stats` return fake data if Parquet files are missing. |
| `api/routers/evidence.py` | 32 | `{1: 36, 0: 7, 2: 7}` | Hardcoded fallback class distribution counts. |
| `api/routers/evidence.py` | 143-185 | `"3/5"`, `"60.0%"`, `0.2500` macro-F1 | Entire benchmark comparison table is hardcoded in the router JSON return. |
| `api/routers/evidence.py` | 200-201 | `{"test_size": 5, "accuracy": "60.0%"}`, `{"test_size": 8, "accuracy": "50.0%"}` | Hardcoded ablation test numbers. |
| `api/routers/evidence.py` | 213, 220, 227, 234 | `4785`, `4533`, `358`, `371` bytes | Hardcoded file sizes for artifacts. |
| `web/index.html` | 1156-1159 | `<div class="stat-number" id="evPapers">50</div>` ... `2,170` | Numbers hardcoded directly into frontend HTML structure. |
| `web/index.html` | 1182-1205 | `3 / 5`, `60.0%`, `0.2500` | Model benchmark rows hardcoded directly into table HTML. |

---

### C. Fallback Defaults Masking Missing Data & Heuristic Keyword Matching

| File | Line(s) | Flawed Fallback / Matching Logic | Issue |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 51-61 | `return {"quartile": "Q2", "badge": "Q2 HIGH TIER", ...}` | If venue metadata is missing, it defaults to $Q_2$ rather than `null`. |
| `api/routers/paper.py` | 68-72 | `any(k in name for k in ["nature", "science", "cell", "acl", ...])` | Keyword substring matching: `"cell"` matches *Cellular and Molecular Biology*; `"acl"` matches *Oracle*; `"science"` matches any social-science journal. |
| `api/routers/paper.py` | 354 | `any(k in iname.lower() for k in ["microsoft", "google", "harvard", ...])` | Substring checking for hardcoded list of corporate/ivy-league names to assign "Tier-1 Global Research Center". |
| `api/routers/paper.py` | 837 | `is_q1 = (paper_metadata or {}).get("venue_quartile", {}).get("is_q1", True)` | Defaults missing venues to $Q_1$ Tier-1 Elite (`True`). |
| `api/routers/paper.py` | 838 | `is_oa = (paper_metadata or {}).get("access_economics", {}).get("is_oa", True)` | Defaults missing access economics to Open Access (`True`). |
| `api/routers/paper.py` | 798 | `df["id"].str.contains(clean_id, case=False, na=False)` | Substring matching on paper ID allows partial IDs (e.g. `W2194`) to accidentally match arbitrary papers. |

---

### D. Silent Exception Handling & Upstream Failure Masking

| File | Line(s) | Code Block | Consequence |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 23-31, 608-618 | `try: ... except Exception: pass` | If checkpoint fails to load, silently falls back to a randomly initialized model with no warning. |
| `api/routers/paper.py` | 190-202 | `except Exception: return {}` | Any OpenAlex network failure (timeout, HTTP 429 rate limit, HTTP 500 error) returns an empty dict `{}`. |
| `api/routers/paper.py` | 800-804 | `if live_data and live_data.get("title"): ... else: raise HTTPException(status_code=404)` | Converts upstream network drops or rate limits into a misleading `404 Not Found: Paper does not exist`. |
| `ml/verification/simulator.py` | 60-61 | `except Exception as e: logger.warning(...)` | Silently ignores checkpoint loading failure in verification engine. |

---

### E. Graph Topology Flaws & Linear Chains

| File | Line(s) | Topology Implementation | Reality |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 390-398 | `for ca_i in range(min(len(authorships) - 1, 4)): e_coauth(auth_i, auth_{i+1})` | Generates a single linear chain of co-authors (`A0-A1, A1-A2, A2-A3`), not an authentic symmetric clique or pairwise co-authorship collaboration. |
| `ml/graph/build_graph.py` | 86-96 | `HeteroData()` with only `author, writes, paper` and `paper, cites, paper` | Missing reverse edges, missing venue nodes, missing topic nodes, missing institution edges, and missing node features (`x`). |

---

### F. Environment & Configuration Violations

| File | Line(s) | Hardcoded Item | Rule Violated |
| :--- | :--- | :--- | :--- |
| `api/routers/paper.py` | 191 | `params={"mailto": "hassan@rcoem.edu"}` | **R7**: Hardcoded personal email in API client request. |
| `api/routers/paper.py` | 439 | `params={"mailto": "hassan@rcoem.edu"}` | **R7**: Hardcoded personal email in forward citation request. |

---

### G. Tests Asserting Hardcoded Mock Outputs Instead of System Invariants

| File | Line(s) | Test Assertion | Flaw |
| :--- | :--- | :--- | :--- |
| `tests/test_evidence_dashboard.py` | 17-23 | `assert data["total_papers"] == 50`, `== 257`, `== 215`, `== 2170` | Tests fail immediately if a real dataset of 10,000+ papers is ingested, because they assert hardcoded placeholder numbers. |
| `tests/test_evidence_dashboard.py` | 33-35 | `assert data["test_set_size"] == 5` | Asserts hardcoded 5-sample temporal test set. |
| `tests/test_scientometrics.py` | 53 | `assert "PREPRINT" in arxiv_res["quartile_badge"]` | Asserts exact hardcoded keyword match. |
| `tests/test_scientometrics.py` | 74, 82 | `assert unpaid_res["oa_citation_boost_pct"] == 28.4`, `== 31.2` | Asserts hardcoded float constants rather than empirically calculated data distributions. |

---

### H. Single Source of Truth & Typing Violations

1. **Duplicate Alias Keys in API Responses:**
   In `api/routers/paper.py#L575-L585`:
   `inflow_precursor_count` and `precursor_inflow_count`, `outflow_downstream_count` and `downstream_outflow_count`, `lineage_role` and `classification`, `downstream_sample` and `top_downstream_works`, `badge` and `quartile_badge`.
2. **Presentation Logic in Backend Payloads:**
   In `api/routers/paper.py#L137, L146, L173`:
   `"color": "var(--accent-green)"` and `"var(--accent-amber)"` returned inside API JSON responses.
3. **Pervasive Untyped Dictionaries:**
   `api/routers/paper.py` and `api/routers/evidence.py` return raw `dict` structures with `Dict[str, Any]` in schemas.

---

## 3. Discrepancies Between Documentation Claims & Code Reality

1. **Claim:** *"HeteroGraphSAGE 2-Hop message passing across heterogeneous graph."*  
   **Reality:** Graph structure is ignored; the model executes as an MLP on 5 scalar features.
2. **Claim:** *"GNN model incorporates venue prestige, access economics, and bidirectional lineage momentum into prediction."*  
   **Reality:** These features are calculated post-hoc in Python and never enter the PyTorch tensor.
3. **Claim:** *"Feature attribution generated by graph neural network explanation."*  
   **Reality:** Attributions are generated by static algebraic formulas (`w_cit = 0.30 + 0.30 * (cit / (cit + 50))`).
4. **Claim:** *"Empirical open access citation boost of +31.2%."*  
   **Reality:** Hardcoded constant embedded in an `oa_profiles` dictionary.
5. **Claim:** *"Time-consistent temporal evaluation proving leakage elimination."*  
   **Reality:** Evaluated on a 5-paper test set where all models tie with majority baseline prediction ($60.0\%$, Macro-F1: $0.2500$).

---

## 4. Phase 0 Audit Conclusion & Remediation Roadmap

The current application functions as a prototype demonstration. To fulfill the teacher's requirements and academic integrity standards:

1. **Phase 1 (Data):** Fetch an authentic corpus of 20,000–100,000 works via cursor pagination across 5 fields (2010–2020) with strict as-of-cutoff ($T_{\text{cutoff}} = \text{pub\_year} + 2$) feature extraction and cohort-normalized labels ($\ge 90\text{th}$ percentile for High Impact).
2. **Phase 2 (Features):** Replace keyword matching with official SCImago/CORE ranking datasets, replace static OA boosts with empirical dataset distributions, and calculate real author $h$-indices as of cutoff.
3. **Phase 3 (Model):** Build an authentic PyG `HeteroData` graph with message passing (`HeteroConv` / `to_hetero`), text embeddings, genuine baselines (XGBoost, Logistic Regression, MLP), and true feature attribution (Captum / Permutation Importance).
4. **Phase 4 & 5 (Backend & UI):** Strongly typed Pydantic models, rate-limited HTTPX client with caching, structured error codes, backtesting panel, and citation trajectory curves.
5. **Phase 6 & 7 (Tests & Docs):** Recorded fixture tests and factual model cards.

---
*End of Phase 0 Audit.*
