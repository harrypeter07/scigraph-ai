# SciGraph AI — Implementation Pipeline & Work Done (5-Slide Master Deck)

**Target Scope**: Implementation Journey, Pipeline Flowcharts, Training Protocol & Model Rankings  
**Session**: B.Tech CSE Major Project (Sem VII 2026–2027)  

---

## 🗺️ Master Pipeline Flowchart

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SCIGRAPH AI END-TO-END IMPLEMENTATION PIPELINE                         │
└──────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 [ 1. DATASET ACQUISITION ] ────────► Fetch Works, Authors, Institutions, Citations from OpenAlex API
                                   │   (Stored in data/interim/*.parquet)
                                   ▼
 [ 2. TEMPORAL FREEZE PROTOCOL ] ───► Strict Observation Freeze at T_cutoff = Y_pub (ml/temporal/snapshotter.py)
                                   │   (Hides future citations post-publication to prevent leakage)
                                   ▼
 [ 3. DYNAMIC COHORT LABELING ] ────► Calculate 5-year growth ΔCitations_5 & assign P50/P90 Percentiles
                                   │   (Class 0: Low <50%, Class 1: Med 50-90%, Class 2: High ≥90%)
                                   ▼
 [ 4. HETEROGENEOUS GRAPH BUILD ] ──► Construct HeteroData: Paper, Author, Institution, Topic Nodes
                                   │   (4 Relational Edge Types in ml/graph/build_graph.py)
                                   ▼
 [ 5. MULTI-MODEL TRAINING ] ───────► Train Baselines (LogReg, GBDT) & PyTorch GNN (HeteroGraphSAGE, HeteroGAT)
                                   │   (Temporal Train Y≤2018, Val Y=2019, Test Y≥2020)
                                   ▼
 [ 6. VERIFICATION & DEPLOYMENT ] ──► Retrospective Time-Travel Simulator (ml/verification/simulator.py)
                                       + FastAPI Server (api/main.py) + Cytoscape.js Web UI
```

---

## 📌 SLIDE 1: The Implementation Journey (Where We Started to Current State)

### 📋 Slide Title: **The Implementation Lifecycle: From Raw Data to Graph Neural Networks**

### 🧩 3 Major Implementation Milestones:

1. **Phase A: Data Engineering & Zero-Leakage Architecture (Where We Started)**
   * Built automated ingestion scripts querying the **OpenAlex Open-Access Bibliographic API**.
   * Implemented high-throughput columnar storage using **Apache Parquet** (`data/interim/`).
   * Engineered the **Temporal Snapshotter**: enforced a strict observation boundary at publication year ($T_{\text{cutoff}} = Y_{\text{pub}}$) to guarantee zero future data contamination.

2. **Phase B: Dynamic Normalization & PyTorch Graph Construction**
   * Solved disciplinary citation inflation by building **Dynamic Cohort-Percentile Labeling** ($P_{50}$ and $P_{90}$).
   * Built multi-relational academic networks using **PyTorch Geometric `HeteroData`** connecting 4 distinct entity types ($\text{Papers}, \text{Authors}, \text{Institutions}, \text{Topics}$).

3. **Phase C: Model Training, Retrospective Engine & Web Deployment (Current State)**
   * Trained traditional ML baselines alongside deep **HeteroGraphSAGE** and **HeteroGAT** models.
   * Built an interactive **Retrospective Verification Engine** benchmarking forecasts against real history.
   * Deployed a production **FastAPI + Uvicorn server** with a custom **Cytoscape.js Web UI**.

---

## 📌 SLIDE 2: Dataset Acquisition & Schema Pipeline

### 📋 Slide Title: **Dataset Ingestion: Multi-Relational OpenAlex Entities**

### 📊 Raw Ingestion Schema:
* **Works Entity** (`papers.parquet`): OpenAlex ID, DOI, Title, Publication Year, Abstract inverted index, Primary Topic ID, Referenced Works list, Historical citation counts by year.
* **Authors Entity** (`authors.parquet`): Author ID, Display Name, Historical $h$-index, Total works count, Primary institution affiliation ID.
* **Institutions Entity** (`institutions.parquet`): Institution ID, University/Lab name, Country code, Global research citation count.
* **Citations & Authorships** (`paper_citations.parquet`, `authorships.parquet`): Relational adjacency edges mapping citations and author-paper relationships.

### 🛡️ Key Implementation Rule:
> Raw lifetime citations (`raw_cited_by_count`) are **strictly excluded** from all feature extraction and model inputs. Citations are strictly sliced at publication year.

---

## 📌 SLIDE 3: Preprocessing, Temporal Freeze & Dynamic Cohort Labeling

### 📋 Slide Title: **Feature Engineering & Dynamic Percentile Formulation**

### 🧮 1. The 5-Dimensional Normalized Input Vector ($\mathbf{x} \in \mathbb{R}^5$):
* $x_0 = \frac{Y_{\text{pub}} - 2012}{10}$ $\to$ Normalizes publication era into $[0.0, 1.0]$.
* $x_1 = \frac{\text{Citations}_{\le T_{\text{cutoff}}}}{100}$ $\to$ Scaled early adoption velocity at release date.
* $x_2 = \frac{\text{Title Word Count}}{20}$ $\to$ Lexical title complexity ratio.
* $x_3 = \frac{\text{Co-Author Count}}{10}$ $\to$ Collaborative team size ratio.
* $x_4 = \ln(1 + \text{Citations}_{\le T_{\text{cutoff}}})$ $\to$ Compresses heavy-tailed power-law citation outliers.

### 🏷️ 2. Dynamic Cohort Percentile Labeling (`ml/labels/labeler.py`):
$$\Delta \text{Citations}_5(p) = \text{Citations}(p, Y_{\text{pub}} + 5) - \text{Citations}(p, Y_{\text{pub}})$$
* **Class 0 (Low Impact)**: $\Delta \text{Citations}_5 < P_{50}(Y_{\text{pub}})$ (Bottom 50% of papers published in that year).
* **Class 1 (Medium Impact)**: $P_{50} \le \Delta \text{Citations}_5 < P_{90}$ (Mainstream solid contribution).
* **Class 2 (High Impact)**: $\Delta \text{Citations}_5 \ge P_{90}(Y_{\text{pub}})$ (**Top 10% Landmark Breakthrough Papers**).

---

## 📌 SLIDE 4: Model Architecture & Training Protocol

### 📋 Slide Title: **Model Architectures: Traditional Baselines vs. PyTorch GNNs**

### 🔬 Model Comparison & Training Setup:

| Model Architecture | Framework | Input Features | Aggregation Mechanism | Optimization & Loss |
|---|---|---|---|---|
| **Majority-Class Baseline** | Heuristic Anchor | Training Mode | None (Global Prior) | None (Deterministic) |
| **Logistic Regression** | Scikit-Learn | Flat Vector $\mathbf{x} \in \mathbb{R}^5$ | None (0-Hop Isolated Node) | L2 Regularization, L-BFGS |
| **Gradient Boosting (GBDT)** | Scikit-Learn | Flat Vector $\mathbf{x} \in \mathbb{R}^5$ | None (Non-Linear Splits) | Multi-Class Log-Loss, 100 Trees |
| **HeteroGraphSAGE** | PyTorch / PyG | Vector $\mathbf{x}$ + Adjacency | Inductive Mean / Pool Aggregation | Adam ($\text{lr}=0.01$), Cross-Entropy |
| **HeteroGAT** | PyTorch / PyG | Vector $\mathbf{x}$ + Adjacency | Multi-Head Self-Attention ($\alpha_{ij}$) | Adam ($\text{lr}=0.01$), Cross-Entropy |

### 🏋️ Strict Time-Series Splitting Protocol:
* **Training Split** (`train_temporal.parquet`): Papers published $Y \le 2018$ (40 papers).
* **Validation Split** (`val_temporal.parquet`): Papers published in $Y = 2019$.
* **Held-Out Test Split** (`test_temporal.parquet`): Future papers published $Y \ge 2020$ (5 papers).

---

## 📌 SLIDE 5: Model Rankings, Benchmark Results & Retrospective Verification

### 📋 Slide Title: **Model Ranking Results & Retrospective Verification**

### 🏆 Model Performance Ranking (Strict Temporal Test Split: $Y \ge 2020$):

| Rank | Model Architecture | Accuracy Fraction | Accuracy % | Macro-F1 | Empirical Verdict |
|:---:|---|:---:|:---:|:---:|---|
| **🥇 1** | **Tier 4 Full Hetero GNN (with Topics)** | **$4/5$** | **$80.0\%$** | **$0.6667$** | **Beats Baseline by +20.0%** |
| **🥈 2** | **HeteroGraphSAGE (Basic Topology)** | **$3/5$** | **$60.0\%$** | **$0.4286$** | Tied with Baseline Anchor |
| **🥈 2** | **Gradient Boosting (GBDT)** | **$3/5$** | **$60.0\%$** | **$0.4286$** | Tied with Baseline Anchor |
| **🥈 2** | **Logistic Regression** | **$3/5$** | **$60.0\%$** | **$0.4286$** | Tied with Baseline Anchor |
| **⚓ 5** | **Majority-Class Baseline (Anchor)** | **$3/5$** | **$60.0\%$** | **$0.3750$** | Trivial Starting Anchor |

---

### 🔍 Retrospective Verification Case Studies:
* **PRISMA 2020 Statement (2021)**: Cutoff Citations: $2,956$ $\to$ Forecasted: *Low Impact ($78.0\%$)* $\to$ Actual: *Low Impact ($25^{\text{th}}$ %ile)* $\to$ **CORRECT ✅**
* **ResNet (2016)**: Cutoff Citations: $918$ $\to$ Forecasted: *Low/Med Impact ($56.5\%$)* $\to$ Actual: *Medium Impact ($75^{\text{th}}$ %ile)* $\to$ **Documented Cold-Start Miss (Zero Leakage Proof)**.

---

## 🎙️ 1-Minute Master Seminar Defense Summary:

> *"Respected Teachers,*
> 
> *Our implementation journey began with ingesting multi-relational academic data from OpenAlex. We established an audited observation freeze at publication year ($T_{\text{cutoff}}$) to eliminate temporal data leakage, and formulated dynamic cohort percentiles ($P_{50}$ and $P_{90}$) to eliminate citation inflation bias.*
> 
> *We extracted a 5-dimensional normalized feature vector and built a heterogeneous graph in PyTorch Geometric across Papers, Authors, Institutions, and Topics.*
> 
> *When evaluating models on unseen 2020+ test papers, traditional tabular models tie with the majority baseline at 60.0%. However, incorporating full heterogeneous graph message passing boosts prediction accuracy to **80.0% (+20% improvement)**. Our live retrospective verification engine allows anyone to test historical papers interactively and verify forecasts against genuine ground truth."*
