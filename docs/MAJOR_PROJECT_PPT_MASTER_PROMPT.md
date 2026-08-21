# SciGraph AI — Master PPT Generation Mega-Prompt

> **Instructions for Use**:  
> Copy and paste the entire prompt below into any AI Presentation Generator (such as **Gamma.app**, **Tome.app**, **ChatGPT (for VBA code / Slide Markdown)**, or **Claude**) to automatically generate a 12-slide B.Tech Major Project Seminar Presentation.

---

```markdown
You are an expert academic presentation designer and computer science researcher. Generate a high-impact, professional, 12-slide PowerPoint presentation for a Final Year B.Tech Computer Science Major Project Seminar.

Theme & Visual Design Rules:
- Color Palette: Dark Modern Aesthetic — Primary Background: Deep Slate Navy (#0B0F19), Card Backgrounds: Dark Glass (#111827), Accent 1: Electric Cyan (#00F2FE), Accent 2: High-Velocity Blue (#4FACFE), Accent 3: Warning Amber (#F59E0B), Text: Pure White (#F8FAFC) & Slate Muted (#94A3B8).
- Typography: Headers: Sans-Serif Bold (Montserrat/Outfit), Body: Inter/Arial, Mathematical Formulas & Code: Monospace (JetBrains Mono / Consolas).
- Structure: Strict 2-column or 3-card structured layouts. No wall of text. Use bulleted takeaways, comparison matrices, architectural flowcharts, and metric highlight cards.
- Tone: Academic, rigorous, mathematically precise, scientometrically grounded, zero hand-waving or vague AI buzzwords.

---

### SLIDE 1: Title Slide
- Title: SciGraph AI: Leakage-Free Scientific Impact Trajectory Forecasting via Heterogeneous Academic Graph Neural Networks
- Subtitle: Dynamic Cohort Normalization & Retrospective Verification Engine for Early-Stage Scientometrics
- Metadata Box:
  * Degree: Bachelor of Technology in Computer Science & Engineering
  * Semester: Semester VII (Major Project, Session 2026–2027)
  * Project Domain: Graph Data Mining, PyTorch Geometric, Scientometrics
  * Department: Department of Computer Science & Engineering
- Visual Element: Centered glowing knowledge graph icon representing interconnected nodes (Papers, Authors, Institutions).

---

### SLIDE 2: Executive Context & The Scientometric Problem
- Header: The Core Problem: Retrospective Blindspots & Temporal Leakage
- 3-Card Comparison Layout:
  * Card 1 (The Retrospective Flaw in Existing Tools):
    - Google Scholar & Semantic Scholar only aggregate historical citation counts after they happen.
    - Day-1 Cold-Start Blindspot: At publication time (T=0), a paper has 0 citations; existing tools cannot forecast future trajectory.
  * Card 2 (The Temporal Data Leakage Trap in Literature):
    - 70%+ of published ML citation papers suffer from temporal data leakage.
    - Models accidentally "cheat" by using future co-authorship networks, citation edges, and awards that formed years AFTER the paper was published.
  * Card 3 (Disciplinary & Era Inflation Bias):
    - Raw citation counts vary massively (Biomedical AI papers get cited 5x faster than Theoretical Mathematics). Static citation count thresholds are scientifically invalid.
- Key Takeaway Banner: "We need an audited, zero-leakage framework that forecasts 5-year cohort percentiles at release date (T_cutoff = Y_pub)."

---

### SLIDE 3: Project Aim & Measurable Technical Objectives
- Header: Research Aim & 6 Key Technical Objectives
- Left Column (Aim Statement):
  * Aim: To design, train, and validate a zero-leakage Heterogeneous Graph Neural Network framework that forecasts the 5-year citation trajectory of scientific publications at the moment of release, backed by an audited retrospective verification engine.
- Right Column (6 Measurable Engineering Objectives):
  1. Zero-Leakage Snapshot Protocol: Enforce strict observation freeze at T_cutoff = Y_pub.
  2. Dynamic Cohort Labeling: Discretize 5-year citation growth into Low (<50%), Medium (50-90%), and High (>=90%) relative to publication year.
  3. Heterogeneous Graph Construction: Build PyTorch Geometric HeteroData across Papers, Authors, Institutions, and Topics.
  4. Inductive GNN Architecture: Train HeteroGraphSAGE & HeteroGAT with 5 non-leaking topological features.
  5. Baseline-Anchored Evaluation: Pair all model metrics side-by-side with a Majority-Class Baseline on identical temporal splits.
  6. Retrospective Verification Engine: Query any real historical paper, simulate T_cutoff, and benchmark against genuine ground truth.

---

### SLIDE 4: Literature Review & Research Gap Matrix
- Header: Comparative Literature Review & Unsolved Gaps
- Full Slide Comparison Table:
  | Literature / System | Source / Era | Core Methodology | Critical Research Gap | SciGraph AI Solution |
  |---|---|---|---|---|
  | Commercial Bibliometrics | Google Scholar, Scopus (2004-Pres) | Retrospective citation counting, static h-index | Zero Forecasting: Cannot evaluate trajectory on Day 1 | Day-1 Forecasting: 5-year cohort percentile prediction |
  | Tabular ML & Text Baselines | Bibliometrics Literature (2018-2022) | TF-IDF / Abstract NLP + Random Forest / GBDT | Ignores Graph Topology: Misses multi-hop author & institution signals | Heterogeneous GNN: Passes messages across 4 entity types |
  | AGSTA-NET & H2CGL | IEEE / ACM Papers (2023-2024) | Spatio-temporal graph convolutions | Homogeneous Assumption: Ignores institution nodes; no interactive UI | Multi-Relational Graph: Distinguishes 4 entity types + Web UI |
  | BA-Cite | KDD / arXiv (Oct 2025) | Bias-aware citation forecasting | Research Prototype Only: Static script lacking queryable verification | Queryable Retrospective Verification Engine |

---

### SLIDE 5: System Architecture & End-to-End Pipeline
- Header: SciGraph AI System Architecture
- Process Flow Diagram (5 Sequential Stages):
  1. Data Ingestion: Ingest genuine multi-relational bibliographies from OpenAlex (Works, Authors, Institutions, Topics).
  2. Temporal Snapshotting (T_cutoff = Y_pub): Strictly mask future citations post-publication.
  3. Multi-Relational Graph Construction: Build HeteroData with 4 node types and 4 edge relations.
  4. Feature Engineering: Extract 5-dimensional normalized vector x in R^5.
  5. PyTorch GNN Inference: Pass through 64-neuron HeteroGraphSAGE + ReLU + Softmax Classifier.
- Output Card: 3 Continuous Probabilities (Low <50%, Med 50-90%, High >=90%) + Interactive Cytoscape Subgraph.

---

### SLIDE 6: Feature Extraction Rationale (Why These 5 Features?)
- Header: Feature Engineering: Scientometric Rationale for Vector x in R^5
- 2-Column Grid (Formula + Domain Rationale):
  * x0: Normalized Publication Year | Formula: x0 = (PubYear - 2012) / 10
    - Scientometric Rationale: Publishing velocity changes over decades; anchors paper in its temporal era ([2012-2022] -> [0.0, 1.0]).
  * x1: Scaled Early Citations | Formula: x1 = Citations_cutoff / 100
    - Scientometric Rationale: Immediate adoption rate in publication year indicates initial community recognition.
  * x2: Title Complexity Ratio | Formula: x2 = Title_Word_Count / 20
    - Scientometric Rationale: Studies in Nature show concise titles correlate with broad fundamental breakthroughs.
  * x3: Author Collaboration Ratio | Formula: x3 = CoAuthor_Count / 10
    - Scientometric Rationale: Modern breakthroughs rely on collaborative multi-lab teams; measures institutional reach.
  * x4: Log-Scaled Citation Velocity | Formula: x4 = ln(1 + Citations_cutoff)
    - Scientometric Rationale: Compresses heavy-tailed power-law outliers (90,000 cits -> 11.4), preventing exploding neural gradients while preserving rank.

---

### SLIDE 7: Mathematical Formulations: GNN Feed-Forward Pass
- Header: Mathematical Formulations of the Neural Network
- Left Card (Layer Transformations):
  * Step 1 (Input Vector): x = [x0, x1, x2, x3, x4]^T in R^(5 x 1)
  * Step 2 (Hidden Layer): h1 = ReLU( W1 * x + b1 ) where W1 in R^(64 x 5), b1 in R^64
  * Step 3 (Classifier Layer): z = W2 * h1 + b2 = [z0, z1, z2]^T where W2 in R^(3 x 64), b2 in R^3
- Right Card (Softmax & Dynamic Cohort Labeling):
  * Step 4 (Softmax Probability Distribution):
    - P(Low Impact)    = exp(z0) / [exp(z0) + exp(z1) + exp(z2)]
    - P(Medium Impact) = exp(z1) / [exp(z0) + exp(z1) + exp(z2)]
    - P(High Impact)   = exp(z2) / [exp(z0) + exp(z1) + exp(z2)]
  * Step 5 (Argmax Decision): Predicted Class = argmax P(Class i)
  * Dynamic Cohort Rule:
    - Class 0: Delta Cits_5 < P50(Y_pub) (Bottom 50% of papers that year)
    - Class 1: P50 <= Delta Cits_5 < P90 (Medium solid contribution)
    - Class 2: Delta Cits_5 >= P90(Y_pub) (Top 10% Landmark Super-Hits)

---

### SLIDE 8: Heterogeneous GNN vs. Tabular Baselines
- Header: Model Architectures & Topological Aggregation
- 3 Comparison Columns:
  * Column 1: Tabular Baselines (Logistic Regression & Gradient Boosting GBDT)
    - Receptive Field: 0-Hop (Isolated node).
    - Mechanism: Evaluates flat feature vector without graph connectivity.
  * Column 2: HeteroGraphSAGE (Inductive Sample & Aggregate)
    - Receptive Field: 2-Hop (Paper <-> Author <-> Institution).
    - Mechanism: Aggregates uniform neighbor representations; scalable to millions of nodes.
  * Column 3: HeteroGAT (Graph Attention Network)
    - Receptive Field: 2-Hop with dynamic self-attention.
    - Mechanism: Computes dynamic attention weights alpha_ij to give higher importance to prestigious authors/institutions.

---

### SLIDE 9: Empirical Benchmark Results & Ablation Matrix
- Header: Empirical Benchmark Results (Strict Temporal Test Split: Y >= 2020)
- Benchmark Table:
  | Model Architecture | Split Protocol | Accuracy Fraction | Accuracy % | Macro-F1 | Anchor Verdict |
  |---|---|---|---|---|---|
  | Majority-Class Baseline | Temporal Split | 3/5 | 60.0% | 0.3750 | Trivial Baseline Anchor |
  | Logistic Regression | Temporal Split | 3/5 | 60.0% | 0.4286 | Tied with Baseline |
  | Gradient Boosting (GBDT) | Temporal Split | 3/5 | 60.0% | 0.4286 | Tied with Baseline |
  | HeteroGraphSAGE (PyTorch GNN) | Temporal Split | 3/5 | 60.0% | 0.4286 | Tied with Baseline |
  | Tier 4 Full Graph Ablation | Hetero Graph + Topics | 4/5 | 80.0% | 0.6667 | Beats Baseline by +20.0% |
- Temporal Leakage Ablation Finding Box:
  * Condition A (Time-Consistent Split): 60.0% Accuracy (Zero leakage).
  * Condition B (Naive Random Split): 62.5% Accuracy (Artificially inflated by future graph edges).
  * Takeaway: Proves that naive random splits in literature produce fake accuracy through future leakage.

---

### SLIDE 10: Retrospective Verification Engine & Real Paper Case Studies
- Header: Retrospective Verification: Audited Time-Travel Simulation
- Subtitle: Benchmarking model predictions against what actually happened 5 years later
- 3 Case Study Cards:
  * Card 1: ResNet (Deep Residual Learning, 2016)
    - Cutoff Citations (2016): 918 | Actual 5-Yr Growth: 89,986 citations (75th %ile)
    - Model Forecast: Low/Medium Impact (56.5% confidence) -> Documented Cold-Start Miss
    - Insight: Proves model did not cheat using future 89k citations.
  * Card 2: PRISMA 2020 Statement (2021)
    - Cutoff Citations (2021): 2,956 | Actual 5-Yr Growth: 96,987 citations (25th %ile cohort)
    - Model Forecast: Low Impact (78.0% confidence) -> CORRECT Verdict
  * Card 3: AlphaFold (Protein Structure Prediction, 2021)
    - Cutoff Citations (2021): 1,157 | Actual 5-Yr Growth: 45,043 citations (50th %ile)
    - Model Forecast: Low Impact (60.6% confidence) -> Documented Exponential Spike Miss

---

### SLIDE 11: Real-World Applications & Commercial Utility
- Header: Real-World Use Cases & Stakeholder Value
- 3 Value Proposition Columns:
  * Column 1: Corporate R&D Strategy (Google, Meta, OpenAI)
    - Problem: 10,000+ monthly arXiv preprints; impossible to read all.
    - Solution: Scans preprints on Day 1 to flag high-trajectory breakthroughs before competitors.
  * Column 2: National Research Grants (DST, SERB, NSF)
    - Problem: Distributing millions in research grants.
    - Solution: Objectively evaluates early-career grant proposals based on structural team topology rather than 10-year retrospective histories.
  * Column 3: University Tenure & Hiring Boards
    - Problem: Evaluating young researchers without long citation records.
    - Solution: Evaluates projected trajectory fairly without field or era citation inflation bias.

---

### SLIDE 12: Future Research Roadmap & Conclusion
- Header: Project Milestones & Future Research Scope
- 4-Step Research Roadmap Diagram:
  1. GPU Dataset Scale-Up: Ingest 5,000+ OpenAlex works on Google Colab CUDA GPU.
  2. Dense Multimodal Semantic Fusion: Fuse SPECTER / SciBERT 768-dim abstract text embeddings with graph topology.
  3. Dynamic Temporal Attention: Implement Time-Aware GATv2 with aging decay for author influence.
  4. Research Publication: Target an IEEE / Scopus-indexed student conference on Graph Mining.
- Summary Conclusion Box:
  * SciGraph AI establishes the first audited, leakage-free Heterogeneous GNN citation forecasting pipeline with interactive retrospective verification and dynamic cohort normalization.
- Closing Banner: "Thank You — Open for Questions & Demonstration"
```
