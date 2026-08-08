# Project Specification: SciGraph AI

> **Institution**: Shri Ramdeobaba College of Engineering and Management (RCOEM), Nagpur  
> **Department**: Computer Science & Engineering  
> **Degree / Session**: B.Tech CSE, Semester VII, Session 2026–27  
> **Guide**: Dr. Rina Damdoo  
> **Team Members**: Hassan (Lead R&D / Engineering) & Team (4 members total)  

---

## 1. Frozen Research Questions

1. **Primary Research Question (RQ1)**: Can a heterogeneous graph neural network (GNN) incorporating structural paper-author-institution-topic topologies outperform tabular baselines (Logistic Regression, XGBoost) in predicting 5-year cohort-normalized citation trajectories for AI/ML publications?
2. **Methodological Research Question (RQ2 - Core Contribution)**: How significantly does **temporal leakage** (evaluating GNNs on static graphs containing future edges or using un-snapshotted features) inflate empirical performance metrics compared to time-consistent temporal evaluation protocols?

---

## 2. Target Variable & Problem Formulation

- **Target Variable**: 3-Class Cohort-Normalized Citation Impact ($Y \in \{0, 1, 2\}$ corresponding to **Low**, **Medium**, and **High** impact).
- **Cohort Normalization**: Computed relative to papers published in the *same year* ($Y_{pub}$) within the *same subfield/topic* ($C$).
- **Prediction Horizon**: 5 years post-publication ($T_{pub} + 5$).
- **Percentile Boundaries**:
  - **Low Impact (0)**: $< 50^{\text{th}}$ percentile of citation trajectory in cohort.
  - **Medium Impact (1)**: $\ge 50^{\text{th}}$ and $< 90^{\text{th}}$ percentile of citation trajectory in cohort.
  - **High Impact (2)**: $\ge 90^{\text{th}}$ percentile of citation trajectory in cohort.

---

## 3. Scope Boundaries

### In-Scope
- OpenAlex data acquisition targeting AI/ML literature (2012–2022).
- Strict temporal feature snapshotting (features at $T \le T_{cutoff}$).
- Shared Supabase Postgres + `pgvector` backend with local Parquet cache layer.
- Tabular baselines (Logistic Regression, XGBoost).
- Heterogeneous graph construction (`HeteroData` with Paper, Author, Institution, Topic nodes).
- GraphSAGE and GAT architectures trained on PyTorch Geometric.
- Empirical temporal leakage ablation study comparing Time-Consistent vs. Naive Random splits.
- Feature/Attribution explainability (GNNExplainer / Attention weights).
- Lightweight FastAPI REST API and Next.js research demonstrator dashboard.

### Out-of-Scope
- GPU-heavy GNN model training on laptop hardware (explicitly assigned to College GPU Lab).
- Real-time continuous scraping of live arXiv feeds.
- Complex multimodal full-text parsing beyond OpenAlex abstracts and title text embeddings.

---

## 4. 16-Week Project Timeline

| Week | Milestone / Phase | Main Deliverables |
|---|---|---|
| Week 1–2 | Phase 0 & 1 | Scaffolding, docs, OpenAlex schema & acquisition spec approval |
| Week 3–4 | Phase 2 & 3 | OpenAlex ingestion, Supabase setup, preprocessing, temporal snapshotting |
| Week 5–6 | Phase 4 & 5 | Cohort-normalized labeling engine, temporal train/val/test splitting |
| Week 7–8 | Phase 6 | EDA generation, tabular baselines (LogReg, XGBoost), laptop gate review |
| Week 9–10 | Phase 7 & 8 | HeteroData construction, CPU smoke tests, Lab GPU GraphSAGE/GAT setup |
| Week 11–12 | Phase 9 & 10 | Temporal leakage ablation experiments on Lab GPU, feature ablation |
| Week 13–14 | Phase 11 & 12 | Explainability integration, FastAPI endpoint creation |
| Week 15–16 | Phase 13 & 14 | Next.js dashboard, E2E validation, final project report & presentation |
