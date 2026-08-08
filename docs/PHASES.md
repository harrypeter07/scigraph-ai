# Project Phases & Status Tracker

> **Execution Protocol**: Each phase must achieve 100% test pass rates and explicit gate approval before advancing.

---

## Master Prompt 1: Foundation, Data & Tabular Baselines

| Phase | Description | Environment | Status | Gate Approved |
|---|---|---|---|---|
| **Phase 0** | Scaffolding & Docs Skeleton | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 1** | OpenAlex Acquisition Plan & Schema Spec | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 2** | OpenAlex Acquisition Implementation & Supabase Setup | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 3** | Preprocessing & Temporal Snapshotting | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 4** | Cohort-Normalized Labeling Engine | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 5** | Temporal & Naive Random Splitting Engine | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 6** | Dataset EDA & Tabular Baselines (LogReg / XGBoost) | Laptop (CPU) | **COMPLETED** | Gate Approved |

---

## Master Prompt 2: Graph Construction, GNNs, Ablation & Deployment

| Phase | Description | Environment | Status | Gate Approved |
|---|---|---|---|---|
| **Phase 7** | Heterogeneous Graph (`HeteroData`) Construction | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 8** | Heterogeneous GNN Training (GraphSAGE / GAT) | **Colab GPU (Tesla T4)** | **COMPLETED** | Gate Approved |
| **Phase 9** | Temporal Leakage Ablation Study | **Colab GPU (Tesla T4)** | **COMPLETED** | Gate Approved |
| **Phase 10** | Full Feature & Model Evaluation Matrix | Laptop / Lab | **PREPARED** | Pending Phase 8/9 |
| **Phase 11** | Model Explainability (GNNExplainer / Attention) | Laptop (CPU) | **PREPARED** | Pending Phase 8/9 |
| **Phase 12** | FastAPI REST Backend Service | Laptop (CPU) | **COMPLETED** | Gate Approved |
| **Phase 13** | Minimal Next.js Research UI Dashboard | Laptop (CPU) | **PREPARED** | Gate Approved |
| **Phase 14** | Integration Testing, Model Card & Final Wrap-up | Laptop (CPU) | **COMPLETED** | Gate Approved |
