# System Architecture: SciGraph AI

The diagram below depicts the data processing, feature transformation, training pipeline, and deployment architecture for SciGraph AI.

---

## 1. High-Level Pipeline Flow Architecture

```
                                  [ OpenAlex API ]
                                         │
                                         ▼
                             [ Data Acquisition Layer ]
                             (ml/acquisition/openalex.py)
                                         │
                                         ├───► Archive Raw JSONL (data/raw/openalex/)
                                         │
                                         ▼
                            [ Supabase Postgres DB ]
                          (Central Source of Truth Layer)
                                         │
                                         ▼
                         [ Preprocessing & Snapshotting ]
                         (ml/preprocessing/, ml/temporal/)
                                         │
                                         ▼
                           [ Parquet Feature Store ]
                         (data/interim/ & data/processed/)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     [ Tabular Baseline Branch ]                    [ Heterogeneous Graph Branch ]
     (Logistic Regression / XGBoost)                (HeteroData: Paper, Author, Topic)
                 │                                               │
                 │ (Laptop - CPU)                                │ (Lab PC - GPU)
                 ▼                                               ▼
      [ Baseline Evaluation ]                         [ GNN Models: GraphSAGE / GAT ]
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             [ Temporal Leakage Audit ]
                          (Time-Consistent vs Naive Split)
                                         │
                                         ▼
                                 [ FastAPI Service ]
                                   (api/main.py)
                                         │
                                         ▼
                             [ Next.js Research UI ]
                                (web/ App & Cytoscape)
```

---

## 2. Component Design Summary

- **Acquisition Layer**: Resumable API ingestion client with cursor pagination and polite pool rate limiting.
- **Data Persistence Layer**: Dual-tiered storage combining Supabase Postgres (remote database) with local Parquet files (offline batch ML execution).
- **Temporal Engine**: Ensures that every paper node's features and graph edges satisfy $t_{\text{edge}} \le T_{\text{cutoff}}$.
- **Labeling Module**: Generates 3-class target labels using subfield/year cohort percentile normalization.
- **Model Engine**:
  - *Laptop Environment*: Fits tabular benchmarks (Logistic Regression, XGBoost) and executes GNN CPU smoke tests.
  - *College Lab GPU Environment*: Fits full PyTorch Geometric Heterogeneous Graph Neural Networks and executes temporal leakage ablation experiments.
- **Serving Layer**: FastAPI service surfacing predictions, GNNExplainer attribution scores, and neighborhood subgraphs to the Next.js visual demonstrator.
