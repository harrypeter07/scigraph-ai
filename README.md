# SciGraph AI: Temporal Citation Trajectory & Impact Prediction

> **RCOEM CSE Major Project (Sem VII, Session 2026-27)**  
> **Guide:** Dr. Rina Damdoo  
> **Team:** Hassan & Team  

SciGraph AI is a research-first platform designed to predict the 5-year impact trajectory of scientific publications in Artificial Intelligence and Machine Learning (AI/ML) using heterogeneous graph neural networks (GNNs), with a specific scientific contribution focused on **temporal leakage auditing**.

---

## 🎯 Key Project Goals

1. **Temporal Leakage Audit**: Empirical demonstration that standard non-temporal graph learning evaluations leak future citation topology, artificially inflating model performance.
2. **Heterogeneous Graph Representation**: Modeling papers, authors, institutions, and research topics as a rich multi-relational graph.
3. **Cohort-Normalized Impact Labels**: Classifying papers into Low, Medium, and High impact relative to their publication year and field cohort, avoiding raw count biases.
4. **Reproducible ML Pipeline**: Rigorous separation of CPU data processing/baselines (laptop environment) and GPU GNN training (college lab environment).

---

## 📁 Repository Structure

```
scigraph-ai/
├── README.md                  # Project overview & quickstart
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
├── requirements-cpu.txt      # Laptop CPU dependencies
├── requirements-gpu.txt      # College Lab GPU dependencies
├── pytest.ini                # Pytest configuration
│
├── docs/                     # Research specs & architecture docs
│   ├── PROJECT_SPEC.md       # Target definitions & project scope
│   ├── RESEARCH_PROTOCOL.md  # Experimental setup & evaluation protocol
│   ├── DATA_DICTIONARY.md    # Field-level data & leakage risk schema
│   ├── OPENALEX_SCHEMA.md    # OpenAlex API filtering & pipeline spec
│   ├── DATABASE.md           # Supabase DDL & local Parquet cache design
│   ├── ARCHITECTURE.md       # End-to-end system architecture
│   ├── COMPUTE_ENVIRONMENTS.md # Laptop vs. Lab GPU setup guide
│   ├── PHASES.md             # Phase tracking & gate status
│   ├── DECISIONS_LOG.md      # Architectural decision record (ADR)
│   ├── LEAKAGE_AUDIT.md      # Catalog of leakage risks & mitigations
│   ├── EXPERIMENTS.md        # Benchmark results & metrics log
│   ├── KNOWN_ISSUES.md       # Technical debt & issue tracking
│   └── GLOSSARY.md           # Domain terminology & definitions
│
├── configs/                  # Yaml research configurations
│   ├── dataset.yaml          # OpenAlex acquisition & path settings
│   ├── labels.yaml           # Citation horizon & percentile cutoffs
│   ├── splits.yaml           # Publication year split boundaries
│   └── baselines.yaml        # Logistic Regression & XGBoost params
│
├── data/                     # Raw, interim, and processed datasets
│   ├── raw/openalex/         # JSONL OpenAlex API archives
│   ├── interim/              # Deduplicated & cleaned Parquet tables
│   └── processed/            # Labeled, feature-engineered datasets
│
├── ml/                       # Machine Learning codebase
│   ├── db/                   # Supabase & Parquet access layer
│   ├── acquisition/          # OpenAlex API pull client
│   ├── preprocessing/        # Data cleaning & normalization
│   ├── labels/               # Cohort-normalized impact labelers
│   ├── temporal/             # Snapshotting & temporal splitters
│   ├── features/             # Tabular & text embedding extractors
│   ├── baselines/            # Tabular baselines (LogReg, XGBoost)
│   └── eval/                 # Metrics & evaluation utilities
│
├── tests/                    # Unit & integration test suite
│   ├── test_acquisition.py
│   ├── test_preprocessing.py
│   ├── test_labels.py
│   ├── test_temporal_splits.py
│   ├── test_leakage_audit.py
│   └── test_baselines.py
│
├── notebooks/                # Exploratory analysis
│   └── eda.ipynb
│
├── reports/                  # Generated PDF/MD analysis reports
└── figures/                  # Publication-ready plots & charts
```

---

## ⚡ Quickstart

### 1. Environment Setup (Laptop / CPU)

```bash
# Clone the repository
git clone https://github.com/your-org/scigraph-ai.git
cd scigraph-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU dependencies
pip install -r requirements-cpu.txt

# Setup Environment Variables
cp .env.example .env
```

### 2. Running Unit Tests

```bash
pytest -v
```

---

## 🔬 Research & Compute Methodology

- **CPU Laptop Phase (Prompt 1)**: OpenAlex acquisition, data cleaning, temporal snapshotting, cohort labeling, temporal splitting, tabular baselines (LogReg, XGBoost), and graph construction tests.
- **GPU Lab PC Phase (Prompt 2)**: Heterogeneous GNN (GraphSAGE / GAT) training, temporal leakage ablation experiments, model explainability (GNNExplainer), FastAPI service, and Next.js research dashboard.
