# Architecture & Design Decisions Log (ADR)

> **Format**: Sequential ADR entries detailing Question, Decision, Rationale, and Date.

---

### ADR-001: Data Source Selection
- **Date**: 2026-08-07
- **Question**: Which academic data provider should serve as the primary source for papers, citations, authors, and topics?
- **Decision**: Select **OpenAlex** as the primary data source.
- **Rationale**: OpenAlex provides an open, comprehensive REST API with complete historical citation trajectories (`counts_by_year`), topic hierarchies, author affiliations, and cursor pagination without paywalls or low quota limits.

---

### ADR-002: Target Horizon & Cohort Normalization
- **Date**: 2026-08-07
- **Question**: How should scientific impact be defined to avoid raw count and field bias?
- **Decision**: Define a **5-year post-publication prediction horizon** with 3-class **cohort-normalized percentile thresholds** (Low $<50\%$, Medium $50-90\%$, High $\ge 90\%$) within publication year and subfield.
- **Rationale**: Raw citation counts favor older papers and high-volume subfields. Cohort percentile normalization evaluates papers fairly against immediate peers published in the same subfield and year.

---

### ADR-003: Database & Local Caching Strategy
- **Date**: 2026-08-07
- **Question**: What database architecture balances multi-user access with fast offline machine learning training loops?
- **Decision**: Use **Supabase (Postgres + pgvector)** as the shared central queryable source of truth, and **Parquet files** (`data/interim/`, `data/processed/`) as a local fast cache for model training.
- **Rationale**: Supabase provides queryable relations and vector embeddings, while local Parquet files bypass network round-trip overhead during high-speed PyTorch / XGBoost training loops.

---

### ADR-004: Compute Environment Split
- **Date**: 2026-08-07
- **Question**: How should compute tasks be partitioned between student laptops and college GPU laboratory hardware?
- **Decision**: Restrict laptop CPU environment to acquisition, preprocessing, snapshotting, tabular baselines, API, UI, and GNN CPU smoke tests. Reserve GNN training and leakage ablation studies for the **College Lab GPU PCs**.
- **Rationale**: Prevents CPU throttling or memory crashes on laptops while ensuring full PyTorch Geometric GPU acceleration during heavy graph model training.
