# Technical Glossary & Domain Terminology

> **Purpose**: Plain-language explanations of domain concepts for project guidance reviews, academic thesis write-ups, and team alignment.

---

## Terminology Definitions

- **Heterogeneous Graph**: A graph structure containing multiple types of nodes (e.g., Papers, Authors, Institutions, Topics) and multiple types of relations/edges connecting them (e.g., *authorship*, *affiliation*, *citation*, *topic classification*).
- **Temporal Leakage**: The invalid inclusion of future state data (features or graph edges timestamped after observation cutoff $T_{\text{cutoff}}$) when training predictive models, resulting in unrealistically inflated test metrics.
- **Cohort-Normalized Labeling**: Ranking a paper's citation impact relative to immediate peers published in the same publication year and subfield cohort, preventing bias towards older publications or high-citation subdisciplines.
- **Prediction Horizon**: The fixed time frame (5 years post-publication) over which future citation accumulation is measured to assign target impact labels.
- **pgvector**: A PostgreSQL extension providing vector similarity searching capabilities over high-dimensional vector embeddings directly within relational Supabase tables.
- **HeteroConv / PyTorch Geometric**: PyTorch Geometric modules tailored for message passing over heterogeneous graphs where different node and edge types maintain distinct transformation matrices.
- **GNNExplainer**: Model explainability technique that identifies the minimal subfield node features and structural subgraph edges most influential for a GNN prediction.
