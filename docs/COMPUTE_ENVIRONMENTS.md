# Compute Environments & Portability Specification

> **Design Principle**: Strict separation of CPU workflow tasks on local student laptops and GPU GNN training tasks on College Lab hardware.

---

## 1. Laptop vs. Lab GPU Responsibility Matrix

| Tasks / Phases | Hassan's Laptop (CPU Environment) | College Lab PC (GPU Environment) |
|---|---|---|
| **Hardware** | Intel/AMD CPU, 16GB RAM, No CUDA | Dedicated NVIDIA GPU (CUDA Enabled) |
| **Data Acquisition (Phase 1–2)** | Primary host | Not required |
| **Preprocessing & Snapshotting (Phase 3)** | Primary host | Not required |
| **Cohort Labeling (Phase 4)** | Primary host | Not required |
| **Temporal Splitting (Phase 5)** | Primary host | Not required |
| **EDA & Tabular Baselines (Phase 6)** | Primary host | Not required |
| **HeteroData Construction (Phase 7)** | Primary host (tiny validation test) | Full graph loading |
| **GNN Training (Phase 8)** | CPU Smoke Test only (<10 secs) | **Primary execution host** |
| **Temporal Leakage Ablation (Phase 9)** | Out of scope | **Primary execution host** |
| **Feature Ablation & Significance (Phase 10)** | Secondary analysis | Secondary analysis |
| **Explainability (Phase 11)** | Visualization rendering | Attention/GNNExplainer calculation |
| **FastAPI & Next.js UI (Phase 12–14)** | Primary host | Not required |

---

## 2. Portability Protocol for Lab GPU Training

To ensure seamless execution when transferring code to college lab computers:

1. **Zero Path Hardcoding**: All file paths use relative path references resolved against project root (`configs/dataset.yaml`).
2. **One-Command Training**: Model training is executed via standardized CLI invocation:
   ```bash
   python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda
   ```
3. **Environment Setup**:
   ```bash
   git clone <repo-url>
   cd scigraph-ai
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-gpu.txt
   ```
4. **Checkpoint Synchronization**: Model checkpoints are stored under `ml/gnn/checkpoints/` (git-ignored) and synced via documented artifact commands or manual copy.
