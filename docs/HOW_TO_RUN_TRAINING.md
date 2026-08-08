# SciGraph AI — Documented Training & Scale-Up Run Guide

This document specifies the official step-by-step procedures for training SciGraph AI models locally and scaling up to large-scale dataset ingestion on Google Colab CUDA GPU when the research team is ready.

---

## 1. Quick Local Execution (Proof-of-Concept 50-Paper Sample)

To run the local evaluation suite and serve the web dashboard on your laptop:

```bash
# 1. Run baseline models and GNN evaluation
python -m ml.evaluation.full_evaluator
python -m ml.evaluation.ablation_runner

# 2. Run unit test suite
pytest -v

# 3. Launch live FastAPI & Evidence Dashboard server
python -m uvicorn api.main:app --port 8000 --reload
```

---

## 2. Documented Path to Scale Up on Google Colab (1,000 to 5,000+ Papers)

Follow these exact steps when ready to scale up dataset ingestion from 50 to 1,000–5,000 papers for final seminar/thesis evaluation.

> [!IMPORTANT]
> **DO NOT MIX SAMPLE SIZES**: When executing a large-scale run, all resulting Parquet datasets, checkpoints, reports, and audit documents must be regenerated fresh from the new dataset. Do not combine 50-paper numbers with scale-up numbers in the same table.

### Step 2.1: Increase Target Dataset Scale in Configuration
Edit `configs/dataset.yaml` on your local repository:
```yaml
acquisition:
  sample_size: 1000  # Scale up from 50 to 1,000 (or 5,000) papers
```

### Step 2.2: Mount Repository & Install GPU Environment in Google Colab
Set Colab Runtime: **Runtime -> Change runtime type -> T4 GPU**.

In Colab Cell #1:
```python
# 1. Clone repository
!git clone https://github.com/harrypeter07/scigraph-ai.git
%cd scigraph-ai

# 2. Install GPU dependencies (PyG & XGBoost)
!pip install torch-geometric xgboost pyyaml pandas pyarrow

# 3. Verify CUDA GPU
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
```

### Step 2.3: Execute Full Pipeline End-to-End in Colab
In Colab Cell #2:
```bash
# 1. Execute OpenAlex acquisition for scaled paper sample
python -m ml.acquisition.openalex

# 2. Execute preprocessing, temporal snapshotting, cohort labeling, and temporal splitting
python -m ml.preprocessing.cleaner
python -m ml.temporal.snapshotter
python -m ml.labels.labeler
python -m ml.temporal.splitter

# 3. Construct Heterogeneous PyG Graph
python -m ml.graph.build_graph

# 4. Train HeteroGraphSAGE & HeteroGAT on CUDA GPU
python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda
python -m ml.gnn.train --config configs/gnn_gat.yaml --device cuda

# 5. Run full evaluator and temporal leakage ablation runner
python -m ml.evaluation.full_evaluator
python -m ml.evaluation.ablation_runner
```

### Step 2.4: Push Scaled Results Back to GitHub
In Colab Cell #3:
```bash
!git config --global user.email "harrypeter07@gmail.com"
!git config --global user.name "harrypeter07"
!git add -A
!git commit -m "feat(colab): scaled 1000-paper GNN training checkpoints, parquets, and reports"
!git push https://github.com/harrypeter07/scigraph-ai.git main
```

### Step 2.5: Regenerate Audit Reports & Evidence Dashboard Locally
After pulling the scaled commit (`git pull`):
```bash
# Regenerate audit report and test suite
pytest -v
python -m ml.evaluation.full_evaluator
```
All dashboard endpoints at `http://localhost:8000` will automatically load the new scaled metrics!
