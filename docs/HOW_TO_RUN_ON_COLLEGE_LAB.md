# How to Run GNN Training on College Lab GPU PCs

> **Target Machine**: Shri Ramdeobaba College of Engineering and Management (RCOEM) CSE Laboratory PC (NVIDIA CUDA GPU).  
> **Purpose**: Execute Phase 8 (GraphSAGE / GAT Training) and Phase 9 (Temporal Leakage Ablation Study).

---

## 🚀 Step-by-Step Instructions

### 1. Clone the Repository
Open Terminal / Command Prompt on the Lab PC:
```bash
git clone https://github.com/your-org/scigraph-ai.git
cd scigraph-ai
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install GPU Dependencies
```bash
pip install -r requirements-gpu.txt
```

### 4. Verify GPU CUDA Availability
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 🏃 Executing Phase 8: GNN Model Training

Run Heterogeneous GraphSAGE on GPU:
```bash
python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda
```

Run Heterogeneous GAT on GPU:
```bash
python -m ml.gnn.train --config configs/gnn_gat.yaml --device cuda
```

---

## 🔬 Executing Phase 9: Temporal Leakage Ablation Study

Run comparative ablation training (Time-Consistent Split vs Naive Random Split):
```bash
python -m ml.evaluation.ablation_runner --device cuda
```

---

## 📦 Retrieving Artifacts & Checkpoints

Model checkpoints are automatically saved to `ml/gnn/checkpoints/`. Copy the generated metrics and checkpoints back to the shared repository:
```bash
git add docs/EXPERIMENTS.md docs/RESULTS.md reports/
git commit -m "feat(lab-gpu): log Phase 8 & 9 GNN training and leakage ablation results"
git push origin main
```
