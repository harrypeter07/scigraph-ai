# How to Run GNN Training on Google Colab (Free GPU)

> **Alternative GPU Environment**: Google Colab provides free T4/V100 NVIDIA GPUs with pre-configured CUDA environments. You can run Phase 8 (GNN Training) and Phase 9 (Temporal Leakage Ablation) in Colab in minutes.

---

## ⚡ 3-Step Setup in Google Colab

### Step 1: Open Google Colab & Enable GPU
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. In the top menu, go to **Runtime** $\rightarrow$ **Change runtime type**.
4. Select **T4 GPU** (or any available GPU accelerator) and click **Save**.

---

### Step 2: Clone Repository & Install Dependencies
Run this in the first Colab cell:

```python
# Clone repository (or upload project files)
!git clone https://github.com/your-username/scigraph-ai.git
%cd scigraph-ai

# Install GPU requirements (PyTorch Geometric, PyG, etc.)
!pip install -r requirements-gpu.txt

# Verify CUDA GPU detection
import torch
print("CUDA GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
```

---

### Step 3: Run GNN Model Training & Leakage Ablation

#### Run Phase 8: Heterogeneous GraphSAGE Training on GPU
```python
!python -m ml.gnn.train --config configs/gnn_graphsage.yaml --device cuda
```

#### Run Phase 8: Heterogeneous GAT Training on GPU
```python
!python -m ml.gnn.train --config configs/gnn_gat.yaml --device cuda
```

---

## 📊 Retrieving Checkpoints & Results from Colab

After training finishes in Colab, save results to Google Drive or download directly:

```python
# Option A: Save checkpoints directly to Google Drive
from google.colab import drive
drive.mount('/content/drive')

!cp -r ml/gnn/checkpoints /content/drive/MyDrive/scigraph_checkpoints
!cp reports/* /content/drive/MyDrive/scigraph_reports/

# Option B: Download checkpoint files directly
from google.colab import files
files.download('ml/gnn/checkpoints/heterographsage_checkpoint.json')
```
