# Model Card: SciGraph AI Heterogeneous Citation GNN

> **Model Architecture**: Heterogeneous GraphSAGE / Heterogeneous GAT (`HeteroConv`)  
> **Task**: 3-Class Cohort-Normalized 5-Year Citation Impact Prediction  

---

## 1. Model Details

- **Model Developer**: Hassan & SciGraph AI Research Team (RCOEM, CSE)
- **Model Type**: Heterogeneous Graph Neural Network
- **Node Types**: Paper, Author, Institution, Topic
- **Edge Types**: `(Author, writes, Paper)`, `(Paper, cites, Paper)`, `(Author, affiliated_with, Institution)`, `(Paper, has_topic, Topic)`
- **Input Features**: Tabular paper statistics at cutoff + graph neighborhood structural topologies.
- **Target Classes**: 0 (Low Impact $<50\%$), 1 (Medium Impact $50-90\%$), 2 (High Impact $\ge 90\%$).

---

## 2. Intended Use & Scope

- **Intended Use**: Academic research evaluation, citation trajectory prediction, and temporal leakage auditing.
- **Out of Scope**: Commercial evaluation of patents, non-English scientific publications, or unreviewed preprint ranking without field calibration.

---

## 3. Training & Evaluation Protocol

- **Time-Consistent Temporal Splitting**:
  - Training Set: $Y_{\text{pub}} \in [2012, 2018]$
  - Validation Set: $Y_{\text{pub}} = 2019$
  - Test Set: $Y_{\text{pub}} \in [2020, 2021]$
- **Temporal Snapshot Constraint**: Every node attribute and graph edge timestamp satisfies $t \le T_{\text{cutoff}} = Y_{\text{pub}}$.
