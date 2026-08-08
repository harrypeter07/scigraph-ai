# Research Protocol & Experimental Design

> **Document Status**: Frozen Specification (Phase 0)  
> **Target Scope**: SciGraph AI Evaluation Framework  

---

## 1. Temporal Snapshot Protocol

To eliminate temporal leakage, every paper $p_i$ published at year $Y_i$ is evaluated under a strict snapshot rule:
- **Observation Window**: Information available up to cutoff year $T_{\text{cutoff}} = Y_i$.
- **Prediction Horizon**: Cumulative citations accumulated between $Y_i + 1$ and $Y_i + 5$.
- **Feature Constraint**: Any node attribute, author affiliation, or citation edge $e = (u, v)$ with timestamp $t > T_{\text{cutoff}}$ is strictly excluded from feature computation and graph construction for paper $p_i$.

---

## 2. Cohort-Normalized Labeling Methodology

Raw citation counts exhibit extreme heavy-tailed distribution and field/year inflation. We apply **cohort-based percentile normalization**:

1. **Cohort Grouping**: Define cohort $C(Y, T)$ as all papers published in publication year $Y \in [2012, 2022]$ under primary concept/topic $T$.
2. **Trajectory Calculation**: For paper $p \in C(Y, T)$, compute 5-year citation trajectory:
   $$\Delta \text{Citations}_5(p) = \text{Citations}(p, Y+5) - \text{Citations}(p, Y)$$
3. **Percentile Ranking**: Rank $\Delta \text{Citations}_5(p)$ within cohort $C(Y, T)$.
4. **Class Assignment**:
   $$Y(p) = \begin{cases} 
   0 \text{ (Low)} & \text{if } \text{Percentile}(p) < 50.0 \\
   1 \text{ (Medium)} & \text{if } 50.0 \le \text{Percentile}(p) < 90.0 \\
   2 \text{ (High)} & \text{if } \text{Percentile}(p) \ge 90.0 
   \end{cases}$$

---

## 3. Data Splitting Strategies

### Strategy A: Time-Consistent Temporal Split (Default / Standard)
- **Training Set**: Papers published in $Y \in [2012, 2018]$ (Evaluated using features up to $Y_{\text{cutoff}} = Y$).
- **Validation Set**: Papers published in $Y = 2019$.
- **Test Set**: Papers published in $Y \in [2020, 2021]$ (Evaluated using 5-year horizon data up to 2025/2026).

### Strategy B: Naive Random Split (Ablation Condition Only)
- Random 70% Train / 15% Validation / 15% Test split across all publication years.
- **Purpose**: Serves as the negative control to quantify empirical performance inflation caused by temporal leakage.

---

## 4. Benchmark Models & Evaluation Metrics

### Models
1. **Tabular Baselines**:
   - Logistic Regression (L2 regularization, balanced class weights)
   - XGBoost Classifier (Multi-class `multi:softprob`)
2. **Graph Neural Networks (Lab GPU)**:
   - Heterogeneous GraphSAGE (`HeteroConv` with SAGEConv)
   - Heterogeneous Graph Attention Network (GATv2 / `HeteroConv`)

### Metrics
- **Primary Metric**: Macro-averaged F1 Score ($\text{Macro-F1}$)
- **Secondary Metrics**:
  - Per-class Precision, Recall, and F1 Score
  - Multi-class Confusion Matrix
  - Accuracy and Weighted F1 Score
