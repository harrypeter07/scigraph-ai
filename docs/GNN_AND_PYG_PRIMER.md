# Graph Neural Networks (GNN) & PyTorch Geometric (PyG) Primer

**Audience**: Seminar Defense Team (Hassan & Team) & Evaluators  
**Scope**: Practical Guide & Hand-Traceable Reference for SciGraph AI  

---

## 1. What a Graph Neural Network (GNN) Actually Computes

In standard machine learning (like Logistic Regression or Random Forests), every data sample is an **isolated row** in a table. It cannot "see" who its neighbors are.

In scientific research, papers do not exist in isolation. A paper is written by **authors**, who belong to **institutions**, and it cites **prior papers**.

A **Graph Neural Network (GNN)** computes predictions through **Neighborhood Message Passing**:
1. **Node Features**: Every node starts with an initial numerical vector (e.g., publication year, early citations, title word count).
2. **Message Generation**: Every neighbor node creates a "message" vector.
3. **Neighborhood Aggregation**: The target node gathers (sums, averages, or pools) all incoming messages from its connected neighbors.
4. **State Update**: The target node combines its aggregated neighbor messages with its own previous state through an activation function ($\text{ReLU}$).

```
  [Author Node A] ──(Message 1)──┐
                                  ├──► [AGGREGATE] ──► [UPDATE Target Paper Vector]
  [Institution Node B] ──(Message 2)──┘
```

After $K$ layers of message passing, a paper node's embedding contains rich contextual information from nodes up to $K$ hops away!

---

## 2. What "Heterogeneous" Adds (`HeteroData`)

In a standard (homogeneous) graph, all nodes are of the same type (e.g., only users in a social network) and all edges are identical.

Academic networks are **Heterogeneous Graphs** because they contain multiple distinct types of entities and relationships:

* **Node Types**:
  1. `paper`: The scientific publication being evaluated.
  2. `author`: The researcher who authored the paper.
  3. `institution`: The university, research lab, or company.
  4. `topic`: The research subfield (e.g., Computer Vision, NLP).

* **Edge Types (Multi-Relational)**:
  1. `(author, writes, paper)`
  2. `(author, affiliated_with, institution)`
  3. `(paper, cites, paper)`
  4. `(paper, has_topic, topic)`

In **PyTorch Geometric (PyG)**, `HeteroData` is the specialized data structure that stores separate feature matrices $\mathbf{X}_{\text{paper}}, \mathbf{X}_{\text{author}}, \mathbf{X}_{\text{inst}}$ and separate adjacency matrices for each edge relationship.

---

## 3. GraphSAGE vs. GAT: How They Differ in SciGraph AI

| Dimension | **HeteroGraphSAGE** (Sample & Aggregate) | **HeteroGAT** (Graph Attention Network) |
|---|---|---|
| **Core Mechanism** | Samples a fixed-size neighborhood and aggregates with uniform weights (e.g., $\text{mean}$ or $\text{sum}$). | Learns dynamic **attention weights** ($\alpha_{ij} \in [0, 1]$) for every connected neighbor. |
| **Project Example** | Gives equal aggregation weight to all co-authors on a paper. | Learns that a senior lead author (high historical h-index) should have higher attention weight than a first-time student co-author. |
| **Computational Cost** | Very fast, scalable to millions of nodes via mini-batch sampling. | Slightly more compute-intensive due to multi-head self-attention mechanisms ($\text{Softmax}(e_{ij})$). |

---

## 4. Why PyTorch Geometric (PyG) vs. Plain PyTorch?

Why not just write graph message passing in a Python `for` loop?

1. **Sparse Matrix Acceleration**: Real academic graphs have millions of zeros in their adjacency matrix. PyG uses C++/CUDA sparse tensor kernels that execute message passing $100\times$ faster than dense matrix multiplications.
2. **Multi-Relational Convolution (`HeteroConv`)**: PyG provides built-in `HeteroConv`, automatically applying different neural weights to different edge types in parallel.
3. **Neighbor Sampling**: PyG allows mini-batch training (`NeighborLoader`) without loading the entire graph into GPU RAM.

---

## 5. Worked, Hand-Traceable Example with Real Dataset Papers

Let's trace one single GNN message-passing step using 3 real papers from our dataset:

### 📄 Input Sample Data (Cutoff Snapshot $T_{\text{cutoff}}$):
* **Paper 1 (`W2194775991` - ResNet)**:
  - Pub Year: 2016 ($T_{\text{cutoff}} = 2016$)
  - Initial Features $\mathbf{x}_1 = [0.40, 0.10, 0.35, 0.40, 2.39]$
  - Connected Author: *Kaiming He* (Author Vector $\mathbf{h}_A = [0.85, 0.90]$)
  - Connected Institution: *Microsoft Research* ($\mathbf{h}_I = [0.95, 0.90]$)
* **Paper 2 (`W3118615836` - PRISMA 2020)**:
  - Pub Year: 2021 ($T_{\text{cutoff}} = 2021$)
  - Initial Features $\mathbf{x}_2 = [0.90, 0.30, 0.50, 0.30, 3.40]$
  - Connected Author: *Matthew J. Page* ($\mathbf{h}_B = [0.60, 0.50]$)
* **Paper 3 (`W3177828909` - AlphaFold)**:
  - Pub Year: 2021 ($T_{\text{cutoff}} = 2021$)
  - Initial Features $\mathbf{x}_3 = [0.90, 0.12, 0.40, 0.50, 2.50]$
  - Connected Author: *John Jumper* / *Demis Hassabis* ($\mathbf{h}_C = [0.90, 0.95]$)

---

### 🔢 Step-by-Step Message Passing Trace:

1. **Message from Author to Target Paper**:
   $$\mathbf{m}_{\text{author} \to \text{paper}} = \mathbf{W}_{\text{writes}} \cdot \mathbf{h}_{\text{author}}$$
   For ResNet, the author message brings strong positive weights ($+0.85$) into the paper's representation.

2. **Aggregation at Target Paper Node**:
   $$\mathbf{z}_1 = \text{ReLU}\left( \mathbf{W}_{\text{self}} \cdot \mathbf{x}_1 + \sum_{u \in \mathcal{N}(1)} \mathbf{m}_{u \to 1} \right)$$
   The paper node combines its early 2-year citations with the institutional and author reputation signals.

3. **Classification Layer**:
   $$\text{Logits} = \mathbf{W}_{\text{out}} \cdot \mathbf{z}_1 \quad \longrightarrow \quad \text{Softmax}(\text{Logits}) = [0.07, 0.21, 0.72]$$
   - $\text{Class 0 (Low Impact)}: 7.0\%$
   - $\text{Class 1 (Medium Impact)}: 21.0\%$
   - $\text{Class 2 (High Impact)}: 72.0\%$ $\longrightarrow$ **Predicted: High Impact!**

---

## 🎙️ 30-Second Seminar Defense Pitch:

> *"Our architecture uses PyTorch Geometric to represent science as a heterogeneous graph of papers, authors, and institutions. Through neighborhood message passing, the GNN aggregates collaborative reputation and institutional velocity to forecast 5-year citation trajectory at the moment of publication without temporal leakage."*
