# SciGraph AI — Mathematical Formulas, Features & Architecture Flowchart

**Official Academic & Mathematical Reference Guide**

---

## 🗺️ 1. Complete Architecture Flowchart

```
[ Raw OpenAlex Paper: Title, Publication Year, Authors, Citations ]
                                │
                                ▼
        [ 1. Temporal Freeze at T_cutoff = Y_pub ]
     (Future citations masked to ensure zero leakage)
                                │
                                ▼
        [ 2. Feature Extraction (5-Dimensional Vector x) ]
  ┌─────────────────────────────────────────────────────────────┐
  │ x0 = (PubYear - 2012) / 10     (Normalized Publication Year) │
  │ x1 = Citations_cutoff / 100    (Early Citation Velocity)     │
  │ x2 = Title_Word_Count / 20     (Title Complexity Ratio)      │
  │ x3 = CoAuthor_Count / 10       (Author Collaboration Ratio)  │
  │ x4 = ln(1 + Citations_cutoff)  (Log-Scaled Citation Velocity)│
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
        [ 3. Feed-Forward Neural Network (GraphSAGE GNN) ]
  ┌─────────────────────────────────────────────────────────────┐
  │ Layer 1: h1 = ReLU( W1 · x + b1 )    [64 Hidden Neurons]    │
  │ Layer 2: z  = W2 · h1 + b2           [3 Raw Logits: z0,z1,z2]│
  │ Softmax: P(Class i) = exp(zi) / sum(exp(zj))                │
  └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
     [ 4. Dynamic Cohort Percentile Decision (Argmax) ]
  ┌─────────────────────────────────────────────────────────────┐
  │ Class 0: Low Impact    (< 50th Percentile of that Year)     │
  │ Class 1: Medium Impact (50th to 90th Percentile of Year)    │
  │ Class 2: High Impact   (>= 90th Percentile - Top 10% Hits)  │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 2. Deep Dive: The 5 Extracted Features (Why Only These & What They Tell Us)

Every scientific paper is converted into a 5-dimensional numerical feature vector:

$$\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^T$$

| Feature Index | Exact Formula | Scientometric Rationale (Why We Extracted This) | What It Tells the Model |
|---|---|---|---|
| **$x_0$** | `x0 = (PubYear - 2012) / 10` | Academic publishing volume and baseline citation inflation change across decades. | Anchors the paper in its temporal era, scaling years `[2012, 2022]` into a clean `[0.0, 1.0]` range. |
| **$x_1$** | `x1 = Citations_cutoff / 100` | Immediate citation velocity in the release year indicates fast community recognition. | Measures early adoption velocity without looking into the future. Dividing by 100 normalizes values for neural gradients. |
| **$x_2$** | `x2 = Title_Word_Count / 20` | Studies in *Nature* show concise titles correlate with broad fundamental breakthroughs, whereas long titles indicate narrow niche applications. | Measures the lexical scope and granularity of the scientific contribution (bounded between 0.2 and 1.0). |
| **$x_3$** | `x3 = CoAuthor_Count / 10` | Modern scientific breakthroughs are collaborative. Multi-author, multi-lab teams have larger dissemination networks. | Indicates the collaborative breadth and institutional reach behind the research. |
| **$x_4$** | `x4 = ln(1 + Citations_cutoff)` | Citations follow a heavy-tailed power-law distribution (0.1% receive thousands of citations). | Compresses extreme outliers so viral papers do not overwhelm neural network weights while preserving rank. |

---

## 🧮 3. Feed-Forward Neural Network Step-by-Step

In [`ml/gnn/models/graphsage.py`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/ml/gnn/models/graphsage.py):

### Step 1: Input Feature Vector
$$\mathbf{x} = [x_0, x_1, x_2, x_3, x_4]^T \in \mathbb{R}^{5 \times 1}$$

### Step 2: Hidden Layer Linear Transformation (64 Neurons)
$$\mathbf{h}_1 = \mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1$$
* $\mathbf{W}_1 \in \mathbb{R}^{64 \times 5}$ (Trained weight matrix)
* $\mathbf{b}_1 \in \mathbb{R}^{64 \times 1}$ (Trained bias vector)

### Step 3: Non-Linear Activation (ReLU)
$$\mathbf{a}_1 = \max(0, \mathbf{h}_1) \quad (\text{Rectified Linear Unit})$$

### Step 4: Output Classifier Layer (3 Logits)
$$\mathbf{z} = \mathbf{W}_2 \cdot \mathbf{a}_1 + \mathbf{b}_2 = [z_0, z_1, z_2]^T$$
* $\mathbf{W}_2 \in \mathbb{R}^{3 \times 64}$ (Trained classifier weights)
* $\mathbf{b}_2 \in \mathbb{R}^{3 \times 1}$ (Trained classifier bias)
* $\mathbf{z}$ represents the raw unnormalized class scores.

### Step 5: Softmax Probability Distribution (Sums to 100%)
$$P(\text{Low Impact}) = \frac{e^{z_0}}{e^{z_0} + e^{z_1} + e^{z_2}}$$
$$P(\text{Medium Impact}) = \frac{e^{z_1}}{e^{z_0} + e^{z_1} + e^{z_2}}$$
$$P(\text{High Impact}) = \frac{e^{z_2}}{e^{z_0} + e^{z_1} + e^{z_2}}$$

### Step 6: Argmax Final Decision Rule
$$\text{Predicted Class} = \arg\max_{i \in \{0, 1, 2\}} P(\text{Class } i)$$

---

## 📊 4. Dynamic Cohort Percentile Labeling

In [`ml/labels/labeler.py`](file:///c:/Users/ASUS/Documents/SECOND%20SEMISTER/INTERNSHIP/scigraph/ml/labels/labeler.py), ground truth labels are **dynamically computed per publication year cohort**:

1. **5-Year Citation Growth**:
   $$\Delta \text{Citations}_5(p) = \text{Citations}(p, Y_{\text{pub}} + 5) - \text{Citations}(p, Y_{\text{pub}})$$

2. **Cohort Percentiles**:
   * $P_{50}(Y_{\text{pub}})$ = Median 5-year gain of papers published in that year.
   * $P_{90}(Y_{\text{pub}})$ = $90^{\text{th}}$ percentile 5-year gain of papers published in that year.

3. **Class Assignment**:
   * **Class 0 (Low Impact)**: $\Delta \text{Citations}_5 < P_{50}$ (Bottom 50% of papers that year).
   * **Class 1 (Medium Impact)**: $P_{50} \le \Delta \text{Citations}_5 < P_{90}$ (Mainstream solid contribution).
   * **Class 2 (High Impact)**: $\Delta \text{Citations}_5 \ge P_{90}$ (Top 10% Landmark Breakthrough Papers).
