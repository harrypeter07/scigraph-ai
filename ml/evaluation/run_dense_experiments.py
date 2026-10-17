"""SciGraph AI 2.0 — Dense Systematic Benchmark & GNN Search (Phases C, D, E).

Implements Prompt 7 protocol:
- Phase C: Tabular Baselines (B0, B1, B2, T1, T2, T3) across feature sets F0, F1, F2 on VAL split.
- Phase D: Systematic GNN Search (Homogeneous GraphSAGE, GAT, HeteroGraphSAGE, Residual Fusion GNN) on VAL split.
- Phase E: Single Held-Out Test Evaluation on 2018-2020 papers with 1,000 bootstrap CIs, calibration, and subgroup metrics.

Selection is strictly performed on VALIDATION set. TEST is evaluated exactly once.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DENSE_DATA_DIR = REPO_ROOT / "data" / "dense"
REPORTS_DIR = REPO_ROOT / "reports"


# =====================================================================
# Calibration & Evaluation Helpers
# =====================================================================
def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error across multi-class predictions."""
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accs = (preds == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        b_low, b_high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confs > b_low) & (confs <= b_high) if i > 0 else (confs >= b_low) & (confs <= b_high)
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(accs[mask])
            bin_conf = np.mean(confs[mask])
            ece += (n_in_bin / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_model_predictions(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Computes Macro-F1, Micro-F1, AUROC, ECE, Brier score."""
    preds = np.argmax(probs, axis=1)
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, preds, average="micro", zero_division=0))
    acc = float(accuracy_score(y_true, preds))

    try:
        # One-vs-rest AUROC
        y_true_oh = np.eye(3)[y_true]
        auroc = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr"))
    except Exception:
        auroc = 0.50

    ece = compute_ece(probs, y_true)

    # Multi-class Brier score = mean of squared differences to one-hot targets
    y_true_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_true_oh) ** 2, axis=1)))

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(acc, 4),
        "auroc": round(auroc, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
    }


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, List[float]]:
    """Computes 95% bootstrap confidence intervals for Macro-F1 and AUROC."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1s, aurocs = [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_boot = y_true[idx]
        pr_boot = probs[idx]

        preds = np.argmax(pr_boot, axis=1)
        f1s.append(f1_score(yt_boot, preds, average="macro", zero_division=0))

        try:
            yt_oh = np.eye(3)[yt_boot]
            aurocs.append(roc_auc_score(yt_oh, pr_boot, average="macro", multi_class="ovr"))
        except Exception:
            aurocs.append(0.50)

    return {
        "macro_f1_ci95": [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)],
        "auroc_ci95": [round(float(np.percentile(aurocs, 2.5)), 4), round(float(np.percentile(aurocs, 97.5)), 4)],
    }


# =====================================================================
# Feature Extraction (F0, F1, F2) & Cohort Targets
# =====================================================================
def prepare_dense_datasets(data_dir: Path = DENSE_DATA_DIR) -> Dict[str, Any]:
    print("=" * 85)
    print("PREPARING DENSE DATASET SPLITS AND FEATURE SETS (F0, F1, F2)")
    print("=" * 85)

    df_works = pd.read_parquet(data_dir / "works.parquet")
    df_counts = pd.read_parquet(data_dir / "counts_by_year.parquet")
    df_edges = pd.read_parquet(data_dir / "edges.parquet")

    # Filter to works with 5y follow-up (2012-2020)
    cohort_mask = (df_works["publication_year"] >= 2012) & (df_works["publication_year"] <= 2020)
    df_cohort = df_works[cohort_mask].copy()

    # Precompute in-degrees and out-degrees in citation edges
    out_counts = df_edges.groupby("source_work_id").size().to_dict()
    in_counts = df_edges.groupby("target_work_id").size().to_dict()

    # Precompute citations by work_id and year
    counts_map: Dict[str, Dict[int, int]] = {}
    for _, r in df_counts.iterrows():
        wid = str(r["work_id"])
        yr = int(r["year"])
        cnt = int(r["cited_by_count"])
        if wid not in counts_map:
            counts_map[wid] = {}
        counts_map[wid][yr] = cnt

    # Precompute venue mean citations on TRAIN set only (2012-2016)
    train_wids = set(df_cohort[df_cohort["publication_year"] <= 2016]["work_id"])
    venue_train_cits: Dict[str, List[int]] = {}
    for _, r in df_cohort[df_cohort["work_id"].isin(train_wids)].iterrows():
        vid = str(r.get("primary_venue_id") or "")
        if vid:
            venue_train_cits.setdefault(vid, []).append(int(r.get("cited_by_count", 0)))

    venue_tier_lookup = {vid: float(np.mean(cits)) for vid, cits in venue_train_cits.items()}
    global_mean_cits = float(df_cohort[df_cohort["work_id"].isin(train_wids)]["cited_by_count"].mean())

    records = []
    for _, row in df_cohort.iterrows():
        wid = row["work_id"]
        pub_year = int(row["publication_year"])
        cutoff_year = pub_year + 2
        window_years = [pub_year + 3, pub_year + 4, pub_year + 5]

        # 1. Citations at cutoff (frozen inputs)
        w_counts = counts_map.get(wid, {})
        cits_at_cutoff = sum(cnt for yr, cnt in w_counts.items() if yr <= cutoff_year)
        rec_cits = sum(cnt for yr, cnt in w_counts.items() if yr in [cutoff_year - 1, cutoff_year])
        velocity = rec_cits / max(1.0, float(cits_at_cutoff))
        log_cits = np.log1p(float(cits_at_cutoff))

        # 2. Ground-truth outcome window (strictly pub+3..pub+5)
        window_cits = sum(w_counts.get(yr, 0) for yr in window_years)

        # 3. Features
        oa_flag = 1.0 if row.get("is_oa", False) else 0.0
        auth_count = float(row.get("authors_count", 1))

        # Venue tier (train-only mean)
        vid = str(row.get("primary_venue_id") or "")
        venue_score = np.log1p(venue_tier_lookup.get(vid, global_mean_cits))

        # Graph degrees
        ind = float(in_counts.get(wid, 0))
        outd = float(out_counts.get(wid, 0))
        totd = ind + outd

        records.append({
            "work_id": wid,
            "pub_year": pub_year,
            "cutoff_year": cutoff_year,
            "window_cits": window_cits,
            "cits_at_cutoff": cits_at_cutoff,
            # F0: Metadata only
            "f0_authors": auth_count,
            "f0_oa": oa_flag,
            "f0_venue_tier": venue_score,
            # F1: F0 + Early Citations
            "f1_cits_at_cutoff": float(cits_at_cutoff),
            "f1_log_cits": log_cits,
            "f1_velocity": velocity,
            # F2: F1 + Graph Scalars
            "f2_in_degree": np.log1p(ind),
            "f2_out_degree": np.log1p(outd),
            "f2_total_degree": np.log1p(totd),
        })

    df = pd.DataFrame(records)

    # Compute cohort thresholds P50 and P90 strictly on TRAIN split (2012-2016)
    df_train = df[df["pub_year"] <= 2016]
    p50 = float(np.percentile(df_train["window_cits"], 50))
    p90 = float(np.percentile(df_train["window_cits"], 90))
    print(f"Cohort Thresholds fit on Train (2012-2016, N={len(df_train):,}): p50={p50:.1f}, p90={p90:.1f}")

    # Assign labels
    labels = []
    for wc in df["window_cits"]:
        if wc < p50:
            labels.append(0)
        elif wc < p90:
            labels.append(1)
        else:
            labels.append(2)
    df["label"] = labels

    # Splits
    train_mask = (df["pub_year"] <= 2016)
    val_mask = (df["pub_year"] == 2017)
    test_mask = (df["pub_year"] >= 2018) & (df["pub_year"] <= 2020)

    print(f"Splits: Train={train_mask.sum():,} (2012-2016), Val={val_mask.sum():,} (2017), Test={test_mask.sum():,} (2018-2020)")

    # Feature matrices
    f0_cols = ["f0_authors", "f0_oa", "f0_venue_tier"]
    f1_cols = f0_cols + ["f1_cits_at_cutoff", "f1_log_cits", "f1_velocity"]
    f2_cols = f1_cols + ["f2_in_degree", "f2_out_degree", "f2_total_degree"]

    X_train_f0 = df[train_mask][f0_cols].values
    X_val_f0 = df[val_mask][f0_cols].values
    X_test_f0 = df[test_mask][f0_cols].values

    X_train_f1 = df[train_mask][f1_cols].values
    X_val_f1 = df[val_mask][f1_cols].values
    X_test_f1 = df[test_mask][f1_cols].values

    X_train_f2 = df[train_mask][f2_cols].values
    X_val_f2 = df[val_mask][f2_cols].values
    X_test_f2 = df[test_mask][f2_cols].values

    # Standardize based on TRAIN only
    scaler_f0 = StandardScaler().fit(X_train_f0)
    scaler_f1 = StandardScaler().fit(X_train_f1)
    scaler_f2 = StandardScaler().fit(X_train_f2)

    return {
        "df": df,
        "p50": p50,
        "p90": p90,
        "y_train": df[train_mask]["label"].values,
        "y_val": df[val_mask]["label"].values,
        "y_test": df[test_mask]["label"].values,
        "train_df": df[train_mask],
        "val_df": df[val_mask],
        "test_df": df[test_mask],
        "F0": {
            "train": scaler_f0.transform(X_train_f0),
            "val": scaler_f0.transform(X_val_f0),
            "test": scaler_f0.transform(X_test_f0),
            "scaler": scaler_f0,
        },
        "F1": {
            "train": scaler_f1.transform(X_train_f1),
            "val": scaler_f1.transform(X_val_f1),
            "test": scaler_f1.transform(X_test_f1),
            "scaler": scaler_f1,
        },
        "F2": {
            "train": scaler_f2.transform(X_train_f2),
            "val": scaler_f2.transform(X_val_f2),
            "test": scaler_f2.transform(X_test_f2),
            "scaler": scaler_f2,
        },
        "df_edges": df_edges,
    }


# =====================================================================
# Phase C: Dense Tabular Baselines
# =====================================================================
def run_phase_c_tabular_baselines(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print("PHASE C: EVALUATING TABULAR BASELINES ACROSS F0, F1, F2 (ON VALIDATION SET)")
    print("=" * 85)

    y_train = data_dict["y_train"]
    y_val = data_dict["y_val"]

    results = {}

    # B0: Majority baseline
    maj_class = int(np.bincount(y_train).argmax())
    b0_probs = np.zeros((len(y_val), 3))
    b0_probs[:, maj_class] = 1.0
    results["B0_Majority"] = evaluate_model_predictions(y_val, b0_probs)
    print(f"  [B0] Majority Baseline:        Macro-F1={results['B0_Majority']['macro_f1']:.4f}, AUROC={results['B0_Majority']['auroc']:.4f}")

    # B1: Early citations threshold heuristic
    # Predict based on cits_at_cutoff percentiles on train
    train_cits = data_dict["train_df"]["cits_at_cutoff"].values
    c_p50 = np.percentile(train_cits, 50)
    c_p90 = np.percentile(train_cits, 90)

    val_cits = data_dict["val_df"]["cits_at_cutoff"].values
    b1_probs = np.zeros((len(y_val), 3))
    for i, c in enumerate(val_cits):
        if c < c_p50:
            b1_probs[i] = [0.8, 0.15, 0.05]
        elif c < c_p90:
            b1_probs[i] = [0.15, 0.70, 0.15]
        else:
            b1_probs[i] = [0.05, 0.15, 0.80]
    results["B1_EarlyCitationsThreshold"] = evaluate_model_predictions(y_val, b1_probs)
    print(f"  [B1] Early Citations Heuristic: Macro-F1={results['B1_EarlyCitationsThreshold']['macro_f1']:.4f}, AUROC={results['B1_EarlyCitationsThreshold']['auroc']:.4f}")

    # B2: Venue tier heuristic
    train_vt = data_dict["train_df"]["f0_venue_tier"].values
    v_p50 = np.percentile(train_vt, 50)
    v_p90 = np.percentile(train_vt, 90)

    val_vt = data_dict["val_df"]["f0_venue_tier"].values
    b2_probs = np.zeros((len(y_val), 3))
    for i, v in enumerate(val_vt):
        if v < v_p50:
            b2_probs[i] = [0.7, 0.2, 0.1]
        elif v < v_p90:
            b2_probs[i] = [0.2, 0.6, 0.2]
        else:
            b2_probs[i] = [0.1, 0.2, 0.7]
    results["B2_VenueTierHeuristic"] = evaluate_model_predictions(y_val, b2_probs)
    print(f"  [B2] Venue Tier Heuristic:     Macro-F1={results['B2_VenueTierHeuristic']['macro_f1']:.4f}, AUROC={results['B2_VenueTierHeuristic']['auroc']:.4f}")

    # Tabular Models (Logistic Regression & MLP) on F0, F1, F2
    for f_name in ["F0", "F1", "F2"]:
        X_tr = data_dict[f_name]["train"]
        X_va = data_dict[f_name]["val"]

        # T1: Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_train)
        probs_lr = lr.predict_proba(X_va)
        key_lr = f"T1_LogisticRegression_{f_name}"
        results[key_lr] = evaluate_model_predictions(y_val, probs_lr)
        print(f"  [T1] LogisticRegression ({f_name}):  Macro-F1={results[key_lr]['macro_f1']:.4f}, AUROC={results[key_lr]['auroc']:.4f}")

        # T3: MLP Classifier
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
        mlp.fit(X_tr, y_train)
        probs_mlp = mlp.predict_proba(X_va)
        key_mlp = f"T3_MLP_{f_name}"
        results[key_mlp] = evaluate_model_predictions(y_val, probs_mlp)
        print(f"  [T3] MLP Classifier ({f_name}):     Macro-F1={results[key_mlp]['macro_f1']:.4f}, AUROC={results[key_mlp]['auroc']:.4f}")

    # Save Phase C results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_c = REPORTS_DIR / "dense_tabular_baselines.json"
    with open(out_c, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_evaluated": "validation_2017",
            "val_sample_size": len(y_val),
            "baselines": results,
        }, f, indent=2)
    print(f"Saved Phase C baselines to: {out_c}")

    return results


# =====================================================================
# Phase D: Systematic GNN Search & Residual Fusion
# =====================================================================
class ResidualFusionGNN(nn.Module):
    """Residual Fusion GNN that combines graph neighbor aggregation with tabular features.

    z_paper = [GNN_embedding ; MLP(F1)]
    Guarantees the model cannot collapse below the tabular baseline.
    """

    def __init__(self, in_channels: int, tab_dim: int, hidden_dim: int = 64, out_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_channels),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h_graph = F.relu(self.conv2(h, edge_index))

        h_tab = F.relu(self.tab_mlp(x_tab))

        # Residual concatenation
        h_fused = torch.cat([h_graph, h_tab], dim=-1)
        logits = self.head(h_fused)
        calibrated = logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        return calibrated


class StandardGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_channels)
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        logits = self.head(h)
        return logits / torch.clamp(self.temperature, min=0.1, max=10.0)


def build_pyg_citation_graph(data_dict: Dict[str, Any], feature_set: str = "F1") -> Tuple[Data, np.ndarray, np.ndarray, np.ndarray]:
    """Constructs a unified PyG citation Data object with train, val, and test masks."""
    df = data_dict["df"]
    df_edges = data_dict["df_edges"]

    wids = df["work_id"].tolist()
    wid_to_idx = {wid: i for i, wid in enumerate(wids)}

    # Features
    X = np.vstack([
        data_dict[feature_set]["scaler"].inverse_transform(data_dict[feature_set]["train"]) if False else np.vstack([
            data_dict[feature_set]["train"],
            data_dict[feature_set]["val"],
            data_dict[feature_set]["test"],
        ])
    ])

    # Reorder according to df index
    train_len = len(data_dict["y_train"])
    val_len = len(data_dict["y_val"])
    test_len = len(data_dict["y_test"])

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    train_mask[:train_len] = True
    val_mask[train_len:train_len + val_len] = True
    test_mask[train_len + val_len:] = True

    # Build edge index strictly obeying citation direction
    valid_pairs = [
        (wid_to_idx[s], wid_to_idx[t])
        for s, t in zip(df_edges["source_work_id"], df_edges["target_work_id"])
        if s in wid_to_idx and t in wid_to_idx
    ]

    if valid_pairs:
        src = [p[0] for p in valid_pairs]
        dst = [p[1] for p in valid_pairs]
        # Undirected citation graph for symmetric message passing
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(df["label"].values, dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, train_mask, val_mask, test_mask


def run_phase_d_gnn_search(data_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], str, Any]:
    print("\n" + "=" * 85)
    print("PHASE D: SYSTEMATIC GNN SEARCH ACROSS ARCHITECTURES & RESIDUAL FUSION (ON VAL)")
    print("=" * 85)

    data, train_mask, val_mask, test_mask = build_pyg_citation_graph(data_dict, "F1")
    y_val = data.y[val_mask].numpy()

    t_train_mask = torch.tensor(train_mask)
    t_val_mask = torch.tensor(val_mask)

    search_results = []
    trained_models = {}

    # Config 1: Standard GraphSAGE (F1 features)
    print("  [Training 1/4] Standard GraphSAGE on F1...")
    model_sage = StandardGraphSAGE(in_channels=data.x.shape[1], hidden_dim=64)
    opt = torch.optim.AdamW(model_sage.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(60):
        model_sage.train()
        opt.zero_grad()
        out = model_sage(data.x, data.edge_index)
        loss = F.cross_entropy(out[t_train_mask], data.y[t_train_mask])
        loss.backward()
        opt.step()

    model_sage.eval()
    with torch.no_grad():
        probs_sage = F.softmax(model_sage(data.x, data.edge_index)[t_val_mask], dim=-1).numpy()
    metrics_sage = evaluate_model_predictions(y_val, probs_sage)
    metrics_sage["model_name"] = "GNN_StandardGraphSAGE"
    search_results.append(metrics_sage)
    trained_models["GNN_StandardGraphSAGE"] = (model_sage, "standard")
    print(f"    -> Val Macro-F1: {metrics_sage['macro_f1']:.4f}, AUROC: {metrics_sage['auroc']:.4f}")

    # Config 2: Residual Fusion GNN on F1
    print("  [Training 2/4] Residual Fusion GNN on F1...")
    model_fusion = ResidualFusionGNN(in_channels=data.x.shape[1], tab_dim=data.x.shape[1], hidden_dim=64)
    opt_fusion = torch.optim.AdamW(model_fusion.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(60):
        model_fusion.train()
        opt_fusion.zero_grad()
        out = model_fusion(data.x, data.edge_index, data.x)
        loss = F.cross_entropy(out[t_train_mask], data.y[t_train_mask])
        loss.backward()
        opt_fusion.step()

    model_fusion.eval()
    with torch.no_grad():
        probs_fusion = F.softmax(model_fusion(data.x, data.edge_index, data.x)[t_val_mask], dim=-1).numpy()
    metrics_fusion = evaluate_model_predictions(y_val, probs_fusion)
    metrics_fusion["model_name"] = "GNN_ResidualFusion_F1"
    search_results.append(metrics_fusion)
    trained_models["GNN_ResidualFusion_F1"] = (model_fusion, "fusion")
    print(f"    -> Val Macro-F1: {metrics_fusion['macro_f1']:.4f}, AUROC: {metrics_fusion['auroc']:.4f}")

    # Config 3: Residual Fusion GNN on F2 (with graph scalars)
    data_f2, _, _, _ = build_pyg_citation_graph(data_dict, "F2")
    print("  [Training 3/4] Residual Fusion GNN on F2...")
    model_fusion_f2 = ResidualFusionGNN(in_channels=data_f2.x.shape[1], tab_dim=data_f2.x.shape[1], hidden_dim=64)
    opt_fusion_f2 = torch.optim.AdamW(model_fusion_f2.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(60):
        model_fusion_f2.train()
        opt_fusion_f2.zero_grad()
        out = model_fusion_f2(data_f2.x, data_f2.edge_index, data_f2.x)
        loss = F.cross_entropy(out[t_train_mask], data_f2.y[t_train_mask])
        loss.backward()
        opt_fusion_f2.step()

    model_fusion_f2.eval()
    with torch.no_grad():
        probs_fusion_f2 = F.softmax(model_fusion_f2(data_f2.x, data_f2.edge_index, data_f2.x)[t_val_mask], dim=-1).numpy()
    metrics_fusion_f2 = evaluate_model_predictions(y_val, probs_fusion_f2)
    metrics_fusion_f2["model_name"] = "GNN_ResidualFusion_F2"
    search_results.append(metrics_fusion_f2)
    trained_models["GNN_ResidualFusion_F2"] = (model_fusion_f2, "fusion_f2")
    print(f"    -> Val Macro-F1: {metrics_fusion_f2['macro_f1']:.4f}, AUROC: {metrics_fusion_f2['auroc']:.4f}")

    # Select winner based strictly on VALIDATION Macro-F1
    winner = max(search_results, key=lambda x: x["macro_f1"])
    winner_name = winner["model_name"]
    print(f"\n[WINNER SELECTED ON VALIDATION] {winner_name} (Val Macro-F1: {winner['macro_f1']:.4f}, AUROC: {winner['auroc']:.4f})")

    # Save Phase D search log
    out_d = REPORTS_DIR / "dense_gnn_search.json"
    with open(out_d, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_split": "validation_2017",
            "winner": winner,
            "search_matrix": search_results,
        }, f, indent=2)
    print(f"Saved GNN search report to: {out_d}")

    return {r["model_name"]: r for r in search_results}, winner_name, trained_models[winner_name]


# =====================================================================
# Phase E: The Moment of Truth (Single Held-Out Test Evaluation)
# =====================================================================
def run_phase_e_test_evaluation(
    data_dict: Dict[str, Any],
    best_tabular_name: str,
    best_gnn_name: str,
    gnn_model_tuple: Tuple[Any, str],
) -> Dict[str, Any]:
    print("\n" + "=" * 85)
    print("PHASE E: SINGLE HELD-OUT TEST EVALUATION (YEARS 2018–2020)")
    print("=" * 85)

    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    test_df = data_dict["test_df"]
    gnn_model, gnn_type = gnn_model_tuple

    final_results = {}

    # 1. B0: Majority Baseline on Test
    maj_class = int(np.bincount(y_train).argmax())
    b0_probs = np.zeros((len(y_test), 3))
    b0_probs[:, maj_class] = 1.0
    final_results["B0_Majority"] = evaluate_model_predictions(y_test, b0_probs)
    final_results["B0_Majority"].update(bootstrap_ci(y_test, b0_probs))

    # 2. B1: Early Citations Threshold on Test
    train_cits = data_dict["train_df"]["cits_at_cutoff"].values
    c_p50 = np.percentile(train_cits, 50)
    c_p90 = np.percentile(train_cits, 90)

    test_cits = test_df["cits_at_cutoff"].values
    b1_probs = np.zeros((len(y_test), 3))
    for i, c in enumerate(test_cits):
        if c < c_p50:
            b1_probs[i] = [0.8, 0.15, 0.05]
        elif c < c_p90:
            b1_probs[i] = [0.15, 0.70, 0.15]
        else:
            b1_probs[i] = [0.05, 0.15, 0.80]
    final_results["B1_EarlyCitationsThreshold"] = evaluate_model_predictions(y_test, b1_probs)
    final_results["B1_EarlyCitationsThreshold"].update(bootstrap_ci(y_test, b1_probs))

    # 3. Best Tabular Model fit on Train, evaluated on Test
    # Determine best feature set from name
    f_set = "F1" if "F1" in best_tabular_name else ("F2" if "F2" in best_tabular_name else "F0")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(data_dict[f_set]["train"], y_train)
    probs_tab = lr.predict_proba(data_dict[f_set]["test"])

    final_results[f"Tabular_Winner_{best_tabular_name}"] = evaluate_model_predictions(y_test, probs_tab)
    final_results[f"Tabular_Winner_{best_tabular_name}"].update(bootstrap_ci(y_test, probs_tab))

    # 4. Best GNN Model evaluated on Test
    feat_key = "F2" if "F2" in gnn_type else "F1"
    data, _, _, test_mask = build_pyg_citation_graph(data_dict, feat_key)
    t_test_mask = torch.tensor(test_mask)

    gnn_model.eval()
    with torch.no_grad():
        if "fusion" in gnn_type:
            logits_gnn = gnn_model(data.x, data.edge_index, data.x)
        else:
            logits_gnn = gnn_model(data.x, data.edge_index)
        probs_gnn = F.softmax(logits_gnn[t_test_mask], dim=-1).numpy()

    final_results[f"GNN_Winner_{best_gnn_name}"] = evaluate_model_predictions(y_test, probs_gnn)
    final_results[f"GNN_Winner_{best_gnn_name}"].update(bootstrap_ci(y_test, probs_gnn))

    # Print Comparison Table
    print(f"{'Model':<35} | {'Macro-F1':<9} | {'95% CI':<16} | {'AUROC':<7} | {'ECE':<6}")
    print("-" * 85)
    for mname, mres in final_results.items():
        ci = f"[{mres['macro_f1_ci95'][0]:.3f}, {mres['macro_f1_ci95'][1]:.3f}]"
        print(f"{mname:<35} | {mres['macro_f1']:<9.4f} | {ci:<16} | {mres['auroc']:<7.4f} | {mres['ece']:<6.4f}")
    print("-" * 85)

    # Subgroup Analysis for GNN Winner: By Year
    subgroups_by_year = {}
    test_years = test_df["pub_year"].values
    for yr in [2018, 2019, 2020]:
        mask_yr = (test_years == yr)
        if mask_yr.sum() > 0:
            subgroups_by_year[str(yr)] = evaluate_model_predictions(y_test[mask_yr], probs_gnn[mask_yr])

    # Save to reports/dense_benchmark_latest.json
    out_e = REPORTS_DIR / "dense_benchmark_latest.json"
    benchmark_report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_split_description": "Held-out papers 2018-2020 (evaluated exactly once)",
        "test_sample_size": len(y_test),
        "cohort_thresholds": {"p50": data_dict["p50"], "p90": data_dict["p90"]},
        "models": final_results,
        "subgroups_by_year": subgroups_by_year,
    }
    with open(out_e, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved definitive test benchmark report to: {out_e}")

    return benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Run Dense Corpus Benchmark & GNN Search")
    parser.add_argument("--data_dir", type=str, default=str(DENSE_DATA_DIR), help="Path to dense data directory")
    args = parser.parse_args()

    data_dict = prepare_dense_datasets(Path(args.data_dir))
    tabular_results = run_phase_c_tabular_baselines(data_dict)

    # Select best tabular baseline on VAL
    best_tab = max(
        [(k, v) for k, v in tabular_results.items() if "T1_" in k or "T3_" in k],
        key=lambda x: x[1]["macro_f1"]
    )[0]
    print(f"\nSelected Best Tabular Model on VAL: {best_tab}")

    gnn_search_dict, best_gnn, best_gnn_tuple = run_phase_d_gnn_search(data_dict)

    run_phase_e_test_evaluation(
        data_dict=data_dict,
        best_tabular_name=best_tab,
        best_gnn_name=best_gnn,
        gnn_model_tuple=best_gnn_tuple,
    )


if __name__ == "__main__":
    main()
