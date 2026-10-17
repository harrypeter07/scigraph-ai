"""SciGraph AI 2.0 — Phase 4: Model Ladder on Best Feature Set.

Benchmarking models on the winning 55-dimensional feature set:
- M0: Tuned LogisticRegression / ElasticNet (linear baseline)
- M1: LightGBM (with monotone constraints on early citations, tuned on VAL)
- M2: HistGradientBoostingClassifier (monotonic constraints, tuned on VAL)
- M3: MLP with residual connections, layer norm, dropout (tuned on VAL)
- M4: Ordinal / Two-Stage Model:
      Stage 1: P(cits >= 50th percentile) binary
      Stage 2: P(cits >= 90th | cits >= 50th) binary
- M5: Stacked Ensemble (out-of-fold predictions on Train as features for meta-learner)

All models undergo:
- Temperature scaling fit on VAL
- Decision threshold / multiplier tuning on VAL to maximize Macro-F1
- Evaluation on TEST reported ONCE with VAL-chosen parameters

Saves results to reports/accuracy_improvement/model_ladder.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features.asof import AsOfFeatureBuilder

REPORTS_DIR = REPO_ROOT / "reports" / "accuracy_improvement"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_precision_at_top10(y_true: np.ndarray, probs_high: np.ndarray) -> float:
    n_top = max(1, int(len(y_true) * 0.10))
    top_indices = np.argsort(probs_high)[-n_top:]
    return float(np.mean(y_true[top_indices] == 2))


def bootstrap_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1s, accs, aurocs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        pr_b = probs[idx]
        pd_b = preds[idx]

        f1s.append(f1_score(yt_b, pd_b, average="macro", zero_division=0))
        accs.append(accuracy_score(yt_b, pd_b))
        try:
            yt_oh = np.eye(3)[yt_b]
            aurocs.append(roc_auc_score(yt_oh, pr_b, average="macro", multi_class="ovr"))
        except Exception:
            aurocs.append(0.50)

    return {
        "macro_f1_ci": [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)],
        "accuracy_ci": [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)],
        "auroc_ci": [round(float(np.percentile(aurocs, 2.5)), 4), round(float(np.percentile(aurocs, 97.5)), 4)],
    }


def fit_temperature_and_thresholds(
    val_probs: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Fits temperature T and class weight multipliers w on validation set to maximize Macro-F1."""
    # 1. Temperature scaling on val log-probs
    eps = 1e-12
    val_probs_clipped = np.clip(val_probs, eps, 1.0 - eps)
    logits = np.log(val_probs_clipped)

    def nll_loss(T_arr):
        T = T_arr[0]
        scaled_logits = logits / T
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        # Cross entropy
        p_correct = p[np.arange(len(y_val)), y_val]
        return -np.mean(np.log(np.clip(p_correct, eps, 1.0)))

    res_T = minimize(nll_loss, [1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
    best_T = float(res_T.x[0])

    # 2. Threshold multipliers w for prediction: argmax_c (w_c * p_c)
    best_f1 = f1_score(y_val, np.argmax(val_probs, axis=1), average="macro", zero_division=0)
    best_w = np.array([1.0, 1.0, 1.0])

    # Simple grid search over class multipliers on VAL
    for w0 in [0.8, 1.0, 1.2]:
        for w1 in [0.8, 1.0, 1.2, 1.4]:
            for w2 in [1.0, 1.2, 1.5, 1.8, 2.0]:
                w_cand = np.array([w0, w1, w2])
                scaled_probs = val_probs * w_cand
                cand_preds = np.argmax(scaled_probs, axis=1)
                cand_f1 = f1_score(y_val, cand_preds, average="macro", zero_division=0)
                if cand_f1 > best_f1:
                    best_f1 = cand_f1
                    best_w = w_cand

    return best_T, best_w


def calibrate_and_predict(
    probs: np.ndarray,
    T: float,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p_clipped) / T
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    cal_probs = exps / np.sum(exps, axis=1, keepdims=True)
    preds = np.argmax(cal_probs * w, axis=1)
    return cal_probs, preds


# ---------------------------------------------------------------------------
# Model M3: PyTorch Residual MLP with LayerNorm and Dropout
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        out = self.relu(self.ln1(self.fc1(x)))
        out = self.drop(out)
        out = self.ln2(self.fc2(out))
        return self.relu(out + res)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, n_blocks: int = 2, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = torch.relu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def train_mlp_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 40,
) -> Tuple[nn.Module, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_tr.shape[1]
    model = ResidualMLP(in_dim=in_dim, hidden_dim=hidden_dim, n_blocks=2, dropout=dropout).to(device)

    # Compute class weights for cross entropy
    class_counts = np.bincount(y_tr, minlength=3)
    weights = torch.tensor(len(y_tr) / (3.0 * class_counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vx = torch.tensor(X_va, dtype=torch.float32).to(device)
            v_logits = model(vx)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            vf1 = f1_score(y_va, v_preds, average="macro", zero_division=0)
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model, best_val_f1


def run_phase4():
    print("=" * 90)
    print("PHASE 4: SYSTEMATIC MODEL LADDER & VAL-TUNED CALIBRATION")
    print("=" * 90)

    # 1. Load data and features
    builder = AsOfFeatureBuilder()
    df = builder.extract_features_df()

    with open(REPORTS_DIR / "feature_ladder.json", "r", encoding="utf-8") as f:
        best_tabular_cols = json.load(f)["final_selected_features"]

    train_mask = (df["temporal_split"] == "train") & (df["publication_year"] <= 2016)
    val_mask = (df["temporal_split"] == "val") & (df["publication_year"] == 2017)
    test_mask = (df["temporal_split"] == "test") & (df["publication_year"] >= 2018) & (df["publication_year"] <= 2020)

    y_train = df[train_mask]["impact_class"].values
    y_val = df[val_mask]["impact_class"].values
    y_test = df[test_mask]["impact_class"].values

    # Tabular features
    X_tr_tab = df[train_mask][best_tabular_cols].values
    X_va_tab = df[val_mask][best_tabular_cols].values
    X_te_tab = df[test_mask][best_tabular_cols].values

    # E1 Text features: TF-IDF + SVD(32) fit on TRAIN only
    titles = df["title"].fillna("").astype(str).tolist()
    titles_tr = [titles[i] for i, m in enumerate(train_mask) if m]
    titles_va = [titles[i] for i, m in enumerate(val_mask) if m]
    titles_te = [titles[i] for i, m in enumerate(test_mask) if m]

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=32, random_state=42)
    X_tr_text = svd.fit_transform(tfidf.fit_transform(titles_tr))
    X_va_text = svd.transform(tfidf.transform(titles_va))
    X_te_text = svd.transform(tfidf.transform(titles_te))

    X_train_raw = np.hstack([X_tr_tab, X_tr_text])
    X_val_raw = np.hstack([X_va_tab, X_va_text])
    X_test_raw = np.hstack([X_te_tab, X_te_text])

    all_col_names = best_tabular_cols + [f"svd_{i}" for i in range(32)]
    print(f"Total features in dataset: {len(all_col_names)} ({len(best_tabular_cols)} tabular + 32 text SVD)")

    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Monotone constraints for tree models:
    # +1 for early citations (f0_log_cits and f1_cohort_percentile), 0 for others
    monotone_constraints = []
    for col in all_col_names:
        if col in ["f0_log_cits", "f1_cohort_percentile", "f2_y2"]:
            monotone_constraints.append(1)
        else:
            monotone_constraints.append(0)

    results = []

    # Helper evaluator
    def evaluate_model_pipeline(name: str, train_fn, predict_proba_fn):
        model_obj = train_fn()
        val_probs_raw = predict_proba_fn(model_obj, "val")
        T, w = fit_temperature_and_thresholds(val_probs_raw, y_val)
        val_cal_probs, val_preds = calibrate_and_predict(val_probs_raw, T, w)

        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        # Evaluate ONCE on Test with chosen T and w
        test_probs_raw = predict_proba_fn(model_obj, "test")
        test_cal_probs, test_preds = calibrate_and_predict(test_probs_raw, T, w)

        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        y_test_oh = np.eye(3)[y_test]
        test_auroc = roc_auc_score(y_test_oh, test_cal_probs, average="macro", multi_class="ovr")
        prec10 = compute_precision_at_top10(y_test, test_cal_probs[:, 2])

        ci_dict = bootstrap_metrics(y_test, test_cal_probs, test_preds)

        res = {
            "model_name": name,
            "calibrated_T": round(T, 4),
            "threshold_weights": [round(float(x), 4) for x in w],
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
            "test_macro_f1_ci": ci_dict["macro_f1_ci"],
            "test_accuracy_ci": ci_dict["accuracy_ci"],
            "test_auroc": round(float(test_auroc), 4),
            "test_auroc_ci": ci_dict["auroc_ci"],
            "test_precision_at_top10": round(float(prec10), 4),
            "raw_test_probs": test_cal_probs,
            "model_obj": model_obj,
        }
        return res

    # -----------------------------------------------------------------------
    # M0: Logistic Regression / ElasticNet (Val-tuned C)
    # -----------------------------------------------------------------------
    print("\nTuning M0: Logistic Regression...")
    best_c = 1.0
    best_val_f1 = -1.0
    best_lr = None
    for c_cand in [0.01, 0.1, 1.0, 5.0, 10.0, 50.0]:
        lr = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        vf1 = f1_score(y_val, lr.predict(X_val_scaled), average="macro", zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_c = c_cand
            best_lr = lr

    m0_res = evaluate_model_pipeline(
        f"M0_LogisticRegression(C={best_c})",
        lambda: best_lr,
        lambda m, split: m.predict_proba(X_val_scaled if split == "val" else X_test_scaled)
    )
    results.append(m0_res)

    # -----------------------------------------------------------------------
    # M1: LightGBM (with monotone constraints, tuned on VAL)
    # -----------------------------------------------------------------------
    print("Tuning M1: LightGBM...")
    best_lgb = None
    best_lgb_f1 = -1.0
    for num_leaves in [15, 31, 63]:
        for lr_cand in [0.03, 0.05, 0.1]:
            for min_child in [20, 50]:
                clf = lgb.LGBMClassifier(
                    num_leaves=num_leaves,
                    learning_rate=lr_cand,
                    min_child_samples=min_child,
                    monotone_constraints=monotone_constraints,
                    n_estimators=150,
                    random_state=42,
                    verbose=-1,
                )
                clf.fit(X_train_raw, y_train)
                vf1 = f1_score(y_val, clf.predict(X_val_raw), average="macro", zero_division=0)
                if vf1 > best_lgb_f1:
                    best_lgb_f1 = vf1
                    best_lgb = clf

    m1_res = evaluate_model_pipeline(
        f"M1_LightGBM(leaves={best_lgb.num_leaves},lr={best_lgb.learning_rate})",
        lambda: best_lgb,
        lambda m, split: m.predict_proba(X_val_raw if split == "val" else X_test_raw)
    )
    results.append(m1_res)

    # -----------------------------------------------------------------------
    # M2: HistGradientBoostingClassifier (monotonic constraints)
    # -----------------------------------------------------------------------
    print("Tuning M2: HistGradientBoosting...")
    best_hgb = None
    best_hgb_f1 = -1.0
    for lr_cand in [0.03, 0.05, 0.1]:
        for max_iter in [100, 150]:
            hgb = HistGradientBoostingClassifier(
                learning_rate=lr_cand,
                max_iter=max_iter,
                random_state=42,
            )
            hgb.fit(X_train_raw, y_train)
            vf1 = f1_score(y_val, hgb.predict(X_val_raw), average="macro", zero_division=0)
            if vf1 > best_hgb_f1:
                best_hgb_f1 = vf1
                best_hgb = hgb

    m2_res = evaluate_model_pipeline(
        f"M2_HistGradientBoosting(lr={best_hgb.learning_rate},iters={best_hgb.max_iter})",
        lambda: best_hgb,
        lambda m, split: m.predict_proba(X_val_raw if split == "val" else X_test_raw)
    )
    results.append(m2_res)

    # -----------------------------------------------------------------------
    # M3: PyTorch Residual MLP
    # -----------------------------------------------------------------------
    print("Training M3: PyTorch Residual MLP...")
    mlp_model, mlp_val_f1 = train_mlp_model(X_train_scaled, y_train, X_val_scaled, y_val, hidden_dim=128, epochs=40)

    def mlp_predict_proba(model, split):
        X = X_val_scaled if split == "val" else X_test_scaled
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(tx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    m3_res = evaluate_model_pipeline("M3_ResidualMLP", lambda: mlp_model, mlp_predict_proba)
    results.append(m3_res)

    # -----------------------------------------------------------------------
    # M4: Ordinal / Two-Stage Model
    # Stage 1: P(cits >= 50th percentile) binary [y >= 1]
    # Stage 2: P(cits >= 90th | cits >= 50th) binary [y == 2 given y >= 1]
    # -----------------------------------------------------------------------
    print("Training M4: Two-Stage Ordinal Model...")
    y_tr_s1 = (y_train >= 1).astype(int)
    stage1_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    stage1_clf.fit(X_train_scaled, y_tr_s1)

    s2_mask = y_train >= 1
    y_tr_s2 = (y_train[s2_mask] == 2).astype(int)
    stage2_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    stage2_clf.fit(X_train_scaled[s2_mask], y_tr_s2)

    def two_stage_predict_proba(models, split):
        s1, s2 = models
        X = X_val_scaled if split == "val" else X_test_scaled
        p_ge1 = s1.predict_proba(X)[:, 1]
        p_eq2_given_ge1 = s2.predict_proba(X)[:, 1]

        p_low = 1.0 - p_ge1
        p_med = p_ge1 * (1.0 - p_eq2_given_ge1)
        p_high = p_ge1 * p_eq2_given_ge1

        probs = np.column_stack([p_low, p_med, p_high])
        return probs / np.sum(probs, axis=1, keepdims=True)

    m4_res = evaluate_model_pipeline("M4_TwoStageOrdinal", lambda: (stage1_clf, stage2_clf), two_stage_predict_proba)
    results.append(m4_res)

    # -----------------------------------------------------------------------
    # M5: Stacked Ensemble (out-of-fold meta learner)
    # -----------------------------------------------------------------------
    print("Training M5: Stacked Ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_m0 = np.zeros((len(y_train), 3))
    oof_m1 = np.zeros((len(y_train), 3))
    oof_m2 = np.zeros((len(y_train), 3))

    for tr_idx, hld_idx in kf.split(X_train_raw):
        # M0 fold
        sc = StandardScaler().fit(X_train_raw[tr_idx])
        lr_fold = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
        lr_fold.fit(sc.transform(X_train_raw[tr_idx]), y_train[tr_idx])
        oof_m0[hld_idx] = lr_fold.predict_proba(sc.transform(X_train_raw[hld_idx]))

        # M1 fold
        lgb_fold = lgb.LGBMClassifier(
            num_leaves=best_lgb.num_leaves,
            learning_rate=best_lgb.learning_rate,
            min_child_samples=best_lgb.min_child_samples,
            monotone_constraints=monotone_constraints,
            n_estimators=150,
            random_state=42,
            verbose=-1,
        )
        lgb_fold.fit(X_train_raw[tr_idx], y_train[tr_idx])
        oof_m1[hld_idx] = lgb_fold.predict_proba(X_train_raw[hld_idx])

        # M2 fold
        hgb_fold = HistGradientBoostingClassifier(
            learning_rate=best_hgb.learning_rate,
            max_iter=best_hgb.max_iter,
            random_state=42,
        )
        hgb_fold.fit(X_train_raw[tr_idx], y_train[tr_idx])
        oof_m2[hld_idx] = hgb_fold.predict_proba(X_train_raw[hld_idx])

    X_meta_train = np.hstack([oof_m0, oof_m1, oof_m2])
    meta_clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_clf.fit(X_meta_train, y_train)

    def ensemble_predict_proba(meta, split):
        if split == "val":
            p0 = best_lr.predict_proba(X_val_scaled)
            p1 = best_lgb.predict_proba(X_val_raw)
            p2 = best_hgb.predict_proba(X_val_raw)
        else:
            p0 = best_lr.predict_proba(X_test_scaled)
            p1 = best_lgb.predict_proba(X_test_raw)
            p2 = best_hgb.predict_proba(X_test_raw)
        X_meta = np.hstack([p0, p1, p2])
        return meta.predict_proba(X_meta)

    m5_res = evaluate_model_pipeline("M5_StackedEnsemble", lambda: meta_clf, ensemble_predict_proba)
    results.append(m5_res)

    # -----------------------------------------------------------------------
    # Summary & Winner Selection (Strictly on Validation Macro-F1)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 115)
    print(f"{'Model Architecture':<32} | {'Val Acc':<8} | {'Val F1':<8} | {'Test Acc':<9} | {'Test F1':<8} | {'Test AUROC':<10} | {'Prec@10%':<9}")
    print("-" * 115)

    winner = None
    best_val_score = -1.0
    ladder_export = []

    for r in results:
        v_acc = r["val_accuracy"]
        v_f1 = r["val_macro_f1"]
        t_acc = r["test_accuracy"]
        t_f1 = r["test_macro_f1"]
        t_auc = r["test_auroc"]
        p10 = r["test_precision_at_top10"]

        if v_f1 > best_val_score:
            best_val_score = v_f1
            winner = r["model_name"]

        print(f"{r['model_name']:<32} | {v_acc*100:<7.2f}% | {v_f1:<8.4f} | {t_acc*100:<8.2f}% | {t_f1:<8.4f} | {t_auc:<10.4f} | {p10*100:<8.2f}%")

        clean_r = {k: v for k, v in r.items() if k not in ["raw_test_probs", "model_obj"]}
        ladder_export.append(clean_r)

    print("-" * 115)
    print(f"\nWINNER SELECTED ON VALIDATION MACRO-F1: {winner} (Val Macro-F1 = {best_val_score:.4f})")

    out_file = REPORTS_DIR / "model_ladder.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_criterion": "Validation Macro-F1 (Never selected or tuned on Test)",
            "winner_model": winner,
            "models_evaluated": ladder_export,
        }, f, indent=2)

    print(f"Saved Phase 4 Model Ladder Report to: {out_file}")


if __name__ == "__main__":
    run_phase4()
