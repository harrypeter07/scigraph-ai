"""SciGraph AI 2.0 — High-Scale Model Training Script (Version 2).

Trains the Version 2 (V2) Model using the As-Of Feature Ladder:
1. Strict Temporal Cutoff (T_cutoff = pub_year + 2).
2. Trajectory Dynamics (y0, y1, y2, velocity, acceleration, growth ratio).
3. Field Cohort Normalization (ratio to discipline median).
4. Balanced Class Weighting & Multi-Class Calibration.
5. Multi-Task Evaluation (3-Tier, Binary Median, Elite Top-10% Discovery).
6. Exports bundle to ml/gnn/checkpoints/primary_model_v2.joblib.
"""

import os
import sys
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINTS_DIR = REPO_ROOT / "ml" / "gnn" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(df: pd.DataFrame, text_enc=None, is_train=False):
    """Extracts the 32-dim As-Of Feature Ladder with zero data leakage."""
    features = []
    for _, row in df.iterrows():
        c_cutoff = float(row.get("citations_at_cutoff", 0))
        log_cits = np.log1p(max(0.0, c_cutoff))

        traj = row.get("historical_citation_trajectory_to_cutoff", [])
        y0 = float(traj[0].get("cited_by_count", 0)) if len(traj) > 0 else 0.0
        y1 = float(traj[1].get("cited_by_count", 0)) if len(traj) > 1 else y0
        y2 = float(traj[2].get("cited_by_count", 0)) if len(traj) > 2 else y1

        v1 = y1 - y0
        v2 = y2 - y1
        acceleration = v2 - v1
        growth_ratio = (y2 + 1.0) / (y0 + 1.0)
        last_year_share = (y2 - y1) / (c_cutoff + 1.0)

        field_cohort_p50 = float(row.get("cohort_p50", 12.0) or 12.0)
        ratio_to_median = (c_cutoff + 1.0) / (field_cohort_p50 + 1.0)

        is_oa = 1.0 if row.get("is_open_access") else 0.0
        authors = min(30.0, float(row.get("author_count", 1) or 1)) / 10.0
        sjr = float(row.get("sjr", 0.0) or 0.0)
        is_top_venue = 1.0 if (sjr > 1.5 or str(row.get("venue_ranking", "")).startswith("A")) else 0.0

        feat_vec = [
            log_cits, np.log1p(y0), np.log1p(y1), np.log1p(y2),
            v1, v2, acceleration, growth_ratio, last_year_share,
            ratio_to_median, is_oa, authors, sjr, is_top_venue
        ]
        features.append(feat_vec)

    X_num = np.array(features, dtype=np.float32)

    titles = df["title"].fillna("Untitled Paper").tolist()
    if is_train or text_enc is None:
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        svd = TruncatedSVD(n_components=18, random_state=42)
        X_tfidf = tfidf.fit_transform(titles)
        X_text = svd.fit_transform(X_tfidf)
        text_enc = (tfidf, svd)
    else:
        tfidf, svd = text_enc
        X_tfidf = tfidf.transform(titles)
        X_text = svd.transform(X_tfidf)

    X_full = np.hstack([X_num, X_text])
    return X_full, text_enc


def run_training_v2():
    print("=" * 75)
    print("SCIGRAPH AI — TRAINING VERSION 2 (V2) HIGH-SCALE MODEL")
    print("=" * 75)

    data_dir = REPO_ROOT / "data" / "processed"
    df_train = pd.read_parquet(data_dir / "train.parquet")
    df_val = pd.read_parquet(data_dir / "val.parquet")
    df_test = pd.read_parquet(data_dir / "test.parquet")

    print(f"Dataset Loaded: Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

    X_train, text_enc = extract_features(df_train, is_train=True)
    X_val, _ = extract_features(df_val, text_enc=text_enc)
    X_test, _ = extract_features(df_test, text_enc=text_enc)

    y_train = df_train["impact_class"].to_numpy(dtype=np.int64)
    y_val = df_val["impact_class"].to_numpy(dtype=np.int64)
    y_test = df_test["impact_class"].to_numpy(dtype=np.int64)

    try:
        import lightgbm as lgb
        clf_v2 = lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        clf_v2.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf_v2 = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            class_weight="balanced",
            random_state=42
        )
        clf_v2.fit(X_train, y_train)

    # Predictions
    preds_3 = clf_v2.predict(X_test)
    probs_3 = clf_v2.predict_proba(X_test)

    acc_3 = accuracy_score(y_test, preds_3)
    bal_acc_3 = balanced_accuracy_score(y_test, preds_3)
    macro_f1_3 = f1_score(y_test, preds_3, average="macro")

    # Binary Above vs Below Median
    y_test_bin = (y_test > 0).astype(int)
    prob_above_median = 1.0 - probs_3[:, 0]
    preds_bin = (prob_above_median >= 0.50).astype(int)
    acc_bin = accuracy_score(y_test_bin, preds_bin)
    auc_bin = roc_auc_score(y_test_bin, prob_above_median)

    # Elite Top 10%
    y_test_top10 = (y_test == 2).astype(int)
    prob_top10 = probs_3[:, 2]
    auc_top10 = roc_auc_score(y_test_top10, prob_top10)
    n_top10 = int(len(y_test_top10) * 0.10)
    top10_idx = np.argsort(prob_top10)[-n_top10:]
    prec_top10 = float(np.mean(y_test_top10[top10_idx] == 1))

    print("\n" + "=" * 75)
    print("                  FINAL V2 EVALUATION SCOREBOARD                  ")
    print("=" * 75)
    print(f"Task 1: 3-Tier Test Accuracy:       {acc_3 * 100:.2f}%")
    print(f"Task 1: Balanced Accuracy:         {bal_acc_3 * 100:.2f}%")
    print(f"Task 1: Macro F1-Score:             {macro_f1_3:.4f}")
    print("-" * 75)
    print(f"Task 2: Binary Median Accuracy:     {acc_bin * 100:.2f}% (Above vs Below Median)")
    print(f"Task 2: Binary Median AUROC:        {auc_bin:.4f}")
    print("-" * 75)
    print(f"Task 3: Elite Top-10% AUROC:        {auc_top10:.4f}")
    print(f"Task 3: Precision @ Top 10%:        {prec_top10 * 100:.2f}% ({prec_top10/0.10:.1f}x Lift)")
    print("=" * 75)

    # Save checkpoints as V2 (preserving V1!)
    v2_payload = {
        "model": clf_v2,
        "text_encoder": text_enc,
        "version": "2.0.0-high-scale",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {
            "acc_3": acc_3,
            "bal_acc_3": bal_acc_3,
            "macro_f1_3": macro_f1_3,
            "acc_bin": acc_bin,
            "auc_bin": auc_bin,
            "auc_top10": auc_top10,
            "prec_top10": prec_top10
        }
    }
    ckpt_out = CHECKPOINTS_DIR / "primary_model_v2.joblib"
    joblib.dump(v2_payload, ckpt_out)
    print(f"\n[OK] Saved Version 2 model to: {ckpt_out}")


if __name__ == "__main__":
    run_training_v2()
