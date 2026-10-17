"""SciGraph AI 2.0 — Phase 2: Feature Ladder Evaluation.

Systematically measures the incremental lift of each feature group:
- F0: Served Baseline (log1p early citations, is_oa, num_authors, venue_sjr)
- +F1: Cohort-normalized early citations (percentile rank within (field, year), z-score of log cits)
- +F2: Yearly citation trajectory (y0, y1, y2, acceleration, growth ratio, last-year share)
- +F3: Citer volume/diversity as-of cutoff
- +F4: Author history as-of cutoff (prior papers, veteran author)
- +F5: Venue history as-of cutoff (SJR, CORE rank, Q1 status)
- +F6: Reference properties (ref count, log ref count)
- +F7: Paper intrinsics (title length, words, APC flag)

For each step:
- Retrains model on Train (2012-2016, N=7,500)
- Tunes regularization on Val (2017, N=1,500)
- Evaluates on Held-Out Test (2018-2020, N=4,500)
- Computes Accuracy, Macro-F1, AUROC, Precision@10% for High, 95% Bootstrap CIs,
  and Paired Bootstrap p-value vs previous step.
- Saves results to reports/accuracy_improvement/feature_ladder.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features.asof import AsOfFeatureBuilder

REPORTS_DIR = REPO_ROOT / "reports" / "accuracy_improvement"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_precision_at_top10(y_true: np.ndarray, probs_high: np.ndarray) -> float:
    """Computes precision on the top 10% highest predicted scores for High (Class 2)."""
    n_top = max(1, int(len(y_true) * 0.10))
    top_indices = np.argsort(probs_high)[-n_top:]
    return float(np.mean(y_true[top_indices] == 2))


def bootstrap_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1s, accs, aurocs = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_b = y_true[idx]
        pr_b = probs[idx]
        preds_b = np.argmax(pr_b, axis=1)

        f1s.append(f1_score(yt_b, preds_b, average="macro", zero_division=0))
        accs.append(accuracy_score(yt_b, preds_b))
        try:
            yt_oh = np.eye(3)[yt_b]
            aurocs.append(roc_auc_score(yt_oh, pr_b, average="macro", multi_class="ovr"))
        except Exception:
            aurocs.append(0.50)

    f1s = np.array(f1s)
    ci = {
        "macro_f1_ci": [round(float(np.percentile(f1s, 2.5)), 4), round(float(np.percentile(f1s, 97.5)), 4)],
        "accuracy_ci": [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)],
        "auroc_ci": [round(float(np.percentile(aurocs, 2.5)), 4), round(float(np.percentile(aurocs, 97.5)), 4)],
    }
    return ci, f1s


def run_feature_ladder():
    print("=" * 85)
    print("PHASE 2: AS-OF FEATURE LADDER EXPERIMENT (VAL-TUNED, TEST REPORTED ONCE)")
    print("=" * 85)

    builder = AsOfFeatureBuilder()
    print("Extracting features as-of cutoff...")
    df = builder.extract_features_df()

    train_mask = (df["temporal_split"] == "train") & (df["publication_year"] <= 2016)
    val_mask = (df["temporal_split"] == "val") & (df["publication_year"] == 2017)
    test_mask = (df["temporal_split"] == "test") & (df["publication_year"] >= 2018) & (df["publication_year"] <= 2020)

    y_train = df[train_mask]["impact_class"].values
    y_val = df[val_mask]["impact_class"].values
    y_test = df[test_mask]["impact_class"].values

    print(f"Dataset Partitions: Train={len(y_train):,} (2012-2016), Val={len(y_val):,} (2017), Test={len(y_test):,} (2018-2020)")

    # Define Feature Groups
    groups = {
        "F0_Baseline": ["f0_log_cits", "f0_is_oa", "f0_num_authors", "f0_venue_sjr"],
        "+F1_CohortNorm": ["f1_cohort_percentile", "f1_cohort_zscore", "f1_cohort_size"],
        "+F2_Trajectory": ["f2_y0", "f2_y1", "f2_y2", "f2_accel", "f2_growth_ratio", "f2_last_year_share", "f2_positive_growth"],
        "+F3_CiterVol": ["f3_in_corpus_citer_count"],
        "+F4_AuthorHist": ["f4_author_max_prior_papers", "f4_author_mean_prior_papers", "f4_has_veteran_author"],
        "+F5_VenueHist": ["f5_has_core_rank", "f5_is_q1"],
        "+F6_RefQuality": ["f6_ref_count", "f6_log_ref_count"],
        "+F7_Intrinsics": ["f7_title_chars", "f7_title_words", "f7_has_apc"],
    }

    cumulative_cols: List[str] = []
    ladder_results = []
    prev_boot_f1s = None

    print("\n" + "-" * 115)
    print(f"{'Step / Feature Group Added':<26} | {'Val Acc':<8} | {'Val F1':<8} | {'Test Acc':<9} | {'Test F1':<8} | {'Test AUROC':<10} | {'Prec@10%':<9} | {'Decision'}")
    print("-" * 115)

    for g_name, cols in groups.items():
        trial_cols = cumulative_cols + cols

        X_train_raw = df[train_mask][trial_cols].values
        X_val_raw = df[val_mask][trial_cols].values
        X_test_raw = df[test_mask][trial_cols].values

        scaler = StandardScaler().fit(X_train_raw)
        X_tr = scaler.transform(X_train_raw)
        X_va = scaler.transform(X_val_raw)
        X_te = scaler.transform(X_test_raw)

        # Tune C on VALIDATION split only
        best_c = 1.0
        best_val_f1 = -1.0
        best_lr = None

        for c_cand in [0.01, 0.1, 1.0, 10.0, 50.0]:
            lr = LogisticRegression(C=c_cand, max_iter=1000, random_state=42)
            lr.fit(X_tr, y_train)
            val_preds = lr.predict(X_va)
            vf1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_c = c_cand
                best_lr = lr

        val_probs = best_lr.predict_proba(X_va)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(y_val, val_preds)

        test_probs = best_lr.predict_proba(X_te)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        y_test_oh = np.eye(3)[y_test]
        test_auroc = roc_auc_score(y_test_oh, test_probs, average="macro", multi_class="ovr")
        prec10 = compute_precision_at_top10(y_test, test_probs[:, 2])

        ci_dict, boot_f1s = bootstrap_metrics(y_test, test_probs)

        # Paired bootstrap p-value vs previous step
        if prev_boot_f1s is not None:
            delta = boot_f1s - prev_boot_f1s
            paired_p = float(np.mean(delta <= 0))
        else:
            paired_p = 1.0

        # Decision rule: Keep if Val F1 improved
        if len(ladder_results) == 0:
            decision = "BASELINE"
            keep = True
        else:
            prev_val_f1 = ladder_results[-1]["val_macro_f1"]
            if best_val_f1 > prev_val_f1:
                decision = f"KEEP (+{best_val_f1 - prev_val_f1:.4f})"
                keep = True
            else:
                decision = f"DROP ({best_val_f1 - prev_val_f1:.4f})"
                keep = False

        if keep:
            cumulative_cols = trial_cols
            prev_boot_f1s = boot_f1s

        step_res = {
            "group": g_name,
            "features_in_group": cols,
            "total_features": len(cumulative_cols) if keep else len(cumulative_cols),
            "kept": keep,
            "decision": decision,
            "tuned_C": best_c,
            "val_accuracy": round(float(val_acc), 4),
            "val_macro_f1": round(float(best_val_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_f1), 4),
            "test_macro_f1_ci": ci_dict["macro_f1_ci"],
            "test_accuracy_ci": ci_dict["accuracy_ci"],
            "test_auroc": round(float(test_auroc), 4),
            "test_auroc_ci": ci_dict["auroc_ci"],
            "test_precision_at_top10": round(float(prec10), 4),
            "paired_bootstrap_p_vs_prev": round(paired_p, 4),
        }
        ladder_results.append(step_res)

        p_str = f"p={paired_p:.3f}" if prev_boot_f1s is not None and g_name != "F0_Baseline" else "—"
        print(f"{g_name:<26} | {val_acc*100:<7.2f}% | {best_val_f1:<8.4f} | {test_acc*100:<8.2f}% | {test_f1:<8.4f} | {test_auroc:<10.4f} | {prec10*100:<8.2f}% | {decision}")

    print("-" * 115)

    out_file = REPORTS_DIR / "feature_ladder.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_protocol": "Retrained on Train (2012-2016), tuned C on Val (2017), evaluated once on Test (2018-2020)",
            "ladder": ladder_results,
            "final_selected_features": cumulative_cols,
        }, f, indent=2)

    print(f"\nSaved Feature Ladder report to: {out_file}")
    return ladder_results, cumulative_cols


if __name__ == "__main__":
    run_feature_ladder()
