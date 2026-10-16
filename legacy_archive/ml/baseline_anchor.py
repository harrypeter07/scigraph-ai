"""Honest Baseline-Anchored Evaluation Layer (Phase 19 Part B).

Guarantees that every model metric (Accuracy, Macro-F1, Confusion Matrix) is structurally
paired with a Majority-Class Baseline computed on the exact same split, along with an explicit
plain-language verdict.
"""

import json
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_majority_class_baseline(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Compute performance of trivial Majority-Class Baseline ('Always predict most frequent training class')."""
    train_classes, counts = np.unique(y_train, return_counts=True)
    majority_class = int(train_classes[np.argmax(counts)])
    
    preds_majority = np.full(len(y_test), majority_class, dtype=int)
    correct_count = int((preds_majority == y_test).sum())
    total_count = len(y_test)
    accuracy = float(correct_count / total_count) if total_count > 0 else 0.0

    p, r, f1, _ = precision_recall_fscore_support(y_test, preds_majority, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(np.mean([f for f, count in zip(f1, [np.sum(y_test == c) for c in [0, 1, 2]]) if count > 0]))

    cm = confusion_matrix(y_test, preds_majority, labels=[0, 1, 2]).tolist()

    return {
        "model_name": "MajorityClass_Baseline",
        "majority_class_id": majority_class,
        "majority_class_name": {0: "Low Impact", 1: "Medium Impact", 2: "High Impact"}.get(majority_class, "Medium"),
        "rule_description": f"Always predict training majority class ({majority_class}) for all samples",
        "test_sample_count": total_count,
        "correct_count": correct_count,
        "accuracy_fraction": f"{correct_count}/{total_count}",
        "accuracy": round(accuracy, 4),
        "accuracy_percentage": f"{accuracy * 100:.1f}%",
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "predictions": preds_majority.tolist()
    }


def evaluate_model_with_baseline_anchor(
    model_name: str,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluate model independently, compute exact confusion matrix & Macro-F1,
    and anchor with side-by-side Majority-Class Baseline and plain-language verdict."""
    y_pred = np.array(y_pred, dtype=int)
    y_test = np.array(y_test, dtype=int)
    y_train = np.array(y_train, dtype=int)

    total_count = len(y_test)
    correct_count = int((y_pred == y_test).sum())
    accuracy = float(correct_count / total_count) if total_count > 0 else 0.0

    # 1. Independent Per-Model Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()

    # 2. Per-class metrics handling absent classes explicitly
    per_class = {}
    f1_list = []
    for c in [0, 1, 2]:
        n_true = int((y_test == c).sum())
        if n_true == 0:
            per_class[f"class_{c}"] = {
                "class_name": {0: "Low", 1: "Medium", 2: "High"}.get(c),
                "true_samples": 0,
                "precision": "undefined — 0 samples in this split",
                "recall": "undefined — 0 samples in this split",
                "f1": "undefined — 0 samples in this split"
            }
        else:
            tp = int(((y_pred == c) & (y_test == c)).sum())
            fp = int(((y_pred == c) & (y_test != c)).sum())
            fn = int(((y_pred != c) & (y_test == c)).sum())

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

            per_class[f"class_{c}"] = {
                "class_name": {0: "Low", 1: "Medium", 2: "High"}.get(c),
                "true_samples": n_true,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4)
            }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # 3. Compute Majority Baseline on identical split
    baseline_stats = compute_majority_class_baseline(y_train, y_test)
    base_acc = baseline_stats["accuracy"]

    # 4. Generate Plain-Language Verdict
    diff = round(accuracy - base_acc, 4)
    if total_count < 10:
        if diff > 0:
            verdict = f"beats baseline by +{diff*100:.1f}% (Notice: sample size n={total_count} is small)"
        elif diff == 0:
            verdict = f"tied with majority baseline at {accuracy*100:.1f}% (n={total_count} sample test split)"
        else:
            verdict = f"does not beat baseline ({accuracy*100:.1f}% vs {base_acc*100:.1f}%)"
    else:
        if diff > 0:
            verdict = f"beats baseline by +{diff*100:.1f}%"
        elif diff == 0:
            verdict = f"tied with baseline at {accuracy*100:.1f}%"
        else:
            verdict = f"does not yet beat baseline ({accuracy*100:.1f}% vs {base_acc*100:.1f}%)"

    return {
        "model_name": model_name,
        "device": device,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "evaluation_summary": {
            "test_sample_count": total_count,
            "correct_count": correct_count,
            "accuracy_fraction": f"{correct_count}/{total_count}",
            "accuracy": round(accuracy, 4),
            "accuracy_percentage": f"{accuracy * 100:.1f}%",
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm,
            "per_class_metrics": per_class
        },
        "baseline_anchor": {
            "baseline_model": "MajorityClass_Baseline",
            "baseline_accuracy": base_acc,
            "baseline_accuracy_percentage": f"{base_acc * 100:.1f}%",
            "baseline_macro_f1": baseline_stats["macro_f1"],
            "accuracy_delta": diff,
            "plain_language_verdict": verdict
        },
        "predictions": y_pred.tolist(),
        "ground_truth": y_test.tolist()
    }
