# SciGraph AI — Phase 19 Temporal Leakage Ablation Report

```json
{
  "comparison": "HeteroGraphSAGE (Condition A: Time-Consistent) vs HeteroGraphSAGE (Condition B: Naive Random)",
  "device_used": "cpu",
  "time_consistent_temporal_split": {
    "model_name": "HeteroGraphSAGE (Time-Consistent)",
    "device": "cpu",
    "evaluation_summary": {
      "test_sample_count": 5,
      "correct_count": 3,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "accuracy_percentage": "60.0%",
      "macro_f1": 0.375,
      "confusion_matrix": [
        [
          0,
          2,
          0
        ],
        [
          0,
          3,
          0
        ],
        [
          0,
          0,
          0
        ]
      ],
      "per_class_metrics": {
        "class_0": {
          "class_name": "Low",
          "true_samples": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "class_1": {
          "class_name": "Medium",
          "true_samples": 3,
          "precision": 0.6,
          "recall": 1.0,
          "f1": 0.75
        },
        "class_2": {
          "class_name": "High",
          "true_samples": 0,
          "precision": "undefined \u2014 0 samples in this split",
          "recall": "undefined \u2014 0 samples in this split",
          "f1": "undefined \u2014 0 samples in this split"
        }
      }
    },
    "baseline_anchor": {
      "baseline_model": "MajorityClass_Baseline",
      "baseline_accuracy": 0.6,
      "baseline_accuracy_percentage": "60.0%",
      "baseline_macro_f1": 0.375,
      "accuracy_delta": 0.0,
      "plain_language_verdict": "tied with majority baseline at 60.0% (n=5 sample test split)"
    },
    "predictions": [
      1,
      1,
      1,
      1,
      1
    ],
    "ground_truth": [
      0,
      1,
      1,
      0,
      1
    ]
  },
  "naive_random_split": {
    "model_name": "HeteroGraphSAGE (Naive-Random)",
    "device": "cpu",
    "evaluation_summary": {
      "test_sample_count": 8,
      "correct_count": 4,
      "accuracy_fraction": "4/8",
      "accuracy": 0.5,
      "accuracy_percentage": "50.0%",
      "macro_f1": 0.2222,
      "confusion_matrix": [
        [
          0,
          2,
          0
        ],
        [
          0,
          4,
          0
        ],
        [
          0,
          2,
          0
        ]
      ],
      "per_class_metrics": {
        "class_0": {
          "class_name": "Low",
          "true_samples": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "class_1": {
          "class_name": "Medium",
          "true_samples": 4,
          "precision": 0.5,
          "recall": 1.0,
          "f1": 0.6667
        },
        "class_2": {
          "class_name": "High",
          "true_samples": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        }
      }
    },
    "baseline_anchor": {
      "baseline_model": "MajorityClass_Baseline",
      "baseline_accuracy": 0.5,
      "baseline_accuracy_percentage": "50.0%",
      "baseline_macro_f1": 0.2222,
      "accuracy_delta": 0.0,
      "plain_language_verdict": "tied with majority baseline at 50.0% (n=8 sample test split)"
    },
    "predictions": [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    "ground_truth": [
      0,
      1,
      2,
      2,
      0,
      1,
      1,
      1
    ]
  },
  "empirical_finding": "Time-Consistent Accuracy: 3/5 vs Naive Random Accuracy: 4/8.",
  "small_sample_caveat": "NOTICE: At n=50 total papers (5 vs 8 test papers), the accuracy difference (60.0% vs 50.0%) is within statistical noise and is not a conclusive proof of temporal leakage. Statistically significant validation requires dataset scale-up on GPU Colab."
}
```
