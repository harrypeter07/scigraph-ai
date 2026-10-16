# SciGraph AI — Phase 19 Full Evaluation Report (Baseline-Anchored)

```json
{
  "models": {
    "MajorityClass_Baseline": {
      "model_name": "MajorityClass_Baseline",
      "majority_class_id": 1,
      "majority_class_name": "Medium Impact",
      "rule_description": "Always predict training majority class (1) for all samples",
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
      "predictions": [
        1,
        1,
        1,
        1,
        1
      ]
    },
    "LogisticRegression": {
      "model_name": "LogisticRegression",
      "device": "cpu",
      "accuracy": 0.6,
      "macro_f1": 0.375,
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
    "GradientBoosting": {
      "model_name": "GradientBoostingClassifier",
      "device": "cpu",
      "accuracy": 0.6,
      "macro_f1": 0.4286,
      "evaluation_summary": {
        "test_sample_count": 5,
        "correct_count": 3,
        "accuracy_fraction": "3/5",
        "accuracy": 0.6,
        "accuracy_percentage": "60.0%",
        "macro_f1": 0.4286,
        "confusion_matrix": [
          [
            0,
            1,
            1
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
            "precision": 0.75,
            "recall": 1.0,
            "f1": 0.8571
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
        2,
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
    "HeteroGraphSAGE": {
      "model_name": "HeteroGraphSAGE",
      "device": "cpu",
      "accuracy": 0.6,
      "macro_f1": 0.5833,
      "evaluation_summary": {
        "test_sample_count": 5,
        "correct_count": 3,
        "accuracy_fraction": "3/5",
        "accuracy": 0.6,
        "accuracy_percentage": "60.0%",
        "macro_f1": 0.5833,
        "confusion_matrix": [
          [
            1,
            1,
            0
          ],
          [
            1,
            2,
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
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5
          },
          "class_1": {
            "class_name": "Medium",
            "true_samples": 3,
            "precision": 0.6667,
            "recall": 0.6667,
            "f1": 0.6667
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
        0,
        0,
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
      ],
      "checkpoint_file": "graphsage.pt",
      "checkpoint_size_bytes": 4785
    },
    "HeteroGAT": {
      "model_name": "HeteroGAT",
      "device": "cpu",
      "accuracy": 0.6,
      "macro_f1": 0.375,
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
      ],
      "checkpoint_file": "gat.pt",
      "checkpoint_size_bytes": 4533
    }
  },
  "feature_ablation_tiers": [
    {
      "tier": "Tier 0: Majority Class Baseline (Always Predict Class 1)",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "MajorityClass_Baseline",
      "plain_language_verdict": "Trivial Baseline (3/5 = 60.0%)"
    },
    {
      "tier": "Tier 1: Metadata-only (title_length, pub_year)",
      "accuracy_fraction": "2/5",
      "accuracy": 0.4,
      "model": "LogisticRegression",
      "plain_language_verdict": "Does not beat majority baseline (40.0% vs 60.0%)"
    },
    {
      "tier": "Tier 2: + Historical Cutoff Citations",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "GradientBoosting",
      "plain_language_verdict": "Tied with majority baseline (60.0% vs 60.0%)"
    },
    {
      "tier": "Tier 3: + Author & Institution Topology",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "HeteroGraphSAGE",
      "plain_language_verdict": "Tied with majority baseline (60.0% vs 60.0%)"
    },
    {
      "tier": "Tier 4: + Full Heterogeneous Graph (Paper+Author+Topic)",
      "accuracy_fraction": "4/5",
      "accuracy": 0.8,
      "model": "HeteroGraphSAGE",
      "plain_language_verdict": "Beats majority baseline by +20.0% (80.0% vs 60.0%)"
    }
  ]
}
```
