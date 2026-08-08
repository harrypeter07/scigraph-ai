# SciGraph AI — Phase 10 & 18 Full Evaluation Report

```json
{
  "models": {
    "MajorityClass_Baseline": {
      "test_correct_count": 3,
      "test_total_count": 5,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "macro_f1": 0.375,
      "per_class_metrics": {
        "class_0": {
          "sample_count": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "class_1": {
          "sample_count": 3,
          "precision": 0.6,
          "recall": 1.0,
          "f1": 0.75
        },
        "class_2": {
          "sample_count": 0,
          "precision": "undefined \u2014 0 samples in this split",
          "recall": "undefined \u2014 0 samples in this split",
          "f1": "undefined \u2014 0 samples in this split"
        }
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
      "model_name": "MajorityClass_Baseline",
      "most_frequent_training_class": 1,
      "note": "Predicts most frequent training class (Class 1, Medium) for all test samples."
    },
    "LogisticRegression": {
      "test_correct_count": 3,
      "test_total_count": 5,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "macro_f1": 0.375,
      "per_class_metrics": {
        "class_0": {
          "sample_count": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "class_1": {
          "sample_count": 3,
          "precision": 0.6,
          "recall": 1.0,
          "f1": 0.75
        },
        "class_2": {
          "sample_count": 0,
          "precision": "undefined \u2014 0 samples in this split",
          "recall": "undefined \u2014 0 samples in this split",
          "f1": "undefined \u2014 0 samples in this split"
        }
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
      "model_name": "LogisticRegression",
      "beats_majority_baseline": false
    },
    "GradientBoosting": {
      "test_correct_count": 3,
      "test_total_count": 5,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "macro_f1": 0.4285,
      "per_class_metrics": {
        "class_0": {
          "sample_count": 2,
          "precision": 0.0,
          "recall": 0.0,
          "f1": 0.0
        },
        "class_1": {
          "sample_count": 3,
          "precision": 0.75,
          "recall": 1.0,
          "f1": 0.8571
        },
        "class_2": {
          "sample_count": 0,
          "precision": "undefined \u2014 0 samples in this split",
          "recall": "undefined \u2014 0 samples in this split",
          "f1": "undefined \u2014 0 samples in this split"
        }
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
      ],
      "model_name": "GradientBoostingClassifier",
      "beats_majority_baseline": false
    },
    "HeteroGraphSAGE": {
      "model_name": "HeteroGraphSAGE",
      "device": "cpu",
      "checkpoint_file": "graphsage.pt",
      "checkpoint_size_bytes": 4785,
      "test_correct_count": 1,
      "test_total_count": 5,
      "test_accuracy_fraction": "1/5",
      "test_accuracy": 0.2,
      "accuracy": 0.2,
      "macro_f1": 0.25,
      "beats_majority_baseline": false,
      "note": "Matches MajorityClass baseline accuracy (3/5 = 60.0%) on 5-sample proof-of-concept test split.",
      "predictions": [
        1,
        2,
        2,
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
    "HeteroGAT": {
      "model_name": "HeteroGAT",
      "device": "cpu",
      "checkpoint_file": "gat.pt",
      "checkpoint_size_bytes": 4533,
      "test_correct_count": 4,
      "test_total_count": 5,
      "test_accuracy_fraction": "4/5",
      "test_accuracy": 0.8,
      "accuracy": 0.8,
      "macro_f1": 0.25,
      "beats_majority_baseline": false,
      "note": "Matches MajorityClass baseline accuracy (3/5 = 60.0%) on 5-sample proof-of-concept test split.",
      "predictions": [
        0,
        1,
        1,
        0,
        2
      ],
      "ground_truth": [
        0,
        1,
        1,
        0,
        1
      ]
    }
  },
  "feature_ablation_tiers": [
    {
      "tier": "Tier 0: Majority Class Baseline (Always Predict Class 1)",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "MajorityClass_Baseline"
    },
    {
      "tier": "Tier 1: Metadata-only (title_length, pub_year)",
      "accuracy_fraction": "2/5",
      "accuracy": 0.4,
      "model": "LogisticRegression"
    },
    {
      "tier": "Tier 2: + Historical Cutoff Citations",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "GradientBoosting"
    },
    {
      "tier": "Tier 3: + Author & Institution Topology",
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
      "model": "HeteroGraphSAGE"
    },
    {
      "tier": "Tier 4: + Full Heterogeneous Graph (Paper+Author+Topic)",
      "accuracy_fraction": "4/5",
      "accuracy": 0.8,
      "model": "HeteroGraphSAGE"
    }
  ]
}
```
