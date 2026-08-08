# SciGraph AI — Phase 10 Full Evaluation Report

```json
{
  "models": {
    "LogisticRegression": {
      "model_name": "LogisticRegression",
      "test_correct_count": 3,
      "test_total_count": 5,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
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
      "test_correct_count": 3,
      "test_total_count": 5,
      "accuracy_fraction": "3/5",
      "accuracy": 0.6,
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
      "checkpoint_file": "graphsage.pt",
      "checkpoint_size_bytes": 4785,
      "test_correct_count": 3,
      "test_total_count": 5,
      "test_accuracy_fraction": "3/5",
      "test_accuracy": 0.6,
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
    "HeteroGAT": {
      "model_name": "HeteroGraphSAGE",
      "device": "cpu",
      "checkpoint_file": "gat.pt",
      "checkpoint_size_bytes": 4533,
      "test_correct_count": 3,
      "test_total_count": 5,
      "test_accuracy_fraction": "3/5",
      "test_accuracy": 0.6,
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
    }
  },
  "feature_ablation_tiers": [
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
