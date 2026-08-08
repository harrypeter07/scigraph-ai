# SciGraph AI — Gradient Boosting (GBDT) Baseline Report

```json
{
  "test_correct_count": 2,
  "test_total_count": 10,
  "accuracy_fraction": "2/10",
  "accuracy": 0.2,
  "macro_f1": 0.1667,
  "per_class_metrics": {
    "class_0": {
      "sample_count": 4,
      "precision": 0.25,
      "recall": 0.25,
      "f1": 0.25
    },
    "class_1": {
      "sample_count": 3,
      "precision": 0.2,
      "recall": 0.3333,
      "f1": 0.25
    },
    "class_2": {
      "sample_count": 3,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0
    }
  },
  "predictions": [
    0,
    1,
    1,
    1,
    0,
    1,
    2,
    0,
    0,
    1
  ],
  "ground_truth": [
    0,
    1,
    2,
    0,
    1,
    2,
    0,
    1,
    2,
    0
  ],
  "model_name": "GradientBoostingClassifier",
  "beats_majority_baseline": false
}
```
