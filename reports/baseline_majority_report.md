# SciGraph AI — Majority Class Baseline Report

```json
{
  "test_correct_count": 4,
  "test_total_count": 10,
  "accuracy_fraction": "4/10",
  "accuracy": 0.4,
  "macro_f1": 0.1905,
  "per_class_metrics": {
    "class_0": {
      "sample_count": 4,
      "precision": 0.4,
      "recall": 1.0,
      "f1": 0.5714
    },
    "class_1": {
      "sample_count": 3,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0
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
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
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
  "model_name": "MajorityClass_Baseline",
  "most_frequent_training_class": 0,
  "note": "Predicts most frequent training class (Class 1, Medium) for all test samples."
}
```
