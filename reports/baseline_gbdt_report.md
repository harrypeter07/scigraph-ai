# SciGraph AI — Gradient Boosting (GBDT) Baseline-Anchored Report

```json
{
  "model_name": "GradientBoostingClassifier",
  "device": "cpu",
  "accuracy": 0.2,
  "macro_f1": 0.1667,
  "evaluation_summary": {
    "test_sample_count": 10,
    "correct_count": 2,
    "accuracy_fraction": "2/10",
    "accuracy": 0.2,
    "accuracy_percentage": "20.0%",
    "macro_f1": 0.1667,
    "confusion_matrix": [
      [
        1,
        2,
        1
      ],
      [
        2,
        1,
        0
      ],
      [
        1,
        2,
        0
      ]
    ],
    "per_class_metrics": {
      "class_0": {
        "class_name": "Low",
        "true_samples": 4,
        "precision": 0.25,
        "recall": 0.25,
        "f1": 0.25
      },
      "class_1": {
        "class_name": "Medium",
        "true_samples": 3,
        "precision": 0.2,
        "recall": 0.3333,
        "f1": 0.25
      },
      "class_2": {
        "class_name": "High",
        "true_samples": 3,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
      }
    }
  },
  "baseline_anchor": {
    "baseline_model": "MajorityClass_Baseline",
    "baseline_accuracy": 0.4,
    "baseline_accuracy_percentage": "40.0%",
    "baseline_macro_f1": 0.1905,
    "accuracy_delta": -0.2,
    "plain_language_verdict": "does not yet beat baseline (20.0% vs 40.0%)"
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
  ]
}
```
