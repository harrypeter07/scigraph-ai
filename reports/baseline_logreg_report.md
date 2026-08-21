# SciGraph AI — Logistic Regression Baseline-Anchored Report

```json
{
  "model_name": "LogisticRegression",
  "device": "cpu",
  "accuracy": 0.4,
  "macro_f1": 0.2929,
  "evaluation_summary": {
    "test_sample_count": 10,
    "correct_count": 4,
    "accuracy_fraction": "4/10",
    "accuracy": 0.4,
    "accuracy_percentage": "40.0%",
    "macro_f1": 0.2929,
    "confusion_matrix": [
      [
        3,
        1,
        0
      ],
      [
        2,
        1,
        0
      ],
      [
        2,
        1,
        0
      ]
    ],
    "per_class_metrics": {
      "class_0": {
        "class_name": "Low",
        "true_samples": 4,
        "precision": 0.4286,
        "recall": 0.75,
        "f1": 0.5455
      },
      "class_1": {
        "class_name": "Medium",
        "true_samples": 3,
        "precision": 0.3333,
        "recall": 0.3333,
        "f1": 0.3333
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
    "accuracy_delta": 0.0,
    "plain_language_verdict": "tied with baseline at 40.0%"
  },
  "predictions": [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
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
