# SciGraph AI — Gradient Boosting (GBDT) Baseline-Anchored Report

```json
{
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
}
```
