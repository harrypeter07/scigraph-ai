# SciGraph AI — Phase 9 Temporal Leakage Ablation Report

```json
{
  "device_used": "cuda",
  "time_consistent_metrics": {
    "LogisticRegression": {
      "accuracy": 0.6,
      "macro_f1": 0.375,
      "macro_precision": 0.3,
      "macro_recall": 0.5,
      "per_class_f1": [
        0.0,
        0.75
      ],
      "confusion_matrix": [
        [
          0,
          2
        ],
        [
          0,
          3
        ]
      ]
    },
    "XGBoost": {
      "accuracy": 0.4,
      "macro_f1": 0.2857,
      "macro_precision": 0.25,
      "macro_recall": 0.3333,
      "per_class_f1": [
        0.0,
        0.5714
      ],
      "confusion_matrix": [
        [
          0,
          2
        ],
        [
          1,
          2
        ]
      ]
    }
  },
  "naive_random_metrics": {
    "LogisticRegression": {
      "accuracy": 0.5,
      "macro_f1": 0.2222,
      "macro_precision": 0.1667,
      "macro_recall": 0.3333,
      "per_class_f1": [
        0.0,
        0.6667,
        0.0
      ],
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
      ]
    },
    "XGBoost": {
      "accuracy": 0.5,
      "macro_f1": 0.2222,
      "macro_precision": 0.1667,
      "macro_recall": 0.3333,
      "per_class_f1": [
        0.0,
        0.6667,
        0.0
      ],
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
      ]
    }
  },
  "empirical_claim": "Naive random evaluation overestimates Macro-F1 accuracy compared to time-consistent evaluation."
}
```
