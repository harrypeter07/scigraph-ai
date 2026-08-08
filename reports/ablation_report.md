# SciGraph AI — Phase 9 & 18 Temporal Leakage Ablation Report

```json
{
  "comparison": "HeteroGraphSAGE (Condition A: Time-Consistent) vs HeteroGraphSAGE (Condition B: Naive Random)",
  "device_used": "cpu",
  "time_consistent_temporal_split": {
    "model_architecture": "HeteroGraphSAGE",
    "random_seed": 42,
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
  "naive_random_split": {
    "model_architecture": "HeteroGraphSAGE",
    "random_seed": 42,
    "test_correct_count": 4,
    "test_total_count": 8,
    "accuracy_fraction": "4/8",
    "accuracy": 0.5,
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
  "empirical_finding": "Time-Consistent Accuracy: 3/5 (60.0%) vs Naive Random Accuracy: 4/8 (50.0%).",
  "small_sample_caveat": "NOTICE: At n=50 total papers (5 vs 8 test papers), this accuracy difference is within statistical noise and is not a conclusive proof of temporal leakage. Statistically significant validation requires dataset scale-up on GPU Colab (Part C)."
}
```
