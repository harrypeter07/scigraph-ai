"""Run global permutation importance across all 14 feature groups and save report."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.explain.attribution import compute_global_permutation_importance


def main():
    print("Computing global permutation importance (seed 42, 10 repeats per group)...")
    res = compute_global_permutation_importance(
        n_repeats=10,
        random_state=42,
        output_path=Path("reports/global_importance.json")
    )
    print("Global Importance Results:")
    for group_name, info in res["feature_groups"].items():
        print(f"  {info['display_name']:35s} | Mean drop: {info['mean_f1_drop']:.4f} | Norm: {info['normalized_importance']:.4f}")


if __name__ == "__main__":
    main()
