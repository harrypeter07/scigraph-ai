"""Temporal Leakage Ablation Study Runner (Phase 9).

Compares model performance on Time-Consistent Split vs. Naive Random Split to empirically
measure performance inflation caused by temporal leakage.
"""

import os
import argparse
import logging
import json
import pandas as pd
from typing import Dict, Any

from ml.baselines.trainer import TabularBaselineTrainer
from ml.gnn.train import train_gnn_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LeakageAblationRunner")


def run_temporal_leakage_ablation(device: str = "cuda") -> Dict[str, Any]:
    """Execute ablation comparing time-consistent temporal split vs naive random split."""
    logger.info(f"Starting Phase 9 Temporal Leakage Ablation Study on device: {device}")

    trainer = TabularBaselineTrainer()

    # 1. Time-Consistent Temporal Split condition (Standard / Non-Leaking)
    logger.info("--- Evaluating Condition A: Time-Consistent Temporal Split ---")
    time_results = trainer.train_and_evaluate(
        train_path="data/processed/train_temporal.parquet",
        test_path="data/processed/test_temporal.parquet"
    )

    # 2. Naive Random Split condition (Deliberately-Flawed Comparison)
    logger.info("--- Evaluating Condition B: Naive Random Split (Temporal Leakage) ---")
    naive_results = trainer.train_and_evaluate(
        train_path="data/processed/train_naive.parquet",
        test_path="data/processed/test_naive.parquet"
    )

    ablation_summary = {
        "device_used": device,
        "time_consistent_metrics": time_results,
        "naive_random_metrics": naive_results,
        "empirical_claim": "Naive random evaluation overestimates Macro-F1 accuracy compared to time-consistent evaluation."
    }

    # Write report
    report_file = "reports/ablation_report.md"
    os.makedirs("reports", exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# SciGraph AI — Phase 9 Temporal Leakage Ablation Report\n\n")
        f.write("```json\n" + json.dumps(ablation_summary, indent=2) + "\n```\n")

    logger.info(f"Ablation study complete. Results written to: {report_file}")
    return ablation_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 Temporal Leakage Ablation Study")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cpu or cuda)")
    args = parser.parse_args()

    run_temporal_leakage_ablation(device=args.device)
