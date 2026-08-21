"""Live Model Training & Evaluation Test Runner for SciGraph AI."""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from ml.baselines.trainer import TabularBaselineTrainer
from ml.gnn.train import train_gnn_model

def run():
    print("=" * 70)
    print("        SCIGRAPH AI — LIVE MODEL TRAINING & TEST SUITE RUNNER       ")
    print("=" * 70)
    print("Dataset Split: Train (Y <= 2018: 40 papers) | Test (Y >= 2020: 5 papers)\n")

    # 1. Traditional Baselines
    trainer = TabularBaselineTrainer()
    base_res = trainer.train_and_evaluate()

    print("\n" + "-" * 70)
    print("1. MAJORITY-CLASS BASELINE (Anchor Heuristic):")
    maj = base_res["MajorityClass_Baseline"]
    print(f"   Accuracy:     {maj['accuracy_fraction']} ({maj['accuracy_percentage']}) | Macro-F1: {maj['macro_f1']}")
    print(f"   Predictions:  {maj['predictions']}")
    print(f"   Ground Truth: [0, 1, 1, 0, 1]")

    print("\n" + "-" * 70)
    print("2. LOGISTIC REGRESSION (Linear Model):")
    lr = base_res["LogisticRegression"]
    print(f"   Accuracy:     {lr['evaluation_summary']['accuracy_fraction']} ({lr['evaluation_summary']['accuracy_percentage']}) | Macro-F1: {lr['evaluation_summary']['macro_f1']}")
    print(f"   Predictions:  {lr['predictions']}")
    print(f"   Ground Truth: {lr['ground_truth']}")
    print(f"   Verdict:      {lr['baseline_anchor']['plain_language_verdict']}")

    print("\n" + "-" * 70)
    print("3. GRADIENT BOOSTING GBDT (Non-Linear Decision Trees):")
    gb = base_res["GradientBoosting"]
    print(f"   Accuracy:     {gb['evaluation_summary']['accuracy_fraction']} ({gb['evaluation_summary']['accuracy_percentage']}) | Macro-F1: {gb['evaluation_summary']['macro_f1']}")
    print(f"   Predictions:  {gb['predictions']}")
    print(f"   Ground Truth: {gb['ground_truth']}")
    print(f"   Verdict:      {gb['baseline_anchor']['plain_language_verdict']}")

    print("\n" + "-" * 70)
    print("4. PYTORCH HETEROGRAPHSAGE (Graph Neural Network):")
    gnn = train_gnn_model("configs/gnn_graphsage.yaml", device="cpu")
    print(f"   Accuracy:     {gnn['evaluation_summary']['accuracy_fraction']} ({gnn['evaluation_summary']['accuracy_percentage']}) | Macro-F1: {gnn['evaluation_summary']['macro_f1']}")
    print(f"   Predictions:  {gnn['predictions']}")
    print(f"   Ground Truth: {gnn['ground_truth']}")
    print(f"   Checkpoint:   {gnn['checkpoint_file']} ({gnn['checkpoint_size_bytes']} bytes)")
    print(f"   Verdict:      {gnn['baseline_anchor']['plain_language_verdict']}")
    print("=" * 70)

if __name__ == "__main__":
    run()
