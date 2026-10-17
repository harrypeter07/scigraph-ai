"""Explicit end-to-end training and evaluation script for SciGraph AI 2.0.

Executes all model training and evaluation explicitly without hidden auto-refitting:
1. Fits and calibrates primary model (M1_LogisticRegression) on clean 2012-2016 train split.
2. Exports held-out predictions and calibration metrics (reports/heldout_predictions.parquet).
3. Generates data manifest (data/MANIFEST.sha256).
4. Runs full benchmark suite across all models and saves reports/benchmark_latest.json.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def run_step(cmd: list, desc: str):
    print("\n" + "=" * 70)
    print(f"STEP: {desc}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 70)
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"ERROR: Step failed with return code {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("=" * 70)
    print("SciGraph AI 2.0 — Explicit Full Training & Benchmark Suite")
    print("=" * 70)

    # 1. Generate Data Manifest
    run_step([sys.executable, "scripts/generate_manifest.py"], "Generate data/MANIFEST.sha256")

    # 2. Fit Primary Served Model (LogisticRegression)
    run_step([sys.executable, "ml/evaluation/fit_primary_model.py"], "Fit & Calibrate Primary Model (M1_LogisticRegression)")

    # 3. Export Held-Out Test Predictions
    run_step([sys.executable, "ml/evaluation/export_heldout.py"], "Export Held-Out Predictions & Summary")

    # 4. Master Benchmark Suite (M0..M5 + B1, B2)
    run_step([sys.executable, "ml/evaluation/run_all.py", "--seeds", "0"], "Master Benchmark Suite (Seed 0)")

    print("\n" + "=" * 70)
    print("SUCCESS: All models trained, calibrated, and benchmarked explicitly.")
    print("=" * 70)

if __name__ == "__main__":
    main()
