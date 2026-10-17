#!/usr/bin/env bash
set -e
echo "======================================================================"
echo "SciGraph AI 2.0 — Explicit Full Training & Benchmark Suite"
echo "======================================================================"

python scripts/generate_manifest.py
python ml/evaluation/fit_primary_model.py
python ml/evaluation/export_heldout.py
python ml/evaluation/run_all.py --seeds 0

echo "Explicit training and evaluation complete."
