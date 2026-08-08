"""Phase 8 Unit Test - GNN CPU Smoke Test.

Verifies GNN trainer initialization and execution on CPU in under a few seconds.
"""

import pytest
import os
from ml.gnn.train import train_gnn_model


def test_gnn_cpu_smoke_test(tmp_path):
    """Assert GNN trainer completes a CPU execution pass cleanly."""
    config_path = "configs/gnn_graphsage.yaml"
    assert os.path.exists(config_path), f"Missing GNN config file: {config_path}"

    metrics = train_gnn_model(config_path=config_path, device="cpu")
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
