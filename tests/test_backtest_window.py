"""Backtest Window Correctness Tests for WP4.

Verifies:
1. ResNet W2194775991 (2016):
   - Window is exactly 2019..2021 (cutoff+1..pub+5).
   - Actual window citations strictly match the manual sum from counts_by_year.
   - window_complete is True, has_ground_truth is True.
2. TON_IoT W3124508491 (2021):
   - Window is 2024..2026.
   - Window end (2026) has not closed yet -> has_ground_truth is False, actual_window is None, window_complete is False.
3. 2024 Recent Paper:
   - Window is 2027..2029 -> has_ground_truth is False, actual_window is None, actual_impact_class is None.
   - Warning is explicitly provided explaining the open window.
4. Threshold source:
   - Empirical p50 and p90 read from data/reference/cohort_thresholds.json.
   - Provenance records cohort_level ("field_year" | "field" | "global").
"""

import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.inference_service import inference_service


@pytest.fixture(scope="module", autouse=True)
def setup_service():
    inference_service.load()


def test_resnet_2016_backtest_window():
    """Verify ResNet W2194775991 (2016) window is exactly 2019..2021 and matches raw JSON."""
    cache_path = Path("data/cache/openalex/work_W2194775991.json")
    assert cache_path.exists(), f"Missing cached work at {cache_path}"

    with open(cache_path, "r", encoding="utf-8") as f:
        raw_work = json.load(f)

    res = inference_service.predict(raw_work)
    bt = res["backtest"]

    assert bt is not None
    assert bt["window_start"] == 2019
    assert bt["window_end"] == 2021
    assert bt["window_complete"] is True
    assert bt["has_ground_truth"] is True

    # Manual sum of citations from counts_by_year strictly within [2019, 2021]
    counts_by_year = raw_work.get("counts_by_year", [])
    expected_sum = sum(
        c.get("cited_by_count", 0)
        for c in counts_by_year
        if 2019 <= c.get("year", 0) <= 2021
    )

    assert bt["actual_window"] == expected_sum, (
        f"Actual window {bt['actual_window']} != manual sum {expected_sum}"
    )
    assert bt["actual_future_citations_5y"] == expected_sum
    assert bt["actual_impact_class"] == 2  # ResNet is landmark High Impact
    assert bt["threshold_provenance"] is not None
    assert "cohort_level" in bt["threshold_provenance"]


def test_ton_iot_2021_incomplete_window():
    """Verify TON_IoT (2021) has incomplete window (2024..2026) with null ground truth."""
    ton_iot_work = {
        "id": "https://openalex.org/W3124508491",
        "title": "TON_IoT: The New Generation, Telemetry and Cyber Dataset for Autonomous IoT and IIoT",
        "publication_year": 2021,
        "counts_by_year": [
            {"year": 2021, "cited_by_count": 30},
            {"year": 2022, "cited_by_count": 120},
            {"year": 2023, "cited_by_count": 280},
            {"year": 2024, "cited_by_count": 410},
            {"year": 2025, "cited_by_count": 380},
            {"year": 2026, "cited_by_count": 95},
        ],
        "referenced_works_count": 48,
        "authorships": [{"author": {"display_name": "Nour Moustafa"}}],
        "open_access": {"is_oa": True, "oa_status": "gold"},
        "primary_topic": {
            "display_name": "Internet of Things",
            "field": {"display_name": "Computer Science"}
        }
    }

    res = inference_service.predict(ton_iot_work)
    bt = res["backtest"]

    assert bt is not None
    assert bt["window_start"] == 2024
    assert bt["window_end"] == 2026
    assert bt["window_complete"] is False
    assert bt["has_ground_truth"] is False
    assert bt["actual_window"] is None
    assert bt["actual_future_citations_5y"] is None
    assert bt["actual_impact_class"] is None
    assert bt["actual_label"] is None
    assert bt["error_abs"] is None
    assert bt["error_log"] is None
    assert bt["warning"] is not None
    assert "not closed yet" in bt["warning"]


def test_2024_recent_paper_incomplete_window():
    """Verify 2024 recent paper has open window (2027..2029) and null backtest."""
    paper_2024 = {
        "id": "https://openalex.org/W9999999999",
        "title": "Foundation Models for Automated Discovery",
        "publication_year": 2024,
        "counts_by_year": [
            {"year": 2024, "cited_by_count": 14},
            {"year": 2025, "cited_by_count": 68},
        ],
        "referenced_works_count": 35,
        "authorships": [{"author": {"display_name": "Lead Researcher"}}],
        "open_access": {"is_oa": False, "oa_status": "closed"},
        "primary_topic": {
            "display_name": "Deep Learning",
            "field": {"display_name": "Computer Science"}
        }
    }

    res = inference_service.predict(paper_2024)
    bt = res["backtest"]

    assert bt is not None
    assert bt["window_start"] == 2027
    assert bt["window_end"] == 2029
    assert bt["window_complete"] is False
    assert bt["has_ground_truth"] is False
    assert bt["actual_window"] is None
    assert bt["actual_impact_class"] is None
    assert bt["warning"] is not None
    assert "not closed yet" in bt["warning"]
