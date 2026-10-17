"""Comprehensive Test Suite for Audit Fixes, Verification Endpoints, and Fraud Remediation.

Verifies:
1. Served model is NOT constant (predictions on test papers span >= 2 classes, Macro-F1 >= 0.37).
2. No fake authors ("Author 1", "Author 2", etc.) in any API response.
3. Corpus sample endpoint returns only held-out test split papers (split="test", in_training_data=False).
4. No forbidden hardcoded fabricated numbers in web/index.html or README.md.
5. Verification endpoints (/verify/summary, /verify/examples, /verify/poison) function correctly.
"""

import re
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from ml.inference_service import inference_service

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_served_model_not_constant(client):
    """Verify served model predictions are not constant and achieve Macro-F1 >= 0.37 on heldout test papers."""
    test_parquet = REPO_ROOT / "data" / "processed" / "test.parquet"
    assert test_parquet.exists(), "data/processed/test.parquet missing"
    df_test = pd.read_parquet(test_parquet)

    # Sample 15 test papers per class (45 total) for balanced, non-degenerate verification
    sample = df_test.groupby("impact_class", group_keys=False).apply(
        lambda g: g.sample(min(15, len(g)), random_state=42)
    )
    preds = []
    actuals = []

    for _, row in sample.iterrows():
        raw_work = row.to_dict()
        res = inference_service.predict(raw_work)
        preds.append(res["predicted_impact_class"])
        actuals.append(int(row["impact_class"]))

    unique_preds = set(preds)
    assert len(unique_preds) >= 2, f"Served model predicted a single constant class: {unique_preds}"

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(actuals, preds, average="macro")
    assert macro_f1 >= 0.37, f"Served model Macro-F1 too low ({macro_f1:.4f} < 0.37) — model might be degenerate"


def test_no_fake_authors(client):
    """Verify no 'Author 1', 'Author 2', etc. appear in sample or prediction responses."""
    res = client.get("/api/v1/papers/corpus-sample?limit=10")
    assert res.status_code == 200
    samples = res.json()["sample"]

    for s in samples:
        pid = s["paper_id"]
        pred_res = client.get(f"/api/v1/papers/predict/{pid}")
        assert pred_res.status_code == 200
        data = pred_res.json()
        authors = data.get("authors", [])
        for a in authors:
            assert not re.match(r"^Author\s+\d+$", a.strip(), re.IGNORECASE), (
                f"Fabricated author string detected: '{a}' for paper {pid}"
            )


def test_corpus_sample_test_split_only(client):
    """Verify /corpus-sample serves papers strictly from the held-out test split."""
    res = client.get("/api/v1/papers/corpus-sample?limit=15")
    assert res.status_code == 200
    data = res.json()
    assert "sample" in data
    assert len(data["sample"]) == 15

    for item in data["sample"]:
        assert item.get("split") == "test", f"Paper {item['paper_id']} is not test split: {item.get('split')}"
        assert item.get("in_training_data") is False, f"Paper {item['paper_id']} flagged as in_training_data=True"
        assert item.get("publication_year") >= 2018, f"Paper {item['paper_id']} published before test split start 2018"


def test_no_hardcoded_numbers():
    """Verify no forbidden fabricated numbers or claims remain in web/index.html or README.md."""
    forbidden = [
        "0.5381", "19.3%", "0.0083", "0.5284", "16/16",
        "9,000 Papers", "3,000 Papers", "Zero Temporal Leakage",
        "Independently Verifiable", "world's first", "70% of papers"
    ]

    html_path = REPO_ROOT / "web" / "index.html"
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
        for num in forbidden:
            assert num not in html_content, f"Found forbidden fabricated string '{num}' in web/index.html"

    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text(encoding="utf-8")
        for num in forbidden:
            assert num not in readme_content, f"Found forbidden fabricated string '{num}' in README.md"


def test_verification_summary_endpoint(client):
    """Verify /api/v1/verify/summary returns test evaluation metrics."""
    res = client.get("/api/v1/verify/summary")
    assert res.status_code == 200
    data = res.json()
    assert "macro_f1" in data or "test_macro_f1" in data or "test_metrics" in data


def test_verification_examples_endpoint(client):
    """Verify /api/v1/verify/examples returns held-out examples."""
    res = client.get("/api/v1/verify/examples?split=test&n=5&outcome=all")
    assert res.status_code == 200
    data = res.json()
    assert "examples" in data
    assert len(data["examples"]) == 5
    for ex in data["examples"]:
        assert "is_correct" in ex
        assert "pred_class" in ex
        assert "actual_class" in ex


def test_verification_poison_endpoint(client):
    """Verify /api/v1/verify/poison/{work_id} confirms zero temporal leakage."""
    sample_res = client.get("/api/v1/papers/corpus-sample?limit=1")
    sample_id = sample_res.json()["sample"][0]["paper_id"]

    res = client.get(f"/api/v1/verify/poison/{sample_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["temporal_firewall_intact"] is True
    assert data["max_probability_delta"] < 1e-4


def test_verification_receipt_and_reveal_flow(client):
    """Test cryptographic receipt generation, ledger persistence, and reveal flow."""
    sample_res = client.get("/api/v1/papers/corpus-sample?limit=1")
    sample_id = sample_res.json()["sample"][0]["paper_id"]

    # 1. Generate receipt
    rec_res = client.post(f"/api/v1/verify/receipt/{sample_id}")
    assert rec_res.status_code == 200
    receipt = rec_res.json()
    assert "receipt_sha256" in receipt
    assert receipt["work_id"] == sample_id
    receipt_hash = receipt["receipt_sha256"]

    # 2. Check receipt against ledger
    check_res = client.get(f"/api/v1/verify/receipt/{receipt_hash}/check")
    assert check_res.status_code == 200
    check_data = check_res.json()
    assert check_data["valid"] is True
    assert check_data["work_id"] == sample_id

    # 3. Reveal ground truth
    reveal_res = client.get(f"/api/v1/verify/reveal/{sample_id}")
    assert reveal_res.status_code == 200
    reveal = reveal_res.json()
    assert reveal["receipt_unchanged"] is True
    assert "window_total_citations" in reveal
    assert "actual_tier" in reveal
    assert "is_correct" in reveal


def test_leakage_controls_endpoint(client):
    """Test /api/v1/verify/leakage-controls dual controls (negative poison + positive sensitivity)."""
    sample_res = client.get("/api/v1/papers/corpus-sample?limit=1")
    sample_id = sample_res.json()["sample"][0]["paper_id"]

    res = client.get(f"/api/v1/verify/leakage-controls/{sample_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["negative_control"]["passed"] is True
    assert data["positive_control"]["passed"] is True
    assert data["firewall_integrity"] is True


def test_heldout_csv_endpoint(client):
    """Test /api/v1/verify/heldout.csv downloads test split evaluation predictions."""
    res = client.get("/api/v1/verify/heldout.csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    text = res.text
    lines = text.strip().split("\n")
    assert len(lines) >= 100
    header = lines[0].split(",")
    assert "work_id" in header
    assert "actual_class" in header
    assert "pred_class" in header
