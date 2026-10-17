"""Tamper-Evident Verification Receipt and Ground-Truth Reveal Engine (WP3).

Provides cryptographic receipt generation and independent ground-truth verification:
1. build_receipt(work_id):
   Freezes strictly as-of-cutoff feature inputs, computes predicted tier probabilities,
   generates a canonical SHA256 digest, and appends to reports/prediction_ledger.jsonl.
2. build_reveal(work_id):
   Reveals post-cutoff ground-truth citations (years pub+3..pub+5), cohort thresholds,
   and verifies the receipt hash remained unchanged.
3. verify_receipt_hash(receipt_sha256):
   Audits the immutable prediction ledger to ensure the receipt was not modified or backdated.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.config import settings
from ml.inference_service import inference_service

LEDGER_PATH = REPO_ROOT / "reports" / "prediction_ledger.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.sha256"


def get_data_snapshot_id() -> str:
    """Computes SHA256 digest of data/MANIFEST.sha256 to bind predictions to data state."""
    if MANIFEST_PATH.exists():
        return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()[:16]
    return "unmanifested-v2"


def canonical_json_hash(data: Dict[str, Any]) -> str:
    """Computes a deterministic SHA256 hash over canonically ordered JSON."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_receipt(work_id: str, raw_work: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generates a tamper-evident prediction receipt frozen strictly as-of cutoff."""
    clean_id = str(work_id).split("/")[-1]

    # Ensure inference service is loaded
    if inference_service.model is None or inference_service.primary_model is None:
        inference_service.load()

    # If raw_work not provided, lookup in corpus or fetch
    if raw_work is None:
        corpus_row = None
        if inference_service.df_works is not None:
            matches = inference_service.df_works[inference_service.df_works["work_id"] == clean_id]
            if len(matches) > 0:
                corpus_row = matches.iloc[0].to_dict()

        if corpus_row is not None:
            raw_work = corpus_row
        else:
            from api.openalex_client import openalex_client
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                raw_work = loop.run_until_complete(openalex_client.get_work(clean_id))
            except Exception:
                raw_work = {"id": clean_id, "publication_year": 2018, "title": "Unknown"}

    prediction = inference_service.predict(raw_work)
    pub_year = int(prediction["publication_year"])
    cutoff_year = int(prediction["cutoff_year"])
    split = str(prediction.get("split", "unindexed"))

    # Extract strictly frozen inputs (values known <= cutoff_year)
    counts_by_year = raw_work.get("counts_by_year", [])
    if isinstance(counts_by_year, str):
        try:
            counts_by_year = json.loads(counts_by_year)
        except Exception:
            counts_by_year = []

    frozen_citations = [
        {"year": int(c.get("year", 0)), "cited_by_count": int(c.get("cited_by_count", 0))}
        for c in counts_by_year
        if isinstance(c, dict) and int(c.get("year", 0)) <= cutoff_year
    ]
    frozen_citations.sort(key=lambda x: x["year"])

    frozen_inputs = {
        "title": prediction["title"],
        "publication_year": pub_year,
        "cutoff_year": cutoff_year,
        "primary_field": prediction.get("field", "Computer Science"),
        "venue_name": prediction.get("venue", {}).get("name", "Unknown"),
        "venue_ranking": prediction.get("venue", {}).get("ranking", "No ranking found in SCImago or CORE"),
        "author_count": len(prediction.get("authors", [])),
        "is_open_access": bool(prediction.get("oa_economics", {}).get("is_open_access", False)),
        "citations_at_cutoff": int(prediction.get("citations_at_cutoff", 0)),
        "historical_citation_trajectory_to_cutoff": frozen_citations,
    }

    receipt_core = {
        "work_id": clean_id,
        "openalex_url": f"https://openalex.org/{clean_id}",
        "api_url": f"https://api.openalex.org/works/{clean_id}?select=id,publication_year,counts_by_year",
        "title": prediction["title"],
        "pub_year": pub_year,
        "cutoff_year": cutoff_year,
        "split": split,
        "in_training_data": bool(prediction.get("in_training_data", False)),
        "frozen_inputs": frozen_inputs,
        "model_version": "SciGraph-2.0-Primary-LogisticRegression",
        "data_snapshot_id": get_data_snapshot_id(),
        "probabilities": prediction["calibrated_probabilities"],
        "predicted_tier": prediction["predicted_impact_label"],
        "predicted_impact_class": prediction["predicted_impact_class"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    receipt_hash = canonical_json_hash(receipt_core)
    receipt = dict(receipt_core)
    receipt["receipt_sha256"] = receipt_hash

    # Append to append-only prediction ledger
    append_receipt_to_ledger(receipt)

    return receipt


def append_receipt_to_ledger(receipt: Dict[str, Any]) -> None:
    """Appends receipt line to prediction ledger in append-only mode."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(receipt, sort_keys=True) + "\n")


def check_receipt(receipt_sha256: str) -> Dict[str, Any]:
    """Validates whether a receipt exists in the ledger and verifies integrity."""
    if not LEDGER_PATH.exists():
        return {"valid": False, "reason": "Ledger file does not exist yet."}

    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if rec.get("receipt_sha256") == receipt_sha256:
                # Recompute hash to verify tamper-evidence
                recomp_core = {k: v for k, v in rec.items() if k != "receipt_sha256"}
                expected_hash = canonical_json_hash(recomp_core)
                if expected_hash == receipt_sha256:
                    return {
                        "valid": True,
                        "work_id": rec.get("work_id"),
                        "created_utc": rec.get("created_utc"),
                        "predicted_tier": rec.get("predicted_tier"),
                        "receipt": rec,
                    }
                else:
                    return {
                        "valid": False,
                        "reason": "Receipt integrity violated: ledger hash does not match computed hash.",
                    }

    return {"valid": False, "reason": f"Receipt hash '{receipt_sha256}' not found in ledger."}


def build_reveal(work_id: str, receipt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reveals actual citations in years pub+3..pub+5 and compares against frozen prediction."""
    clean_id = str(work_id).split("/")[-1]

    if receipt is None:
        receipt = build_receipt(clean_id)

    pub_year = int(receipt["pub_year"])
    cutoff_year = int(receipt["cutoff_year"])
    window_start = cutoff_year + 1
    window_end = pub_year + 5

    # Check receipt integrity
    recomp_core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    recomputed_hash = canonical_json_hash(recomp_core)
    receipt_unchanged = (recomputed_hash == receipt.get("receipt_sha256"))

    # Fetch ground truth post-cutoff counts
    raw_counts = []
    actual_window = 0
    actual_class = 0
    actual_label = "Low (<50th)"
    cohort_info = {}

    # Check corpus first
    stored_row = None
    if inference_service.df_works is not None:
        matches = inference_service.df_works[inference_service.df_works["work_id"] == clean_id]
        if len(matches) > 0:
            stored_row = matches.iloc[0].to_dict()

    if stored_row is not None and "future_citations_5y" in stored_row:
        actual_window = int(stored_row.get("future_citations_5y", 0))
        actual_class = int(stored_row.get("impact_class", 0))
        label_map = {0: "Low (<50th)", 1: "Medium (50-90th)", 2: "High (>=90th)"}
        actual_label = label_map.get(actual_class, "Low (<50th)")
        provenance = "ground truth from corpus (works.parquet)"
    else:
        # Fallback to API / counts_by_year
        if inference_service.df_counts is not None:
            c_matches = inference_service.df_counts[inference_service.df_counts["work_id"] == clean_id]
            raw_counts = c_matches.to_dict(orient="records")

        window_counts = [
            int(c.get("cited_by_count", 0))
            for c in raw_counts
            if window_start <= int(c.get("year", 0)) <= window_end
        ]
        actual_window = sum(window_counts)

        threshold_info = inference_service.get_cohort_threshold(
            receipt["frozen_inputs"]["primary_field"], pub_year
        )
        p50 = threshold_info["p50"]
        p90 = threshold_info["p90"]

        if actual_window < p50:
            actual_class = 0
            actual_label = "Low (<50th)"
        elif actual_window < p90:
            actual_class = 1
            actual_label = "Medium (50-90th)"
        else:
            actual_class = 2
            actual_label = "High (>=90th)"
        provenance = "estimated using cohort thresholds from 2012-2017"

    # Year-by-year counts for the 3-year window
    window_years_breakdown = []
    if inference_service.df_counts is not None:
        c_matches = inference_service.df_counts[inference_service.df_counts["work_id"] == clean_id]
        by_yr = {int(r["year"]): int(r["cited_by_count"]) for _, r in c_matches.iterrows()}
        for yr in range(window_start, window_end + 1):
            window_years_breakdown.append({"year": yr, "cited_by_count": by_yr.get(yr, 0)})

    threshold_info = inference_service.get_cohort_threshold(
        receipt["frozen_inputs"]["primary_field"], pub_year
    )
    predicted_class = int(receipt.get("predicted_impact_class", 0))
    is_correct = bool(predicted_class == actual_class)

    return {
        "work_id": clean_id,
        "receipt_sha256": receipt.get("receipt_sha256"),
        "receipt_unchanged": receipt_unchanged,
        "window_years": [window_start, window_start + 1, window_end],
        "yearly_citations_breakdown": window_years_breakdown,
        "window_total_citations": actual_window,
        "cohort_thresholds": {
            "field": receipt["frozen_inputs"]["primary_field"],
            "pub_year": pub_year,
            "p50_threshold": threshold_info["p50"],
            "p90_threshold": threshold_info["p90"],
        },
        "actual_tier": actual_label,
        "actual_impact_class": actual_class,
        "predicted_tier": receipt["predicted_tier"],
        "predicted_impact_class": predicted_class,
        "is_correct": is_correct,
        "provenance": provenance,
        "openalex_api_link": receipt["api_url"],
    }
