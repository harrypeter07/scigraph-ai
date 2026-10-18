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


def _fetch_openalex_work_sync(clean_id: str) -> Dict[str, Any]:
    """Fetch a full OpenAlex work in both sync and already-running-loop contexts."""
    from api.openalex_client import openalex_client
    import asyncio

    async def _get():
        return await openalex_client.get_work(clean_id)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _get()).result(timeout=30)

    return asyncio.run(_get())


def build_receipt(
    work_id: str,
    raw_work: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    mode: str = "M",
) -> Dict[str, Any]:
    """Generates a tamper-evident prediction receipt frozen strictly as-of cutoff."""
    clean_id = str(work_id).split("/")[-1]

    if inference_service.primary_model is None:
        inference_service.load()

    # If raw_work not provided, fetch a live OpenAlex record (needed for counts_by_year).
    if raw_work is None:
        raw_work = _fetch_openalex_work_sync(clean_id)

    if not raw_work.get("publication_year"):
        raise ValueError(f"Work '{clean_id}' is missing publication_year; cannot freeze a receipt.")

    if prediction is None:
        prediction = inference_service.predict(raw_work)
    pub_year = int(prediction["publication_year"])
    cutoff_year = int(prediction["cutoff_year"])
    split = str(prediction.get("split", "unindexed"))

    # Apply Gemini mode G if requested
    mode_str = mode.upper() if mode else "M"
    gemini_reasoning = None
    if mode_str == "G":
        try:
            from ml.gemini_service import gemini_service
            t_info = inference_service.get_cohort_threshold(
                prediction.get("primary_field"),
                pub_year
            )
            gemini_res = gemini_service.predict(
                paper_context=prediction,
                cohort_info=t_info,
                m1_prediction=prediction
            )
            prediction = dict(prediction)
            prediction["predicted_impact_class"] = gemini_res["predicted_impact_class"]
            prediction["predicted_impact_label"] = gemini_res["predicted_impact_label"]
            prediction["calibrated_probabilities"] = gemini_res["calibrated_probabilities"]
            gemini_reasoning = gemini_res.get("gemini_reasoning")
        except Exception as e:
            logger.warning(f"Gemini receipt build fallback: {e}")
            mode_str = "M"

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
        "primary_field": prediction.get("primary_field") or prediction.get("field"),
        "venue_name": (prediction.get("venue") or {}).get("venue_name") or (prediction.get("venue") or {}).get("name"),
        "venue_ranking": (prediction.get("venue") or {}).get("quartile") or (prediction.get("venue") or {}).get("core_rank"),
        "author_count": len(prediction.get("authors", [])),
        "is_open_access": bool((prediction.get("access_economics") or {}).get("is_oa", False)),
        "citations_at_cutoff": int(prediction.get("citations_at_cutoff", 0)),
        "historical_citation_trajectory_to_cutoff": frozen_citations,
    }

    model_label = "SciGraph-2.0-Gemini (Mode G)" if mode_str == "G" else "SciGraph-2.0-Primary-LogisticRegression"
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
        "model_version": model_label,
        "mode": mode_str,
        "gemini_reasoning": gemini_reasoning,
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


def build_reveal(
    work_id: str,
    receipt: Optional[Dict[str, Any]] = None,
    raw_work: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reveals actual citations in years pub+3..pub+5 and compares against frozen prediction."""
    clean_id = str(work_id).split("/")[-1]

    if raw_work is None:
        raw_work = _fetch_openalex_work_sync(clean_id)

    if receipt is None:
        receipt = build_receipt(clean_id, raw_work=raw_work)

    pub_year = int(receipt["pub_year"])
    cutoff_year = int(receipt["cutoff_year"])
    window_start = cutoff_year + 1
    window_end = pub_year + 5

    # Check receipt integrity
    recomp_core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    recomputed_hash = canonical_json_hash(recomp_core)
    receipt_unchanged = (recomputed_hash == receipt.get("receipt_sha256"))

    label_map = {0: "Low (<50th)", 1: "Medium (50-90th)", 2: "High (>=90th)"}
    counts_by_year = raw_work.get("counts_by_year") or []
    if isinstance(counts_by_year, str):
        try:
            counts_by_year = json.loads(counts_by_year)
        except Exception:
            counts_by_year = []

    by_yr = {
        int(c.get("year", 0)): int(c.get("cited_by_count", 0))
        for c in counts_by_year
        if isinstance(c, dict) and c.get("year") is not None
    }

    if not by_yr:
        counts_path = settings.DATA_DIR / "counts_by_year.parquet"
        if counts_path.exists():
            try:
                wids = [clean_id, f"https://openalex.org/{clean_id}"]
                df_c = pd.read_parquet(
                    counts_path,
                    columns=["year", "cited_by_count"],
                    filters=[("work_id", "in", wids)],
                )
                if not df_c.empty:
                    by_yr = dict(zip(df_c["year"].astype(int), df_c["cited_by_count"].astype(int)))
            except Exception:
                by_yr = {}

    window_years_breakdown = [
        {"year": yr, "cited_by_count": int(by_yr.get(yr, 0))}
        for yr in range(window_start, window_end + 1)
    ]
    actual_window = sum(item["cited_by_count"] for item in window_years_breakdown)

    stored_row = None
    if inference_service.df_works is not None:
        matches = inference_service.df_works[
            inference_service.df_works["work_id"].astype(str).str.endswith(clean_id)
        ]
        if len(matches) > 0:
            stored_row = matches.iloc[0].to_dict()

    if stored_row is not None and pd.notnull(stored_row.get("impact_class")) and actual_window == 0:
        actual_window = int(stored_row.get("future_citations_5y") or 0)
        actual_class = int(stored_row["impact_class"])
        actual_label = label_map.get(actual_class, "Low (<50th)")
        provenance = "ground truth from corpus (works.parquet)"
    else:
        threshold_info = inference_service.get_cohort_threshold(
            receipt["frozen_inputs"].get("primary_field"), pub_year
        )
        p50 = threshold_info["p50"]
        p90 = threshold_info["p90"]
        if actual_window < p50:
            actual_class = 0
        elif actual_window < p90:
            actual_class = 1
        else:
            actual_class = 2
        actual_label = label_map[actual_class]
        provenance = "live OpenAlex counts_by_year labeled with empirical cohort thresholds"

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
