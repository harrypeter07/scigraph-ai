"""FastAPI Router for SciGraph AI Verification Endpoints.

Provides transparent endpoints to verify:
1. Overall summary metrics and calibration on the held-out test split.
2. Concrete examples of correct vs misaligned predictions.
3. Blind forecast testing vs ground-truth reveal.
4. Temporal poison test (zero leakage proof).
5. Leakage audit status from reports/leakage_audit.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.config import settings
from api.openalex_client import openalex_client, OpenAlexNotFoundError, OpenAlexValidationError
from ml.inference_service import inference_service

logger = logging.getLogger("scigraph.api.verify")

router = APIRouter(tags=["Verification & Proof"])

REPORTS_DIR = settings.BASE_DIR / "reports"


@router.get("/summary")
def get_verification_summary():
    """Return held-out test split evaluation summary, lift, confusion matrix, and calibration."""
    summary_path = REPORTS_DIR / "heldout_predictions_summary.json"
    if not summary_path.exists():
        # Fallback to computing on the fly if export script hasn't run
        from ml.evaluation.export_heldout import export_heldout_predictions
        return export_heldout_predictions()

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"Failed to read heldout predictions summary: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to read verification summary: {exc}")


@router.get("/examples")
def get_verification_examples(
    split: str = Query("test", description="Data split: test or val"),
    n: int = Query(10, ge=1, le=50, description="Number of examples"),
    outcome: str = Query("all", description="Outcome filter: correct, wrong, or all")
):
    """Return genuine held-out prediction examples categorized by correctness."""
    parquet_path = REPORTS_DIR / "heldout_predictions.parquet"
    if not parquet_path.exists():
        from ml.evaluation.export_heldout import export_heldout_predictions
        export_heldout_predictions()

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load heldout predictions parquet: {exc}")

    # Filter split
    df_filtered = df[df["split"] == split]
    if df_filtered.empty:
        df_filtered = df

    # Filter outcome
    if outcome == "correct":
        df_filtered = df_filtered[df_filtered["pred_class"] == df_filtered["actual_class"]]
    elif outcome == "wrong":
        df_filtered = df_filtered[df_filtered["pred_class"] != df_filtered["actual_class"]]

    sample_size = min(n, len(df_filtered))
    if sample_size == 0:
        return {"total_matching": 0, "examples": []}

    sample_df = df_filtered.sample(n=sample_size, random_state=42)
    records = sample_df.to_dict(orient="records")

    # Clean records
    cleaned_records = []
    for r in records:
        cleaned_records.append({
            "work_id": str(r.get("work_id", "")).split("/")[-1],
            "title": str(r.get("title", "Untitled")),
            "publication_year": int(r.get("pub_year", 2018)),
            "field": str(r.get("primary_field_name", "General")),
            "split": str(r.get("split", "test")),
            "citations_at_cutoff": int(r.get("citations_at_cutoff", 0)),
            "actual_window": int(r.get("actual_window", 0)),
            "actual_percentile": float(r.get("actual_percentile", 50.0)),
            "pred_class": int(r.get("pred_class", 0)),
            "pred_label": str(r.get("pred_label", "")),
            "actual_class": int(r.get("actual_class", 0)),
            "actual_label": str(r.get("actual_label", "")),
            "is_correct": bool(r.get("is_correct", False)),
            "p_low": round(float(r.get("p_low", 0.0)), 4),
            "p_med": round(float(r.get("p_med", 0.0)), 4),
            "p_high": round(float(r.get("p_high", 0.0)), 4),
        })

    return {
        "split": split,
        "outcome": outcome,
        "total_matching": len(df_filtered),
        "count": len(cleaned_records),
        "examples": cleaned_records
    }


@router.get("/blind/{work_id}")
async def get_blind_forecast(work_id: str):
    """Return paper prediction with ground truth post-cutoff outcomes strictly hidden."""
    clean_id = openalex_client.clean_identifier(work_id)
    try:
        raw_work = await openalex_client.get_work(clean_id)
    except OpenAlexNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work '{clean_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    full_pred = inference_service.predict(raw_work)

    # Hide actual outcome in blind mode
    blind_response = dict(full_pred)
    blind_response["backtest"] = {
        "has_ground_truth": False,
        "window_complete": False,
        "window_start": full_pred["cutoff_year"] + 1,
        "window_end": full_pred["publication_year"] + settings.TRAJECTORY_HORIZON_YEARS,
        "actual_window": None,
        "actual_future_citations_5y": None,
        "actual_impact_class": None,
        "actual_label": None,
        "predicted_impact_class": full_pred["predicted_impact_class"],
        "predicted_label": full_pred["predicted_impact_label"],
        "is_correct_class": None,
        "warning": "Blind Forecast Mode: Actual citations after T_cutoff are locked until you hit Reveal.",
        "provenance": "model | blind_evaluation"
    }

    # Hide post-cutoff trajectory points
    if "trajectory" in blind_response:
        blind_response["trajectory"] = [
            pt for pt in blind_response["trajectory"]
            if pt.get("year", 0) <= full_pred["cutoff_year"]
        ]

    return blind_response


@router.get("/blind/{work_id}/reveal")
async def reveal_blind_forecast(work_id: str):
    """Reveal the actual ground-truth future citations and backtest outcome."""
    clean_id = openalex_client.clean_identifier(work_id)
    try:
        raw_work = await openalex_client.get_work(clean_id)
    except OpenAlexNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work '{clean_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return inference_service.predict(raw_work)


@router.get("/poison/{work_id}")
async def test_poison_invariance(work_id: str):
    """Verify temporal leakage immunity: inject fake future citations and assert zero prediction delta."""
    clean_id = openalex_client.clean_identifier(work_id)
    try:
        raw_work = await openalex_client.get_work(clean_id)
    except OpenAlexNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work '{clean_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 1. Clean prediction
    clean_pred = inference_service.predict(raw_work)

    # 2. Poisoned work: inject massive citation spike 10 years in the future
    import copy
    poisoned_work = copy.deepcopy(raw_work)
    fut_yr = clean_pred["publication_year"] + 10
    counts = poisoned_work.get("counts_by_year", [])
    counts.append({"year": fut_yr, "cited_by_count": 100000})
    poisoned_work["counts_by_year"] = counts
    poisoned_work["cited_by_count"] = int(poisoned_work.get("cited_by_count", 0)) + 100000

    poisoned_pred = inference_service.predict(poisoned_work)

    # Compare probabilities and class
    clean_probs = clean_pred["calibrated_probabilities"]
    poison_probs = poisoned_pred["calibrated_probabilities"]
    max_prob_diff = max(abs(clean_probs[k] - poison_probs[k]) for k in clean_probs)

    is_invariant = (max_prob_diff < 1e-5) and (clean_pred["predicted_impact_class"] == poisoned_pred["predicted_impact_class"])

    return {
        "work_id": clean_id,
        "title": clean_pred["title"],
        "cutoff_year": clean_pred["cutoff_year"],
        "poison_injected_year": fut_yr,
        "poison_citations_added": 100000,
        "clean_probabilities": clean_probs,
        "poisoned_probabilities": poison_probs,
        "max_probability_delta": float(max_prob_diff),
        "temporal_firewall_intact": is_invariant,
        "verdict": "PASSED: Zero Temporal Leakage (Strict Cutoff Firewall Verified)" if is_invariant else "FAILED: Leakage Detected"
    }


@router.get("/audits")
def get_leakage_audit_report():
    """Return full temporal leakage audit report from reports/leakage_audit.json."""
    audit_path = REPORTS_DIR / "leakage_audit.json"
    if not audit_path.exists():
        raise HTTPException(status_code=503, detail="Leakage audit report not generated. Run scripts/audit_leakage.py.")

    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read leakage audit: {exc}")
