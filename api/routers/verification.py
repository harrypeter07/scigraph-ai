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
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response

from api.config import settings
from api.openalex_client import openalex_client, OpenAlexNotFoundError, OpenAlexValidationError
from ml.inference_service import inference_service
from ml.verification.receipt import build_receipt, build_reveal, check_receipt

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


async def _load_live_work(work_id: str) -> tuple[str, dict]:
    clean_id = openalex_client.clean_identifier(work_id)
    try:
        raw_work = await openalex_client.get_work(clean_id)
        return clean_id, raw_work
    except OpenAlexNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work '{clean_id}' not found.")
    except OpenAlexValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch work: {exc}")


@router.post("/receipt/{work_id}")
async def create_verification_receipt(work_id: str):
    """Generate a tamper-evident prediction receipt frozen strictly as-of cutoff."""
    clean_id, raw_work = await _load_live_work(work_id)
    try:
        return build_receipt(clean_id, raw_work=raw_work)
    except Exception as exc:
        logger.error(f"Failed to build receipt for {clean_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/reveal/{work_id}")
async def reveal_ground_truth(work_id: str):
    """Reveal post-cutoff ground truth citations (years pub+3..pub+5) and compare against receipt."""
    clean_id, raw_work = await _load_live_work(work_id)
    try:
        receipt = build_receipt(clean_id, raw_work=raw_work)
        return build_reveal(clean_id, receipt=receipt, raw_work=raw_work)
    except Exception as exc:
        logger.error(f"Failed to reveal ground truth for {clean_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/compare/{work_id}")
async def compare_predicted_and_actual(work_id: str):
    """Live model prediction and observed outcome in one payload for teacher verification."""
    clean_id, raw_work = await _load_live_work(work_id)
    try:
        prediction = inference_service.predict(raw_work)
        receipt = build_receipt(clean_id, raw_work=raw_work, prediction=prediction)
        reveal = build_reveal(clean_id, receipt=receipt, raw_work=raw_work)
        bt = prediction.get("backtest") or {}
        return {
            "work_id": clean_id,
            "title": prediction.get("title"),
            "publication_year": prediction.get("publication_year"),
            "primary_field": prediction.get("primary_field"),
            "venue": prediction.get("venue"),
            "cutoff_year": prediction.get("cutoff_year"),
            "split": prediction.get("split"),
            "in_training_data": prediction.get("in_training_data"),
            "prediction": prediction,
            "receipt": receipt,
            "reveal": reveal,
            "side_by_side": {
                "citations_at_cutoff": prediction.get("citations_at_cutoff"),
                "early_trajectory": receipt.get("frozen_inputs", {}).get(
                    "historical_citation_trajectory_to_cutoff", []
                ),
                "predicted_tier": prediction.get("predicted_impact_label"),
                "predicted_class": prediction.get("predicted_impact_class"),
                "probabilities": prediction.get("calibrated_probabilities"),
                "predicted_window_citations": prediction.get("predicted_future_citations_5y"),
                "actual_tier": reveal.get("actual_tier") or bt.get("actual_label"),
                "actual_class": reveal.get("actual_impact_class"),
                "actual_window_citations": reveal.get("window_total_citations"),
                "yearly_actual": reveal.get("yearly_citations_breakdown"),
                "cohort_thresholds": reveal.get("cohort_thresholds"),
                "is_correct": reveal.get("is_correct"),
                "receipt_sha256": receipt.get("receipt_sha256"),
            },
        }
    except RuntimeError as r_err:
        raise HTTPException(status_code=503, detail=str(r_err))
    except Exception as exc:
        logger.error(f"Failed live compare for {clean_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/receipt/{receipt_hash}/check")
def verify_receipt_hash_endpoint(receipt_hash: str):
    """Verify cryptographic receipt against the immutable prediction ledger."""
    res = check_receipt(receipt_hash)
    if not res.get("valid"):
        raise HTTPException(status_code=404, detail=res.get("reason", "Invalid receipt hash."))
    return res


@router.get("/heldout.csv")
def download_heldout_csv():
    """Download held-out test split evaluation predictions as a CSV."""
    parquet_path = REPORTS_DIR / "heldout_predictions.parquet"
    if not parquet_path.exists():
        from ml.evaluation.export_heldout import export_heldout_predictions
        export_heldout_predictions()

    try:
        df = pd.read_parquet(parquet_path)
        df_test = df[df["split"] == "test"]
        csv_str = df_test.to_csv(index=False)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=heldout_test_predictions.csv"}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate heldout CSV: {exc}")


@router.get("/spotcheck")
async def spotcheck_live_openalex(n: int = Query(20, ge=1, le=50)):
    """Picks n random heldout test papers and queries api.openalex.org LIVE in parallel to verify ground truth."""
    import asyncio
    import httpx
    from scripts.verify_sample import load_cohort_threshold

    test_path = settings.DATA_DIR / "processed" / "test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=500, detail="data/processed/test.parquet missing")

    df_test = pd.read_parquet(test_path)
    sample = df_test.sample(n=min(n, len(df_test)), random_state=random.randint(1, 10000))

    headers = {"User-Agent": f"SciGraphVerification/2.0 (mailto:{settings.OPENALEX_POLITE_EMAIL})"}

    async def fetch_paper_live(client: httpx.AsyncClient, wid: str):
        url = f"https://api.openalex.org/works/{wid}?select=id,title,publication_year,primary_topic,counts_by_year,cited_by_count&mailto={settings.OPENALEX_POLITE_EMAIL}"
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    wids = [str(r["work_id"]).split("/")[-1] for _, r in sample.iterrows()]
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_paper_live(client, wid) for wid in wids]
        live_data_list = await asyncio.gather(*tasks)

    results = []
    matches = 0
    valid_count = 0

    for (_, row), wid, live_data in zip(sample.iterrows(), wids, live_data_list):
        stored_pub_year = int(row["publication_year"])
        stored_field = row.get("primary_field_name", "Computer Science")
        stored_window = int(row.get("future_citations_5y", 0))
        stored_class = int(row.get("impact_class", 0))

        if not live_data:
            results.append({
                "work_id": wid,
                "title": str(row.get("title", "Untitled"))[:40],
                "publication_year": stored_pub_year,
                "stored_window": stored_window,
                "live_window": None,
                "stored_class": stored_class,
                "live_class": None,
                "live_label": None,
                "tier_match": False,
                "openalex_api_url": f"https://api.openalex.org/works/{wid}?select=id,publication_year,counts_by_year",
                "status": "API_TIMEOUT_OR_NETWORK_ERROR"
            })
            continue

        counts_by_year = live_data.get("counts_by_year", [])
        by_year_dict = {
            int(c.get("year", 0)): int(c.get("cited_by_count", 0))
            for c in counts_by_year
            if isinstance(c, dict)
        }

        cutoff_year = stored_pub_year + 2
        window_counts = [by_year_dict.get(yr, 0) for yr in range(cutoff_year + 1, stored_pub_year + 6)]
        live_window = sum(window_counts)

        thresholds = load_cohort_threshold(stored_field, stored_pub_year)
        p50 = thresholds["p50"]
        p90 = thresholds["p90"]

        if live_window < p50:
            live_class = 0
            live_label = "Low (<50th)"
        elif live_window < p90:
            live_class = 1
            live_label = "Medium (50-90th)"
        else:
            live_class = 2
            live_label = "High (>=90th)"

        tier_match = (stored_class == live_class)
        valid_count += 1
        if tier_match:
            matches += 1

        results.append({
            "work_id": wid,
            "title": str(row.get("title", "Untitled"))[:40],
            "publication_year": stored_pub_year,
            "stored_window": stored_window,
            "live_window": live_window,
            "stored_class": stored_class,
            "live_class": live_class,
            "live_label": live_label,
            "tier_match": tier_match,
            "openalex_api_url": f"https://api.openalex.org/works/{wid}?select=id,publication_year,counts_by_year",
            "status": "VERIFIED_MATCH" if tier_match else "CITATION_DRIFT"
        })

    match_rate = (matches / valid_count * 100.0) if valid_count > 0 else 0.0
    return {
        "n_requested": n,
        "n_audited": len(results),
        "valid_count": valid_count,
        "matches": matches,
        "match_rate_percent": round(match_rate, 2),
        "provenance": "Live external query to api.openalex.org",
        "sample_results": results
    }


@router.get("/leakage-controls/{work_id}")
async def run_leakage_controls(work_id: str):
    """Run dual leakage controls (negative future-poison + positive pre-cutoff sensitivity)."""
    clean_id = openalex_client.clean_identifier(work_id)
    try:
        raw_work = await openalex_client.get_work(clean_id)
    except OpenAlexNotFoundError:
        raise HTTPException(status_code=404, detail=f"Work '{clean_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    import copy

    # 1. Baseline prediction (clean)
    clean_pred = inference_service.predict(raw_work)
    clean_probs = clean_pred["calibrated_probabilities"]
    clean_cits_cutoff = clean_pred["citations_at_cutoff"]
    pub_year = clean_pred["publication_year"]
    cutoff_year = clean_pred["cutoff_year"]

    # 2. Negative Control: Inject +100,000 citations 10 years in future
    neg_work = copy.deepcopy(raw_work)
    fut_yr = pub_year + 10
    counts_neg = list(neg_work.get("counts_by_year", []))
    counts_neg.append({"year": fut_yr, "cited_by_count": 100000})
    neg_work["counts_by_year"] = counts_neg
    neg_work["cited_by_count"] = int(neg_work.get("cited_by_count", 0)) + 100000

    neg_pred = inference_service.predict(neg_work)
    neg_probs = neg_pred["calibrated_probabilities"]
    neg_prob_delta = max(abs(clean_probs[k] - neg_probs[k]) for k in clean_probs)
    neg_cits_delta = abs(neg_pred["citations_at_cutoff"] - clean_cits_cutoff)
    negative_passed = bool(neg_prob_delta < 1e-5 and neg_cits_delta == 0)

    # 3. Positive Control: Inject +50 citations in cutoff_year (valid historical year)
    pos_work = copy.deepcopy(raw_work)
    counts_pos = list(pos_work.get("counts_by_year", []))
    found_yr = False
    for c in counts_pos:
        if c.get("year") == cutoff_year:
            c["cited_by_count"] = int(c.get("cited_by_count", 0)) + 50
            found_yr = True
            break
    if not found_yr:
        counts_pos.append({"year": cutoff_year, "cited_by_count": 50})
    pos_work["counts_by_year"] = counts_pos
    pos_work["cited_by_count"] = int(pos_work.get("cited_by_count", 0)) + 50

    pos_pred = inference_service.predict(pos_work)
    pos_probs = pos_pred["calibrated_probabilities"]
    pos_prob_delta = max(abs(clean_probs[k] - pos_probs[k]) for k in clean_probs)
    pos_cits_delta = abs(pos_pred["citations_at_cutoff"] - clean_cits_cutoff)
    positive_passed = bool(pos_cits_delta > 0 and pos_prob_delta > 1e-4)

    return {
        "work_id": clean_id,
        "title": clean_pred["title"],
        "publication_year": pub_year,
        "cutoff_year": cutoff_year,
        "baseline": {
            "citations_at_cutoff": clean_cits_cutoff,
            "probabilities": clean_probs,
            "predicted_class": clean_pred["predicted_impact_class"],
            "predicted_label": clean_pred["predicted_impact_label"],
        },
        "negative_control": {
            "description": "Injected +100,000 citations to year pub+10 (post-cutoff future). Features and probabilities must be identical.",
            "citations_added": 100000,
            "target_year": fut_yr,
            "result_citations_at_cutoff": neg_pred["citations_at_cutoff"],
            "max_probability_delta": float(neg_prob_delta),
            "citations_at_cutoff_delta": float(neg_cits_delta),
            "passed": negative_passed,
            "verdict": "PASSED: Zero Temporal Leakage (Strict Cutoff Firewall Verified)" if negative_passed else "FAILED: Leakage Detected"
        },
        "positive_control": {
            "description": "Injected +50 citations to cutoff year (pre-cutoff historical). Features and probabilities must change.",
            "citations_added": 50,
            "target_year": cutoff_year,
            "result_citations_at_cutoff": pos_pred["citations_at_cutoff"],
            "max_probability_delta": float(pos_prob_delta),
            "citations_at_cutoff_delta": float(pos_cits_delta),
            "passed": positive_passed,
            "verdict": "PASSED: Sensitive to Pre-Cutoff Signal (Model Actually Uses Real Features)" if positive_passed else "FAILED: Invariant to Historical Signal"
        },
        "firewall_integrity": bool(negative_passed and positive_passed)
    }


@router.get("/poison/{work_id}")
async def test_poison_invariance(work_id: str):
    """Backward-compatible endpoint: calls run_leakage_controls."""
    res = await run_leakage_controls(work_id)
    neg = res["negative_control"]
    return {
        "work_id": res["work_id"],
        "title": res["title"],
        "cutoff_year": res["cutoff_year"],
        "poison_injected_year": neg["target_year"],
        "poison_citations_added": neg["citations_added"],
        "clean_probabilities": res["baseline"]["probabilities"],
        "poisoned_probabilities": neg["result_citations_at_cutoff"],
        "max_probability_delta": neg["max_probability_delta"],
        "temporal_firewall_intact": neg["passed"],
        "verdict": neg["verdict"]
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
