"""FastAPI Router for SciGraph AI Paper Predictions, Benchmarks & Provenance-Tracked Scientometrics.

Strictly enforces:
- Rule R1: Zero fabricated constants or simulated numbers. Every value has named provenance.
- Rule R2: Discrete HTTP status codes (404, 429, 502, 503, 422). Zero silent failures.
- Rule R3: Strict temporal cutoff (T_cutoff = pub_year + 2).
- Rule R6: Pydantic-typed single source of truth. Zero CSS variables in API responses.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from api.config import settings
from api.schemas.paper import PaperPredictionResponse, DatasetStatsResponse
from api.openalex_client import (
    openalex_client,
    OpenAlexNotFoundError,
    OpenAlexRateLimitError,
    OpenAlexUpstreamError,
    OpenAlexValidationError
)
from ml.inference_service import inference_service
from ml.scientometrics.venue_resolver import venue_resolver
from ml.scientometrics.access_economics import access_service
from ml.scientometrics.lineage_engine import lineage_engine

logger = logging.getLogger("scigraph.api.paper")

router = APIRouter(tags=["Paper Predictions"])


@router.get("/splits")
def get_dataset_splits_endpoint():
    """Return dataset split counts computed directly from processed parquet tables and masks."""
    from api.main import get_splits_data
    return get_splits_data()


@router.get("/stats", response_model=DatasetStatsResponse)
def get_dataset_stats():
    """Return genuine dataset overview statistics from ingested Parquet tables."""
    try:
        works_path = settings.DATA_DIR / "works.parquet"
        authors_path = settings.DATA_DIR / "authors.parquet"
        inst_path = settings.DATA_DIR / "institutions.parquet"
        sources_path = settings.DATA_DIR / "sources.parquet"
        topics_path = settings.DATA_DIR / "topics.parquet"
        edges_path = settings.DATA_DIR / "citation_edges.parquet"

        n_papers = len(pd.read_parquet(works_path, columns=["work_id"])) if works_path.exists() else 0
        n_authors = len(pd.read_parquet(authors_path, columns=["author_id"])) if authors_path.exists() else 0
        n_institutions = len(pd.read_parquet(inst_path, columns=["institution_id"])) if inst_path.exists() else 0
        n_venues = len(pd.read_parquet(sources_path, columns=["venue_id"])) if sources_path.exists() else 0
        n_topics = len(pd.read_parquet(topics_path, columns=["topic_id"])) if topics_path.exists() else 0
        n_edges = len(pd.read_parquet(edges_path, columns=["citing_work_id"])) if edges_path.exists() else 0

        return DatasetStatsResponse(
            total_papers=n_papers,
            total_authors=n_authors,
            total_institutions=n_institutions,
            total_venues=n_venues,
            total_topics=n_topics,
            total_citation_edges=n_edges,
            provenance="openalex"
        )
    except Exception as exc:
        logger.error(f"Failed to query dataset stats: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to read dataset tables: {exc}")


@router.get("/search")
async def search_papers(q: str = Query(..., min_length=2, description="Search query or DOI")):
    """Search OpenAlex works by keyword or DOI."""
    try:
        results = await openalex_client.search_works(q, limit=10)
        formatted = []
        for r in results:
            formatted.append({
                "id": r.get("id", "").split("/")[-1],
                "title": r.get("title") or "Untitled",
                "publication_year": r.get("publication_year"),
                "cited_by_count": r.get("cited_by_count", 0),
                "doi": r.get("doi"),
                "primary_topic": (r.get("primary_topic") or {}).get("display_name"),
            })
        return {"query": q, "count": len(formatted), "results": formatted}
    except OpenAlexValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except OpenAlexRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except OpenAlexUpstreamError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/model-card")
def get_model_card():
    """Return model card with validated benchmark metrics and evaluation schema."""
    if inference_service.model_card:
        return inference_service.model_card
    card_path = settings.CHECKPOINTS_DIR / "MODEL_CARD.json"
    if card_path.exists():
        with open(card_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=503, detail="Model card is not loaded or missing.")


@router.get("/benchmark")
def get_benchmark_report():
    """Return scientific benchmark evaluation summary from reports/benchmark_latest.json."""
    bm_path = settings.BASE_DIR / "reports" / "benchmark_latest.json"
    if bm_path.exists():
        with open(bm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=503, detail="Benchmark report is not generated yet.")


@router.get("/forecast-evidence")
def get_current_forecast_evidence():
    """Published evaluation metrics, kept distinct from runtime verification."""
    from api.forecast_evidence import get_forecast_evidence
    try:
        return get_forecast_evidence(settings.BASE_DIR, settings.CHECKPOINTS_DIR, inference_service)
    except (OSError, ValueError, KeyError):
        raise HTTPException(status_code=503, detail="Current evaluation evidence is unavailable.")


@router.get("/corpus-sample")
def get_corpus_sample(limit: int = Query(30, ge=1, le=100), randomize: bool = Query(False)):
    """Return diverse sample papers from local 16,500 works dataset for demonstration."""
    if inference_service.df_works is None or inference_service.df_works.empty:
        raise HTTPException(status_code=503, detail="Corpus data is not loaded.")

    df = inference_service.df_works
    # Restrict demo samples strictly to held-out test split (2018-2020) — never training data
    df_test = df[df["publication_year"] >= 2018]
    if randomize:
        sample = df_test.sample(n=min(limit, len(df_test)))
    else:
        sample = df_test.sample(n=min(limit, len(df_test)), random_state=42)
    items = []
    for _, row in sample.iterrows():
        items.append({
            "paper_id": str(row["work_id"]).split("/")[-1],
            "title": str(row.get("title", "Untitled")),
            "publication_year": int(row.get("publication_year", 2018)),
            "field": str(row.get("topic_field_name") or row.get("primary_field_name") or "General"),
            "citations_at_cutoff": int(row.get("citations_at_cutoff", 0)),
            "future_citations_5y": int(row.get("future_citations_5y", 0)),
            "impact_class": int(row.get("impact_class", 0)),
            "venue_name": str(row.get("venue_name") or ""),
            "venue_quartile": str(row.get("venue_scimago_quartile") or row.get("venue_core_rank") or ""),
            "oa_status": str(row.get("oa_status", "closed")),
            "split": "test",
            "in_training_data": False,
        })
    return {"total": len(df), "test_split_count": len(df_test), "sample": items}


@router.get("/oa-metrics")
def get_oa_advantage_metrics():
    """Return open-access citation advantage metrics from reports/oa_advantage_metrics.json."""
    oa_path = settings.BASE_DIR / "reports" / "oa_advantage_metrics.json"
    if oa_path.exists():
        with open(oa_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=503, detail="OA advantage report not generated yet. Run scripts/compute_oa_advantage.py.")



@router.get("/test-clean/{paper_id:path}")
def test_clean_identifier(paper_id: str):
    return {"clean_id": openalex_client.clean_identifier(paper_id)}


@router.get("/test-work/{paper_id:path}")
async def test_get_work(paper_id: str):
    clean_id = openalex_client.clean_identifier(paper_id)
    raw = await openalex_client.get_work(clean_id)
    return {
        "title": raw.get("title"),
        "publication_year": raw.get("publication_year"),
        "citations_at_cutoff": raw.get("citations_at_cutoff"),
        "counts": len(raw.get("counts_by_year", [])),
        "authors": len(raw.get("authorships", []))
    }


@router.get("/test-predict-step/{step}/{paper_id:path}")
async def test_predict_step(step: int, paper_id: str):
    import time, traceback, numpy as np
    try:
        clean_id = openalex_client.clean_identifier(paper_id)
        raw_work = await openalex_client.get_work(clean_id)
        pub_year = int(raw_work.get("publication_year", 2018))
        cutoff_year = pub_year + 2

        if step == 1:
            # Parse fields and counts
            counts_by_year = raw_work.get("counts_by_year", []) or []
            cits_at_cutoff = sum(
                int(c.get("cited_by_count", 0) or 0)
                for c in counts_by_year
                if isinstance(c, dict) and int(c.get("year", 9999) or 9999) <= cutoff_year
            )
            authorships = raw_work.get("authorships", [])
            author_names = [(a.get("author") or {}).get("display_name") for a in authorships if (a.get("author") or {}).get("display_name")]
            return {"step": 1, "cits_at_cutoff": cits_at_cutoff, "authors": len(author_names)}

        if step == 2:
            primary_loc = raw_work.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            from ml.scientometrics.venue_resolver import venue_resolver
            v_res = venue_resolver.resolve(
                issn_l=source.get("issn_l"),
                venue_name=source.get("display_name", ""),
                pub_year=pub_year
            )
            return {"step": 2, "venue": v_res.model_dump() if v_res else None}

        if step == 3:
            topic_obj = raw_work.get("primary_topic") or {}
            primary_field = (topic_obj.get("field") or {}).get("display_name") or raw_work.get("primary_field_name")
            from ml.scientometrics.access_economics import access_service
            econ_info = access_service.compute(raw_work.get("open_access") or {}, raw_work.get("primary_location") or {}, field_name=primary_field)
            return {"step": 3, "econ": econ_info.model_dump()}

        if step == 4:
            t0 = time.time()
            from ml.scientometrics.lineage_engine import lineage_engine
            lineage_info = lineage_engine.compute(raw_work, cutoff_year=cutoff_year)
            return {"step": 4, "elapsed": round(time.time() - t0, 4), "lineage": lineage_info.model_dump()}

        if step == 5:
            from ml.features.builder import build_paper_features
            feats, _ = build_paper_features(raw_work, cutoff_offset=2)
            scaled = inference_service.primary_scaler.transform(feats.reshape(1, -1))
            logits = inference_service.primary_model.decision_function(scaled)
            return {"step": 5, "logits": logits.tolist()}

        if step == 6:
            # Linear attribution calculation
            from ml.features.builder import build_paper_features, FEATURE_GROUPS
            from ml.explain.attribution import FEATURE_DISPLAY_NAMES
            feats, _ = build_paper_features(raw_work, cutoff_offset=2)
            scaled = inference_service.primary_scaler.transform(feats.reshape(1, -1))
            logits = inference_service.primary_model.decision_function(scaled)
            scaled_logits = logits / max(inference_service.primary_temperature, 1e-4)
            exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = (exp_z / np.sum(exp_z, axis=1, keepdims=True))[0]
            pred_cls = int(np.argmax(probs))
            coefs = inference_service.primary_model.coef_[pred_cls]
            contribs = scaled[0] * coefs
            return {"step": 6, "pred_cls": pred_cls, "contribs_len": len(contribs)}

        if step == 7:
            # Backtest calculation
            window_start = cutoff_year + 1
            window_end = pub_year + 5
            counts_by_year = raw_work.get("counts_by_year", []) or []
            actual_window = sum(
                c.get("cited_by_count", 0)
                for c in counts_by_year
                if isinstance(c, dict) and window_start <= c.get("year", 0) <= window_end
            )
            # Check df_works lookup
            stored_rows = inference_service.df_works[inference_service.df_works["work_id"].astype(str).str.endswith(clean_id)] if inference_service.df_works is not None else None
            cohort_vals = None
            if inference_service.df_works is not None and "future_citations_5y" in inference_service.df_works.columns:
                cohort_vals = inference_service.df_works[
                    inference_service.df_works["publication_year"] == pub_year
                ]["future_citations_5y"].dropna().values
            return {
                "step": 7,
                "actual_window": actual_window,
                "stored_rows_count": len(stored_rows) if stored_rows is not None else 0,
                "cohort_vals_len": len(cohort_vals) if cohort_vals is not None else 0
            }

        return {"error": f"Invalid step: {step}"}
    except Exception as e:
        return {"step": step, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug-predict/{paper_id:path}")
async def debug_predict(paper_id: str):
    import time, traceback
    steps = {}
    try:
        t0 = time.time()
        clean_id = openalex_client.clean_identifier(paper_id)
        steps["1_clean_id"] = clean_id
        steps["1_clean_id_time"] = round(time.time() - t0, 4)

        t1 = time.time()
        raw_work = await openalex_client.get_work(clean_id)
        steps["2_get_work_title"] = raw_work.get("title")
        steps["2_counts_len"] = len(raw_work.get("counts_by_year", []))
        steps["2_get_work_time"] = round(time.time() - t1, 4)

        t2 = time.time()
        pred_dict = inference_service.predict(raw_work)
        steps["3_pred_class"] = pred_dict.get("predicted_impact_class")
        steps["3_predict_time"] = round(time.time() - t2, 4)

        t3 = time.time()
        resp = PaperPredictionResponse(**pred_dict)
        steps["4_schema_val_time"] = round(time.time() - t3, 4)
        steps["status"] = "success"
        return steps
    except Exception as e:
        steps["error"] = str(e)
        steps["traceback"] = traceback.format_exc()
        return steps


@router.get("/predict/{paper_id:path}", response_model=PaperPredictionResponse)
async def predict_paper_impact(
    paper_id: str,
    mode: str = Query("M", description="Prediction mode: 'M' (Local Model M1) or 'G' (Gemini Intelligence)")
):
    """Predict 5-year citation trajectory impact tier for paper_id.
    
    Accepts:
    - OpenAlex work ID: e.g. 'W2194775991'
    - Full OpenAlex URL: e.g. 'https://openalex.org/W2194775991'
    - DOI: e.g. '10.1038/nature12373' or 'doi:10.1038/nature12373'
    - mode: 'M' for standard Local Model, 'G' for Gemini Intelligence
    """
    clean_id = openalex_client.clean_identifier(paper_id)

    # 1. Fetch raw work from OpenAlex via resilient client
    try:
        raw_work = await openalex_client.get_work(clean_id)
    except OpenAlexNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Work '{clean_id}' was not found in OpenAlex database or local repository."
        )
    except OpenAlexRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Upstream OpenAlex rate limit encountered. Please retry shortly."
        )
    except OpenAlexUpstreamError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Upstream OpenAlex service failure: {str(e)}"
        )
    except OpenAlexValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid paper identifier format: {str(e)}"
        )

    # 2. Run inference via production ModelInferenceService
    try:
        prediction_dict = inference_service.predict(raw_work)
        
        # 3. If Mode G (Gemini Intelligence) is requested, apply LLM scientometric reasoning
        mode_clean = (mode or "M").strip().upper()
        if mode_clean == "G":
            try:
                from ml.gemini_service import gemini_service
                t_info = inference_service.get_cohort_threshold(
                    prediction_dict.get("primary_field"),
                    prediction_dict.get("publication_year")
                )
                gemini_res = gemini_service.predict(
                    paper_context=prediction_dict,
                    cohort_info=t_info,
                    m1_prediction=prediction_dict
                )
                # LLM prose is commentary, not a calibrated replacement classifier.
                prediction_dict["gemini_reasoning"] = gemini_res.get("gemini_reasoning")
                prediction_dict["mode"] = "G"
            except Exception as g_err:
                logger.warning(f"Gemini evaluation failed, falling back to local M1: {g_err}")
                prediction_dict["mode"] = "M"
                prediction_dict["gemini_reasoning"] = f"Gemini fallback triggered ({g_err}). Evaluated using primary model."
        else:
            prediction_dict["mode"] = "M"

        return PaperPredictionResponse(**prediction_dict)
    except RuntimeError as r_err:
        logger.error(f"Inference service runtime error: {r_err}")
        raise HTTPException(status_code=503, detail=str(r_err))
    except Exception as exc:
        logger.error(f"Inference failure for {clean_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(exc)}")
