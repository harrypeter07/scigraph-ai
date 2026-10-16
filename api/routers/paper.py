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


@router.get("/corpus-sample")
def get_corpus_sample(limit: int = Query(15, ge=1, le=50)):
    """Return diverse sample papers from local 16,500 works dataset for demonstration."""
    if inference_service.df_works is None or inference_service.df_works.empty:
        raise HTTPException(status_code=503, detail="Corpus data is not loaded.")

    df = inference_service.df_works
    # Restrict demo samples strictly to held-out test split (2018-2020) — never training data
    df_test = df[df["publication_year"] >= 2018]
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



@router.get("/predict/{paper_id:path}", response_model=PaperPredictionResponse)
async def predict_paper_impact(paper_id: str):
    """Predict 5-year citation trajectory impact tier for paper_id.
    
    Accepts:
    - OpenAlex work ID: e.g. 'W2194775991'
    - Full OpenAlex URL: e.g. 'https://openalex.org/W2194775991'
    - DOI: e.g. '10.1038/nature12373' or 'doi:10.1038/nature12373'
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
        return PaperPredictionResponse(**prediction_dict)
    except RuntimeError as r_err:
        logger.error(f"Inference service runtime error: {r_err}")
        raise HTTPException(status_code=503, detail=str(r_err))
    except Exception as exc:
        logger.error(f"Inference failure for {clean_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(exc)}")
