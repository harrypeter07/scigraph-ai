"""FastAPI Router for Retrospective Verification Engine (Phase 19 Part A)."""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List

from ml.verification.simulator import RetrospectiveVerificationSimulator

router = APIRouter(prefix="/api/v1/verification", tags=["Retrospective Verification"])
simulator = RetrospectiveVerificationSimulator()


@router.get("/list")
def list_verifiable_papers(limit: int = Query(50, ge=1, le=100)):
    """List all verifiable papers in dataset for selection in UI simulator."""
    try:
        papers = simulator.list_available_papers(limit=limit)
        return {
            "total_available": len(papers),
            "papers": papers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulate/{paper_id:path}")
def simulate_retrospective_prediction(paper_id: str):
    """Reconstruct paper's feature & graph snapshot at T_cutoff, run model forecast,
    and compare side-by-side with actual cohort-normalized 5-year outcome."""
    try:
        result = simulator.simulate_paper_forecast(paper_id)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
