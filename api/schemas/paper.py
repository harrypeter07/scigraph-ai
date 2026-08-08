"""FastAPI Request & Response Schemas for Paper Prediction."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class PaperPredictionResponse(BaseModel):
    paper_id: str
    title: str
    publication_year: int
    predicted_impact_class: int
    impact_class_label: str
    class_probabilities: Dict[str, float]
    historical_citations_at_cutoff: int
    explanation: Dict[str, Any]


class DatasetStatsResponse(BaseModel):
    total_papers: int
    total_authors: int
    total_institutions: int
    total_citation_edges: int
