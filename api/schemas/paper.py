"""Pydantic V2 Type-Safe Schemas for SciGraph AI REST Service.

Strictly enforces:
1. Zero presentation strings (no CSS variables like 'var(--accent-green)').
2. Zero duplicate keys (e.g., removed redundant badge vs quartile_badge).
3. Explicit provenance tags on every metric (openalex, scimago, core, computed, model).
4. Nullability on un-sourced data (Rule R1: never fabricate defaults).
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class VenueMetadata(BaseModel):
    venue_name: str
    issn_l: Optional[str] = None
    venue_type: Optional[str] = None
    ranking_system: Optional[Literal["SCImago", "CORE"]] = None
    quartile: Optional[Literal["Q1", "Q2", "Q3", "Q4"]] = None
    core_rank: Optional[Literal["A*", "A", "B", "C", "Unranked", "Australasian"]] = None
    sjr: Optional[float] = None
    h_index: Optional[float] = None
    is_in_doaj: Optional[bool] = None
    provenance: Optional[Literal["scimago", "core"]] = None


class AccessEconomicsMetadata(BaseModel):
    is_oa: bool
    oa_status: Literal["gold", "green", "hybrid", "bronze", "closed"]
    oa_url: Optional[str] = None
    apc_paid_usd: Optional[float] = None
    apc_list_usd: Optional[float] = None
    empirical_cohort_oa_advantage_pct: float
    empirical_ci_95: List[float] = Field(default_factory=list)
    scientometric_notice: str
    provenance: Literal["openalex | computed"] = "openalex | computed"


class CitationTrajectoryPoint(BaseModel):
    year: int
    cited_by_count: int
    is_cutoff: bool = False
    is_future_horizon: bool = False


class LineageNodeModel(BaseModel):
    work_id: str
    title: str
    publication_year: Optional[int] = None
    direction: Literal["inflow", "outflow"]
    citation_count: Optional[int] = None


class BidirectionalLineageMetadata(BaseModel):
    inflow_count: int
    outflow_count: int
    inflow_outflow_ratio: Optional[float] = None
    citation_normalized_percentile: Optional[float] = None
    cd_disruption_index: Optional[float] = None
    inflow_sample: List[LineageNodeModel] = Field(default_factory=list)
    outflow_sample: List[LineageNodeModel] = Field(default_factory=list)
    provenance: Literal["openalex | computed"] = "openalex | computed"


class BacktestComparison(BaseModel):
    has_ground_truth: bool
    actual_future_citations_5y: Optional[int] = None
    actual_impact_class: Optional[int] = None
    predicted_impact_class: int
    is_correct_class: Optional[bool] = None
    actual_log_citations: Optional[float] = None
    predicted_log_citations: float
    absolute_error_log_citations: Optional[float] = None
    cohort_percentile: Optional[float] = None
    provenance: Literal["openalex | model | computed"] = "openalex | model | computed"


class FeatureAttribution(BaseModel):
    feature_name: str
    raw_value: Optional[float] = None
    importance_score: float
    direction: Literal["positive", "negative", "neutral"]
    provenance: Literal["model | lightgbm", "model | shap | lightgbm", "model | gnn"] = "model | lightgbm"


class PaperPredictionResponse(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    title: str
    publication_year: int
    primary_field: Optional[str] = None
    primary_topic: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    cutoff_year: int
    citations_at_cutoff: int
    predicted_impact_class: int
    predicted_impact_label: Literal["Low (<50th)", "Medium (50-90th)", "High (>=90th)"]
    calibrated_probabilities: Dict[str, float]
    predicted_future_citations_5y: float
    venue: Optional[VenueMetadata] = None
    access_economics: AccessEconomicsMetadata
    lineage: BidirectionalLineageMetadata
    trajectory: List[CitationTrajectoryPoint] = Field(default_factory=list)
    backtest: Optional[BacktestComparison] = None
    attributions: List[FeatureAttribution] = Field(default_factory=list)
    model_version: str = "PyG HeteroGraphSAGE v2.0"
    provenance: str = "openalex | scimago | core | model | computed"


class DatasetStatsResponse(BaseModel):
    total_papers: int
    total_authors: int
    total_institutions: int
    total_venues: int
    total_topics: int
    total_citation_edges: int
    provenance: Literal["openalex"] = "openalex"
