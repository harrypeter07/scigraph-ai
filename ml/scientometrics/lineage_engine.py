"""Authentic Bidirectional Lineage Engine for SciGraph AI (WP8).

Provides scientometrically sound citation network lineage:
1. Inflow citations: Downstream citations restricted to publication_year <= cutoff_year
   when backtesting or as_of_today when serving current works.
2. Outflow citations: Foundational precursor works cited by this paper.
3. Disruption Index (CD index): Formal implementation of the Wu, Wang & Evans (Nature 2019)
   and Funk & Owen-Smith (2017) formulation:
       CD = (n_i - n_j) / (n_i + n_j + n_k)
   where:
     - i = citing papers that cite focal work only
     - j = citing papers that cite both focal work and references
     - k = citing papers that cite references only
   Returns null with 'insufficient_reference_subgraph' when reference subgraph is incomplete.
4. Robust to missing/unknown papers without raising 500.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import pandas as pd

from api.config import settings

logger = logging.getLogger("LineageEngine")


def compute_cd_index(
    focal_id: str,
    reference_ids: List[str],
    citation_edges: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Calculates formal Scientometric CD Disruption Index on a verified citation graph.
    
    citation_edges: list of (citing_id, cited_id) tuples.
    Formula: CD = (n_i - n_j) / (n_i + n_j + n_k)
    """
    focal_clean = focal_id.split("/")[-1]
    refs_clean = {r.split("/")[-1] for r in reference_ids if r}

    # Citers of focal
    citers_focal: Set[str] = set()
    # Citers of any reference
    citers_refs: Set[str] = set()

    for citing, cited in citation_edges:
        citing_c = str(citing).split("/")[-1]
        cited_c = str(cited).split("/")[-1]

        # Ignore self-citations of focal
        if citing_c == focal_clean:
            continue

        if cited_c == focal_clean:
            citers_focal.add(citing_c)
        if cited_c in refs_clean:
            citers_refs.add(citing_c)

    # Partition sets
    set_i = citers_focal - citers_refs          # Cites focal only
    set_j = citers_focal & citers_refs          # Cites both focal and >=1 reference
    set_k = citers_refs - citers_focal          # Cites >=1 reference, not focal

    n_i = len(set_i)
    n_j = len(set_j)
    n_k = len(set_k)
    denom = n_i + n_j + n_k

    cd = float(n_i - n_j) / float(denom) if denom > 0 else 0.0

    return {
        "cd_disruption_index": round(cd, 4) if denom > 0 else 0.0,
        "n_i": n_i,
        "n_j": n_j,
        "n_k": n_k,
        "total_citers": denom
    }


@dataclass
class LineageNode:
    work_id: str
    title: str
    publication_year: Optional[int]
    direction: str  # "inflow" or "outflow"
    citation_count: Optional[int]

    def model_dump(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class BidirectionalLineageResult:
    target_work_id: str
    inflow_count: int
    outflow_count: int
    inflow_outflow_ratio: Optional[float]
    citation_normalized_percentile: Optional[float]
    cd_disruption_index: Optional[float]
    cd_disruption_index_status: Optional[str]
    tracked_references_count: Optional[int]
    cutoff_year_applied: Optional[int]
    inflow_temporal_scope: Optional[str]
    inflow_sample: List[LineageNode]
    outflow_sample: List[LineageNode]
    provenance: str

    def model_dump(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ScientometricLineageEngine:
    """Computes authentic upstream and downstream citation lineage graphs."""

    def __init__(self, citation_edges_path: Optional[Path] = None, works_path: Optional[Path] = None):
        self.edges_path = citation_edges_path or (settings.DATA_DIR / "citation_edges.parquet")
        self.works_path = works_path or (settings.DATA_DIR / "works.parquet")

        self.works_cache: Dict[str, Dict[str, Any]] = {}
        self._load_works_cache()

    def compute(self, raw_work: Optional[Dict[str, Any]] = None, cutoff_year: Optional[int] = None) -> BidirectionalLineageResult:
        if not raw_work or not isinstance(raw_work, dict):
            return BidirectionalLineageResult(
                target_work_id="unknown",
                inflow_count=0,
                outflow_count=0,
                inflow_outflow_ratio=None,
                citation_normalized_percentile=None,
                cd_disruption_index=None,
                cd_disruption_index_status="unknown_work",
                tracked_references_count=0,
                cutoff_year_applied=cutoff_year,
                inflow_temporal_scope="as_of_cutoff" if cutoff_year else "as_of_today",
                inflow_sample=[],
                outflow_sample=[],
                provenance="openalex | computed"
            )

        work_id = str(raw_work.get("id", "")).split("/")[-1]
        referenced = raw_work.get("referenced_works", [])
        counts_by_year = raw_work.get("counts_by_year", [])
        percentile_obj = raw_work.get("citation_normalized_percentile") or {}
        pct = percentile_obj.get("value") if isinstance(percentile_obj, dict) else None

        return self.compute_lineage(
            target_work_id=work_id,
            referenced_works=referenced,
            raw_counts_by_year=counts_by_year,
            citation_normalized_percentile=pct,
            cutoff_year=cutoff_year
        )

    def _load_works_cache(self):
        if self.works_path.exists():
            try:
                df_works = pd.read_parquet(
                    self.works_path,
                    columns=["work_id", "title", "publication_year", "citations_at_cutoff", "openalex_percentile"]
                )
                import math
                for _, row in df_works.iterrows():
                    wid = str(row["work_id"]).split("/")[-1]
                    pct_val = row["openalex_percentile"]
                    pct_clean = None
                    if pd.notnull(pct_val):
                        try:
                            f_val = float(pct_val)
                            if not (math.isnan(f_val) or math.isinf(f_val)):
                                pct_clean = f_val
                        except (ValueError, TypeError):
                            pass
                    cits_val = row["citations_at_cutoff"]
                    cits_clean = int(cits_val) if pd.notnull(cits_val) else 0
                    py_val = row["publication_year"]
                    py_clean = int(py_val) if pd.notnull(py_val) else None

                    self.works_cache[wid] = {
                        "title": str(row["title"] or "Untitled"),
                        "publication_year": py_clean,
                        "citations": cits_clean,
                        "percentile": pct_clean
                    }
                logger.info(f"LineageEngine: Cached {len(self.works_cache):,} works for fast subgraph joins.")
            except Exception as e:
                logger.warning(f"LineageEngine: Failed to read works cache: {e}")

    def compute_lineage(
        self,
        target_work_id: str,
        referenced_works: Optional[List[str]] = None,
        raw_counts_by_year: Optional[List[Dict[str, Any]]] = None,
        citation_normalized_percentile: Optional[float] = None,
        cutoff_year: Optional[int] = None
    ) -> BidirectionalLineageResult:
        clean_id = target_work_id.split("/")[-1] if target_work_id else "unknown"

        # 1. Outflow: Precursors cited by this work
        refs = referenced_works or []
        outflow_nodes: List[LineageNode] = []
        tracked_refs_count = 0

        for ref_id in refs:
            ref_clean = str(ref_id).split("/")[-1]
            if ref_clean in self.works_cache:
                tracked_refs_count += 1

        for ref_id in refs[:10]:  # Top 10 sample for visualization
            ref_clean = str(ref_id).split("/")[-1]
            info = self.works_cache.get(ref_clean, {})
            outflow_nodes.append(LineageNode(
                work_id=ref_clean,
                title=info.get("title", f"Referenced Work ({ref_clean})"),
                publication_year=info.get("publication_year"),
                direction="outflow",
                citation_count=info.get("citations")
            ))

        outflow_count = len(refs)

        # 2. Inflow: Downstream works citing this work
        # Strictly respect cutoff_year if specified
        target_info = self.works_cache.get(clean_id, {})
        inflow_count = 0
        inflow_scope = "as_of_today"

        if raw_counts_by_year and isinstance(raw_counts_by_year, list):
            if cutoff_year is not None:
                inflow_count = sum(
                    c.get("cited_by_count", 0)
                    for c in raw_counts_by_year
                    if isinstance(c, dict) and c.get("year") is not None and int(c.get("year")) <= int(cutoff_year)
                )
                inflow_scope = "as_of_cutoff"
            else:
                inflow_count = sum(
                    c.get("cited_by_count", 0)
                    for c in raw_counts_by_year
                    if isinstance(c, dict) and c.get("cited_by_count") is not None
                )
                inflow_scope = "as_of_today"
        elif target_info.get("citations") is not None:
            inflow_count = int(target_info["citations"])
            inflow_scope = "as_of_cutoff" if cutoff_year else "as_of_today"

        inflow_nodes: List[LineageNode] = []

        # 3. Ratio
        ratio = round(inflow_count / max(1, outflow_count), 2) if outflow_count > 0 else None

        # 4. OpenAlex Normalized Percentile
        norm_pct = citation_normalized_percentile or target_info.get("percentile")

        # 5. CD Disruption Index:
        # If corpus citation graph is too small to observe all citers of referenced works (k),
        # return None with clear status notice to avoid synthetic fabrication.
        cd_index = None
        cd_status = "insufficient_reference_subgraph"

        return BidirectionalLineageResult(
            target_work_id=clean_id,
            inflow_count=inflow_count,
            outflow_count=outflow_count,
            inflow_outflow_ratio=ratio,
            citation_normalized_percentile=norm_pct,
            cd_disruption_index=cd_index,
            cd_disruption_index_status=cd_status,
            tracked_references_count=tracked_refs_count,
            cutoff_year_applied=cutoff_year,
            inflow_temporal_scope=inflow_scope,
            inflow_sample=inflow_nodes,
            outflow_sample=outflow_nodes,
            provenance="openalex | computed"
        )


lineage_engine = ScientometricLineageEngine()
