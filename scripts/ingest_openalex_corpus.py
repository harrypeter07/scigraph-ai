"""OpenAlex Corpus Ingestion Pipeline for SciGraph AI.

Fetches 20,000+ authentic peer-reviewed research papers across 5 major scientific fields
(Computer Science, Medicine, Physics, Biology, Engineering) published between 2010 and 2020.
Uses cohort-balanced pagination (55 field x year cohorts, ~370 works per cohort) to avoid
cursor-paging authentication gates and ensure balanced temporal and disciplinary representation.

Enforces strict scientometric temporal cutoffs:
- T_cutoff = pub_year + 2
- Historical citations: counts observed where year <= T_cutoff
- Evaluation horizon: counts observed where T_cutoff < year <= pub_year + 5
- Cohort percentiles: ranked within (primary_field_id, publication_year)
- Impact classes: 0 (<50%), 1 (50-90%), 2 (>=90%)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OpenAlexIngestion")

FIELDS = [
    {"id": "fields/17", "name": "Computer Science"},
    {"id": "fields/27", "name": "Medicine"},
    {"id": "fields/31", "name": "Physics and Astronomy"},
    {"id": "fields/13", "name": "Biochemistry, Genetics and Molecular Biology"},
    {"id": "fields/22", "name": "Engineering"},
]

TARGET_PER_COHORT = 365  # 5 fields x 11 years (2010-2020) x 365 ~= 20,075 works


def extract_apc_usd(apc_obj: Optional[Dict[str, Any]]) -> Optional[float]:
    """Extract APC value in USD if present."""
    if not apc_obj or not isinstance(apc_obj, dict):
        return None
    val = apc_obj.get("value_usd")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def fetch_cohort_page(
    client: httpx.Client,
    field_id: str,
    pub_year: int,
    page: int,
    per_page: int = 100,
    max_retries: int = 5
) -> Optional[Dict[str, Any]]:
    """Fetch a single page for a specific (field, pub_year) cohort."""
    url = f"{settings.OPENALEX_BASE_URL}/works"
    params = {
        "filter": f"primary_topic.field.id:{field_id},publication_year:{pub_year},has_abstract:true",
        "per-page": per_page,
        "page": page,
        "select": (
            "id,doi,title,publication_year,publication_date,open_access,primary_location,"
            "authorships,counts_by_year,primary_topic,apc_list,apc_paid,fwci,"
            "citation_normalized_percentile,referenced_works"
        ),
        "mailto": settings.OPENALEX_POLITE_EMAIL,
    }

    for attempt in range(max_retries):
        try:
            response = client.get(url, params=params, timeout=30.0)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait_sec = 2.0 ** attempt + 1.0
                logger.warning(f"Rate limited (429). Waiting {wait_sec:.1f}s...")
                time.sleep(wait_sec)
            elif response.status_code >= 500:
                wait_sec = 2.0 ** attempt + 0.5
                logger.warning(f"Upstream {response.status_code}. Retrying in {wait_sec:.1f}s...")
                time.sleep(wait_sec)
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")
                return None
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            wait_sec = 2.0 ** attempt + 1.0
            logger.warning(f"Connection error ({exc}). Retrying in {wait_sec:.1f}s...")
            time.sleep(wait_sec)

    return None


def run_ingestion():
    """Main ingestion coordinator."""
    logger.info("=" * 70)
    logger.info("SciGraph AI: Starting Authentic OpenAlex Corpus Ingestion Pipeline")
    logger.info(f"Target: ~20,000 papers across {len(FIELDS)} fields x 11 years (2010-2020)")
    logger.info(f"Polite pool mailto: {settings.OPENALEX_POLITE_EMAIL}")
    logger.info("=" * 70)

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    works_records: List[Dict[str, Any]] = []
    authors_dict: Dict[str, Dict[str, Any]] = {}
    institutions_dict: Dict[str, Dict[str, Any]] = {}
    authorships_records: List[Dict[str, Any]] = []
    sources_dict: Dict[str, Dict[str, Any]] = {}
    topics_dict: Dict[str, Dict[str, Any]] = {}
    citation_edges_records: List[Dict[str, Any]] = []
    counts_by_year_records: List[Dict[str, Any]] = []

    total_works_ingested = 0
    start_time = time.time()

    with httpx.Client(timeout=35.0) as client:
        for field in FIELDS:
            field_id = field["id"]
            field_name = field["name"]
            field_works_count = 0

            logger.info(f"\n>>> Starting Field: {field_name} ({field_id})")

            for year in range(settings.MIN_PUB_YEAR, settings.MAX_PUB_YEAR + 1):
                cohort_works_count = 0
                page = 1

                while cohort_works_count < TARGET_PER_COHORT and page <= 5:
                    per_page = min(100, TARGET_PER_COHORT - cohort_works_count)
                    data = fetch_cohort_page(
                        client=client,
                        field_id=field_id,
                        pub_year=year,
                        page=page,
                        per_page=per_page
                    )

                    if not data:
                        break

                    results = data.get("results", [])
                    if not results:
                        break

                    for work in results:
                        work_id = work.get("id")
                        if not work_id:
                            continue

                        pub_year = work.get("publication_year", year)

                        # 1. Primary Source / Venue
                        loc = work.get("primary_location") or {}
                        source = loc.get("source") or {}
                        venue_id = source.get("id")
                        venue_name = source.get("display_name")
                        venue_issn_l = source.get("issn_l")
                        venue_type = source.get("type")
                        venue_publisher = source.get("host_organization_name")

                        if venue_id and venue_id not in sources_dict:
                            sources_dict[venue_id] = {
                                "venue_id": venue_id,
                                "display_name": venue_name or "Unknown Venue",
                                "issn_l": venue_issn_l,
                                "type": venue_type or "unknown",
                                "host_organization_name": venue_publisher,
                                "is_in_doaj": source.get("is_in_doaj", False)
                            }

                        # 2. Topic, Subfield, Field
                        topic_obj = work.get("primary_topic") or {}
                        topic_id = topic_obj.get("id")
                        topic_name = topic_obj.get("display_name")
                        subfield_obj = topic_obj.get("subfield") or {}
                        field_obj = topic_obj.get("field") or {}
                        domain_obj = topic_obj.get("domain") or {}

                        topic_field_id = field_obj.get("id") or field_id
                        topic_field_name = field_obj.get("display_name") or field_name

                        if topic_id and topic_id not in topics_dict:
                            topics_dict[topic_id] = {
                                "topic_id": topic_id,
                                "topic_name": topic_name or "Unknown Topic",
                                "subfield_id": subfield_obj.get("id"),
                                "subfield_name": subfield_obj.get("display_name"),
                                "field_id": topic_field_id,
                                "field_name": topic_field_name,
                                "domain_id": domain_obj.get("id"),
                                "domain_name": domain_obj.get("display_name")
                            }

                        # 3. Authorships & Affiliations
                        authorships = work.get("authorships") or []
                        num_authors = len(authorships)
                        for auth in authorships:
                            author_data = auth.get("author") or {}
                            auth_id = author_data.get("id")
                            if auth_id:
                                if auth_id not in authors_dict:
                                    authors_dict[auth_id] = {
                                        "author_id": auth_id,
                                        "display_name": author_data.get("display_name") or "Unknown Author",
                                        "orcid": author_data.get("orcid")
                                    }

                                inst_list = auth.get("institutions") or []
                                inst_id = None
                                if inst_list:
                                    primary_inst = inst_list[0]
                                    inst_id = primary_inst.get("id")
                                    if inst_id and inst_id not in institutions_dict:
                                        institutions_dict[inst_id] = {
                                            "institution_id": inst_id,
                                            "display_name": primary_inst.get("display_name") or "Unknown Institution",
                                            "country_code": primary_inst.get("country_code"),
                                            "type": primary_inst.get("type")
                                        }

                                authorships_records.append({
                                    "work_id": work_id,
                                    "author_id": auth_id,
                                    "institution_id": inst_id,
                                    "author_position": auth.get("author_position", "middle")
                                })

                        # 4. Outgoing Citations (Lineage Edges)
                        ref_works = work.get("referenced_works") or []
                        for ref_id in ref_works:
                            citation_edges_records.append({
                                "citing_work_id": work_id,
                                "referenced_work_id": ref_id
                            })

                        # 5. Temporal Counts by Year & Strict Cutoffs
                        counts = work.get("counts_by_year") or []
                        cutoff_year = pub_year + settings.CUTOFF_OFFSET_YEARS  # pub_year + 2
                        eval_horizon_year = pub_year + settings.TRAJECTORY_HORIZON_YEARS  # pub_year + 5

                        citations_at_cutoff = 0
                        future_citations_5y = 0

                        for c in counts:
                            yr = c.get("year")
                            cnt = c.get("cited_by_count", 0)
                            if yr is not None and cnt is not None:
                                counts_by_year_records.append({
                                    "work_id": work_id,
                                    "year": yr,
                                    "cited_by_count": cnt
                                })
                                # Cutoff: only counts up to pub_year + 2
                                if yr <= cutoff_year:
                                    citations_at_cutoff += cnt
                                # Future Horizon: counts between cutoff and pub_year + 5
                                if cutoff_year < yr <= eval_horizon_year:
                                    future_citations_5y += cnt

                        # 6. Open Access & Economics
                        oa = work.get("open_access") or {}
                        is_oa = oa.get("is_oa", False)
                        oa_status = oa.get("oa_status", "closed")
                        oa_url = oa.get("oa_url")
                        apc_list_usd = extract_apc_usd(work.get("apc_list"))
                        apc_paid_usd = extract_apc_usd(work.get("apc_paid"))

                        fwci = work.get("fwci")
                        cnp = work.get("citation_normalized_percentile")
                        oa_percentile = cnp.get("value") if isinstance(cnp, dict) else None

                        # Temporal Split
                        if pub_year <= 2016:
                            split = "train"
                        elif pub_year == 2017:
                            split = "val"
                        else:
                            split = "test"

                        works_records.append({
                            "work_id": work_id,
                            "title": work.get("title") or "Untitled Paper",
                            "doi": work.get("doi"),
                            "publication_year": pub_year,
                            "publication_date": work.get("publication_date"),
                            "is_oa": is_oa,
                            "oa_status": oa_status,
                            "oa_url": oa_url,
                            "apc_list_usd": apc_list_usd,
                            "apc_paid_usd": apc_paid_usd,
                            "fwci": fwci,
                            "openalex_percentile": oa_percentile,
                            "primary_topic_id": topic_id,
                            "primary_field_id": topic_field_id,
                            "primary_field_name": topic_field_name,
                            "primary_subfield_id": subfield_obj.get("id"),
                            "primary_subfield_name": subfield_obj.get("display_name"),
                            "venue_id": venue_id,
                            "venue_name": venue_name,
                            "venue_issn_l": venue_issn_l,
                            "venue_type": venue_type,
                            "venue_publisher": venue_publisher,
                            "num_authors": num_authors,
                            "referenced_works_count": len(ref_works),
                            "cutoff_year": cutoff_year,
                            "citations_at_cutoff": citations_at_cutoff,
                            "future_citations_5y": future_citations_5y,
                            "temporal_split": split
                        })

                        cohort_works_count += 1
                        field_works_count += 1
                        total_works_ingested += 1

                    page += 1
                    time.sleep(settings.RATE_LIMIT_DELAY_SECONDS)

                logger.info(f"[{field_name} - {year}] Ingested {cohort_works_count} works. (Total so far: {total_works_ingested})")

    elapsed = time.time() - start_time
    logger.info(f"\nIngestion finished in {elapsed:.1f}s. Assembling DataFrames and calculating cohort rankings...")

    df_works = pd.DataFrame(works_records)
    # Deduplicate works by work_id
    df_works = df_works.drop_duplicates(subset=["work_id"]).reset_index(drop=True)

    # Compute cohort percentiles strictly within (primary_field_id, publication_year)
    logger.info("Computing cohort percentiles strictly within (primary_field_id, publication_year)...")
    df_works["cohort_percentile"] = df_works.groupby(
        ["primary_field_id", "publication_year"]
    )["future_citations_5y"].rank(pct=True) * 100.0

    # 3-Class Label assignment:
    # 0 = Low (<50th percentile)
    # 1 = Medium (50th - 90th percentile)
    # 2 = High (>=90th percentile)
    def assign_class(pct: float) -> int:
        if pct >= 90.0:
            return 2
        elif pct >= 50.0:
            return 1
        return 0

    df_works["impact_class"] = df_works["cohort_percentile"].apply(assign_class)

    logger.info(f"Class distribution:\n{df_works['impact_class'].value_counts().sort_index()}")
    logger.info(f"Temporal split distribution:\n{df_works['temporal_split'].value_counts()}")

    # Build other relational DataFrames
    df_authors = pd.DataFrame(list(authors_dict.values())).drop_duplicates(subset=["author_id"]).reset_index(drop=True)
    df_institutions = pd.DataFrame(list(institutions_dict.values())).drop_duplicates(subset=["institution_id"]).reset_index(drop=True)
    df_authorships = pd.DataFrame(authorships_records).drop_duplicates().reset_index(drop=True)
    df_sources = pd.DataFrame(list(sources_dict.values())).drop_duplicates(subset=["venue_id"]).reset_index(drop=True)
    df_topics = pd.DataFrame(list(topics_dict.values())).drop_duplicates(subset=["topic_id"]).reset_index(drop=True)
    df_citation_edges = pd.DataFrame(citation_edges_records).drop_duplicates().reset_index(drop=True)
    df_counts_by_year = pd.DataFrame(counts_by_year_records).drop_duplicates().reset_index(drop=True)

    # Write Parquet files
    logger.info(f"Saving tables to {settings.DATA_DIR}...")
    df_works.to_parquet(settings.DATA_DIR / "works.parquet", index=False)
    df_authors.to_parquet(settings.DATA_DIR / "authors.parquet", index=False)
    df_institutions.to_parquet(settings.DATA_DIR / "institutions.parquet", index=False)
    df_authorships.to_parquet(settings.DATA_DIR / "authorships.parquet", index=False)
    df_sources.to_parquet(settings.DATA_DIR / "sources.parquet", index=False)
    df_topics.to_parquet(settings.DATA_DIR / "topics.parquet", index=False)
    df_citation_edges.to_parquet(settings.DATA_DIR / "citation_edges.parquet", index=False)
    df_counts_by_year.to_parquet(settings.DATA_DIR / "counts_by_year.parquet", index=False)

    # Save temporal train/val/test splits to data/processed/
    logger.info(f"Saving splits to {settings.DATA_PROCESSED_DIR}...")
    df_train = df_works[df_works["temporal_split"] == "train"].reset_index(drop=True)
    df_val = df_works[df_works["temporal_split"] == "val"].reset_index(drop=True)
    df_test = df_works[df_works["temporal_split"] == "test"].reset_index(drop=True)

    df_train.to_parquet(settings.DATA_PROCESSED_DIR / "train.parquet", index=False)
    df_val.to_parquet(settings.DATA_PROCESSED_DIR / "val.parquet", index=False)
    df_test.to_parquet(settings.DATA_PROCESSED_DIR / "test.parquet", index=False)

    logger.info("=" * 70)
    logger.info("Authentic Data Pipeline Complete!")
    logger.info(f"Works: {len(df_works):,}")
    logger.info(f"Authors: {len(df_authors):,}")
    logger.info(f"Institutions: {len(df_institutions):,}")
    logger.info(f"Authorships: {len(df_authorships):,}")
    logger.info(f"Sources (Venues): {len(df_sources):,}")
    logger.info(f"Topics: {len(df_topics):,}")
    logger.info(f"Citation Edges: {len(df_citation_edges):,}")
    logger.info(f"Counts By Year: {len(df_counts_by_year):,}")
    logger.info(f"Train / Val / Test: {len(df_train):,} / {len(df_val):,} / {len(df_test):,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_ingestion()
