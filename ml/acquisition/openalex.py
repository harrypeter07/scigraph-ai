"""OpenAlex API Acquisition Client.

Fetches AI/ML research publications from OpenAlex REST API with cursor pagination,
rate-limit throttling, polite pool identification, resumable JSONL batch archiving,
and optional Supabase database upserting.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Generator
import requests
import yaml
from dotenv import load_dotenv

from ml.db.supabase_client import SupabaseDBClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OpenAlexAcquisition")


def parse_abstract_inverted_index(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct plain text abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for _, word in word_positions])


def parse_openalex_record(raw_work: Dict[str, Any]) -> Dict[str, Any]:
    """Parse raw OpenAlex work object into normalized dictionary matching DATA_DICTIONARY.md.
    
    CRITICAL: Does NOT compute leakage features. Retains raw counts_by_year for trajectory reconstruction.
    """
    work_id = raw_work.get("id", "")
    title = raw_work.get("title") or ""
    pub_year = raw_work.get("publication_year")
    pub_date = raw_work.get("publication_date")
    doi = raw_work.get("doi")
    
    # Abstract reconstruction
    inv_index = raw_work.get("abstract_inverted_index")
    abstract_text = parse_abstract_inverted_index(inv_index)
    
    # Citation trajectory array
    counts_by_year = raw_work.get("counts_by_year", [])
    
    # Referenced works
    referenced_works = raw_work.get("referenced_works", [])
    
    # Topics & Primary Topic
    primary_topic_obj = raw_work.get("primary_topic") or {}
    primary_topic_id = primary_topic_obj.get("id", "")
    
    concepts = []
    for c in raw_work.get("concepts", []):
        concepts.append({
            "id": c.get("id", ""),
            "display_name": c.get("display_name", ""),
            "level": c.get("level", 0),
            "score": c.get("score", 0.0)
        })
        
    # Authorships & Institutions
    authorships = []
    institutions = []
    authors = []
    
    for position, auth in enumerate(raw_work.get("authorships", [])):
        author_obj = auth.get("author") or {}
        author_id = author_obj.get("id", "")
        author_name = author_obj.get("display_name", "")
        
        if author_id:
            authors.append({
                "id": author_id,
                "display_name": author_name
            })
            
        inst_list = auth.get("institutions", [])
        inst_id = inst_list[0].get("id", "") if inst_list else ""
        
        for inst in inst_list:
            if inst.get("id"):
                institutions.append({
                    "id": inst.get("id", ""),
                    "display_name": inst.get("display_name", ""),
                    "country_code": inst.get("country_code", ""),
                    "type": inst.get("type", "")
                })
                
        if author_id:
            authorships.append({
                "paper_id": work_id,
                "author_id": author_id,
                "institution_id": inst_id,
                "author_position": position
            })

    return {
        "paper": {
            "id": work_id,
            "doi": doi,
            "title": title,
            "publication_year": pub_year,
            "publication_date": pub_date,
            "abstract_text": abstract_text,
            "primary_topic_id": primary_topic_id,
            "referenced_works_count": len(referenced_works),
            "referenced_works": referenced_works,
            "counts_by_year": counts_by_year,
            "raw_cited_by_count": raw_work.get("cited_by_count", 0)  # For archiving, NOT for features
        },
        "authors": authors,
        "institutions": institutions,
        "concepts": concepts,
        "authorships": authorships
    }


class OpenAlexAcquisitionClient:
    """Ingestion client for OpenAlex API."""

    def __init__(self, config_path: str = "configs/dataset.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        acq = self.config.get("acquisition", {})
        self.concept_filter = acq.get("concept_filter", {}).get("value", "C41008148")
        self.min_year = acq.get("year_range", {}).get("min_year", 2012)
        self.max_year = acq.get("year_range", {}).get("max_year", 2022)
        self.sample_size = acq.get("sample_size", 10000)
        self.delay = acq.get("rate_limit_delay_seconds", 0.1)
        self.batch_size = acq.get("batch_size", 200)

        self.polite_email = os.getenv("OPENALEX_POLITE_EMAIL", "hassan@example.edu")
        self.raw_dir = self.config.get("paths", {}).get("raw_dir", "data/raw/openalex")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.state_file = os.path.join(self.raw_dir, "ingestion_state.json")
        self.log_file = os.path.join(self.raw_dir, "ingestion_log.json")
        self.db_client = SupabaseDBClient()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"cursor": "*", "total_ingested": 0, "completed": False}

    def _save_state(self, cursor: str, total_ingested: int, completed: bool = False):
        state = {
            "cursor": cursor,
            "total_ingested": total_ingested,
            "completed": completed,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _log_query_config(self, query_params: Dict[str, Any]):
        log_data = {
            "query_params": query_params,
            "config": self.config,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    def fetch_batch(self, cursor: str = "*", limit: int = 200) -> Dict[str, Any]:
        """Fetch a single page of results from OpenAlex with rate-limit retries."""
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"concepts.id:{self.concept_filter},publication_year:{self.min_year}-{self.max_year},has_abstract:true",
            "per-page": limit,
            "cursor": cursor,
            "mailto": self.polite_email
        }
        
        self._log_query_config(params)

        for attempt in range(5):
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429 or response.status_code >= 500:
                    wait_time = 2 ** attempt
                    logger.warning(f"HTTP {response.status_code}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP {response.status_code}: {response.text}")
                    response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Request failed ({e}). Retrying in {2**attempt}s...")
                time.sleep(2 ** attempt)

        raise RuntimeError("Failed to fetch data from OpenAlex API after 5 attempts.")

    def run_acquisition(self, max_records: Optional[int] = None) -> int:
        """Execute acquisition loop with cursor pagination, writing to raw JSONL."""
        target_records = max_records or self.sample_size
        state = self._load_state()

        if state.get("completed", False):
            logger.info("Ingestion state indicates completion. Reset state_file to restart.")
            return state.get("total_ingested", 0)

        cursor = state.get("cursor", "*")
        total_ingested = state.get("total_ingested", 0)
        batch_index = total_ingested // self.batch_size + 1

        logger.info(f"Starting ingestion: target={target_records}, cursor={cursor}, ingested_so_far={total_ingested}")

        while total_ingested < target_records and cursor:
            per_page = min(self.batch_size, target_records - total_ingested)
            data = self.fetch_batch(cursor=cursor, limit=per_page)

            results = data.get("results", [])
            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")

            if not results:
                logger.info("No more results returned from OpenAlex.")
                self._save_state(cursor, total_ingested, completed=True)
                break

            # Write batch to JSONL
            batch_filename = os.path.join(self.raw_dir, f"openalex_batch_{batch_index:04d}.jsonl")
            with open(batch_filename, "a", encoding="utf-8") as f:
                for work in results:
                    f.write(json.dumps(work) + "\n")

            total_ingested += len(results)
            batch_index += 1
            cursor = next_cursor

            self._save_state(cursor, total_ingested, completed=(total_ingested >= target_records or not cursor))
            logger.info(f"Batch {batch_index-1} written ({len(results)} records). Total: {total_ingested}/{target_records}")

            time.sleep(self.delay)

        logger.info(f"Acquisition completed. Total records saved: {total_ingested}")
        return total_ingested


if __name__ == "__main__":
    client = OpenAlexAcquisitionClient()
    # Ingest a small test batch (e.g. 50 records) for verification
    client.run_acquisition(max_records=50)
