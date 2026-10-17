"""Dense Corpus Downloader for SciGraph AI 2.0 (Phase A).

Features:
- Resumable download with state persistence (data/cache/dense/state.json)
- Partitioned by publication year (2012-2022) to avoid API key requirement for cursor pagination
- Polite pool email and custom User-Agent
- Exponential backoff and retry on transient errors or HTTP 429
- Local disk caching with annual JSONL shards (data/cache/dense/shards/)
- Conversion to structured Parquet tables in data/dense/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.config import settings

POLITE_EMAIL = settings.OPENALEX_POLITE_EMAIL or "researcher@scigraph.org"
USER_AGENT = f"SciGraphDenseIngest/2.0 (mailto:{POLITE_EMAIL})"
BASE_URL = "https://api.openalex.org/works"

CACHE_DIR = REPO_ROOT / "data" / "cache" / "dense"
SHARDS_DIR = CACHE_DIR / "shards"
STATE_FILE = CACHE_DIR / "state.json"
DENSE_DATA_DIR = REPO_ROOT / "data" / "dense"


def clean_id(raw_id: Optional[str]) -> str:
    if not raw_id:
        return ""
    return str(raw_id).split("/")[-1]


class DenseCorpusDownloader:
    def __init__(
        self,
        subfield_id: str = "1707",
        subfield_name: str = "Computer Vision and Pattern Recognition",
        start_year: int = 2012,
        end_year: int = 2022,
        works_per_year: int = 3500,
        per_page: int = 200,
        requests_per_second: float = 6.0,
    ):
        self.subfield_id = subfield_id
        self.subfield_name = subfield_name
        self.start_year = start_year
        self.end_year = end_year
        self.works_per_year = works_per_year
        self.per_page = per_page
        self.pages_per_year = (works_per_year + per_page - 1) // per_page
        self.min_interval = 1.0 / requests_per_second

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        SHARDS_DIR.mkdir(parents=True, exist_ok=True)
        DENSE_DATA_DIR.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load state file: {e}. Starting fresh.")
        return {
            "subfield_id": self.subfield_id,
            "subfield_name": self.subfield_name,
            "years_completed": [],
            "current_year": self.start_year,
            "current_page": 1,
            "total_downloaded": 0,
            "elapsed_seconds": 0.0,
            "completed": False,
        }

    def _save_state(self):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def download(self) -> int:
        if self.state.get("completed", False):
            print(f"[INFO] Dense download already marked completed ({self.state.get('total_downloaded', 0)} works).")
            return self.state.get("total_downloaded", 0)

        total_downloaded = self.state.get("total_downloaded", 0)
        years_completed = set(self.state.get("years_completed", []))

        fields_select = (
            "id,doi,title,publication_year,publication_date,cited_by_count,"
            "counts_by_year,referenced_works,primary_topic,primary_location,"
            "authorships,open_access"
        )

        client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )

        total_planned = (self.end_year - self.start_year + 1) * self.works_per_year
        print("=" * 85)
        print(f"STARTING DENSE INGESTION FOR {self.subfield_name} (ID: {self.subfield_id})")
        print(f"Years: {self.start_year}–{self.end_year} | Works/Year: {self.works_per_year:,} | Planned: {total_planned:,}")
        print(f"Per Page: {self.per_page} | Pages/Year: {self.pages_per_year} | Resuming Total: {total_downloaded:,}")
        print("=" * 85)

        t_start = time.time()
        last_req_time = 0.0

        try:
            for year in range(self.start_year, self.end_year + 1):
                if year in years_completed:
                    print(f"  [SKIPPING] Year {year} already marked completed.")
                    continue

                shard_file = SHARDS_DIR / f"shard_{year}.jsonl"
                year_works = []
                # If shard file already exists partially, read it
                if shard_file.exists():
                    with open(shard_file, "r", encoding="utf-8") as sf:
                        for l in sf:
                            if l.strip():
                                year_works.append(json.loads(l))
                    start_page = (len(year_works) // self.per_page) + 1
                else:
                    start_page = 1

                print(f"\n--- Ingesting Year {year} (Pages {start_page}..{self.pages_per_year}, Current Count: {len(year_works)}) ---")

                with open(shard_file, "a", encoding="utf-8") as sf:
                    for page in range(start_page, self.pages_per_year + 1):
                        # Rate limit
                        elapsed_since_last = time.time() - last_req_time
                        if elapsed_since_last < self.min_interval:
                            time.sleep(self.min_interval - elapsed_since_last)

                        filter_str = (
                            f"primary_topic.subfield.id:{self.subfield_id},"
                            f"publication_year:{year},"
                            "type:article|proceedings-article,"
                            "has_abstract:true,"
                            "cited_by_count:>0"
                        )

                        params = {
                            "filter": filter_str,
                            "sort": "cited_by_count:desc",
                            "per_page": self.per_page,
                            "page": page,
                            "select": fields_select,
                            "mailto": POLITE_EMAIL,
                        }

                        resp_data = None
                        for attempt in range(5):
                            last_req_time = time.time()
                            try:
                                resp = client.get(BASE_URL, params=params)
                                if resp.status_code == 200:
                                    try:
                                        resp_data = resp.json()
                                        break
                                    except Exception as json_err:
                                        print(f"    [RETRY {attempt+1}/5] JSON decode error: {json_err}. Waiting...")
                                        time.sleep(2.0 * (2 ** attempt))
                                        continue
                                elif resp.status_code in [429, 500, 502, 503, 504]:
                                    backoff = 2.0 * (2 ** attempt)
                                    print(f"    [RETRY {attempt+1}/5] HTTP {resp.status_code}. Waiting {backoff:.1f}s...")
                                    time.sleep(backoff)
                                else:
                                    resp.raise_for_status()
                            except Exception as exc:
                                if attempt == 4:
                                    print(f"    [WARN] Max retries reached for Year {year} Page {page}: {exc}. Continuing...")
                                    break
                                backoff = 2.0 * (2 ** attempt)
                                print(f"    [RETRY {attempt+1}/5] Request exception: {exc}. Waiting {backoff:.1f}s...")
                                time.sleep(backoff)

                        if not resp_data:
                            print(f"    [WARN] No response for Year {year} Page {page}.")
                            break

                        results = resp_data.get("results", [])
                        if not results:
                            print(f"    [INFO] No more results for Year {year} at page {page}.")
                            break

                        for w in results:
                            sf.write(json.dumps(w) + "\n")

                        year_works.extend(results)
                        total_downloaded += len(results)

                        if page % 5 == 0 or page == self.pages_per_year:
                            print(f"    [Page {page:02d}/{self.pages_per_year:02d}] Year {year}: {len(year_works):,} works | Overall: {total_downloaded:,}")

                        if len(year_works) >= self.works_per_year:
                            break

                years_completed.add(year)
                self.state["years_completed"] = sorted(list(years_completed))
                self.state["total_downloaded"] = total_downloaded
                self.state["elapsed_seconds"] += time.time() - t_start
                self._save_state()
                t_start = time.time()

            self.state["completed"] = True
            self._save_state()

        finally:
            client.close()

        print(f"\n[DONE] Total works downloaded in annual shards: {total_downloaded:,}")
        return total_downloaded

    def process_shards_to_parquet(self):
        """Converts raw JSONL shards into structured, relational parquet tables."""
        print("=" * 85)
        print("CONVERTING ANNUAL SHARDS TO PARQUET TABLES IN data/dense/")
        print("=" * 85)

        shard_files = sorted(SHARDS_DIR.glob("shard_*.jsonl"))
        if not shard_files:
            print("[ERROR] No shard files found in", SHARDS_DIR)
            return

        works_records = []
        counts_records = []
        edges_records = []
        authors_records = []
        institutions_records = []

        seen_works: Set[str] = set()
        seen_institutions: Set[str] = set()

        for s_idx, sf in enumerate(shard_files):
            print(f"Processing shard {s_idx+1}/{len(shard_files)}: {sf.name}...")
            with open(sf, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        w = json.loads(line)
                    except Exception:
                        continue
                    wid = clean_id(w.get("id"))
                    if not wid or wid in seen_works:
                        continue
                    seen_works.add(wid)

                    pub_year = int(w.get("publication_year") or 0)
                    cited_by_count = int(w.get("cited_by_count") or 0)

                    # Venue extraction
                    prim_loc = w.get("primary_location") or {}
                    source = prim_loc.get("source") or {}
                    venue_id = clean_id(source.get("id"))
                    venue_name = source.get("display_name", "")

                    # Topic extraction
                    prim_topic = w.get("primary_topic") or {}
                    field_obj = prim_topic.get("field") or {}
                    subfield_obj = prim_topic.get("subfield") or {}
                    field_name = field_obj.get("display_name", "Computer Science")
                    subfield_name = subfield_obj.get("display_name", self.subfield_name)

                    # Open Access
                    oa = w.get("open_access") or {}
                    is_oa = bool(oa.get("is_oa", False))
                    oa_status = oa.get("oa_status", "closed")

                    authorships = w.get("authorships") or []
                    authors_count = len(authorships)

                    works_records.append({
                        "work_id": wid,
                        "doi": w.get("doi", ""),
                        "title": w.get("title", "Untitled") or "Untitled",
                        "publication_year": pub_year,
                        "publication_date": w.get("publication_date", ""),
                        "cited_by_count": cited_by_count,
                        "primary_field_name": field_name,
                        "primary_subfield_name": subfield_name,
                        "primary_venue_id": venue_id,
                        "primary_venue_name": venue_name,
                        "is_oa": is_oa,
                        "oa_status": oa_status,
                        "authors_count": authors_count,
                    })

                    # Counts by year
                    for c in w.get("counts_by_year") or []:
                        if isinstance(c, dict):
                            yr = int(c.get("year", 0))
                            cnt = int(c.get("cited_by_count", 0))
                            if yr > 0:
                                counts_records.append({
                                    "work_id": wid,
                                    "year": yr,
                                    "cited_by_count": cnt,
                                })

                    # Citation edges (pointing to referenced_works)
                    for ref in w.get("referenced_works") or []:
                        ref_id = clean_id(ref)
                        if ref_id:
                            edges_records.append({
                                "source_work_id": wid,
                                "target_work_id": ref_id,
                                "source_pub_year": pub_year,
                            })

                    # Authors & Institutions
                    for pos_idx, auth in enumerate(authorships):
                        a_obj = auth.get("author") or {}
                        aid = clean_id(a_obj.get("id"))
                        aname = a_obj.get("display_name", "")
                        pos = auth.get("author_position", "middle")

                        insts = auth.get("institutions") or []
                        inst_id = ""
                        inst_name = ""
                        if insts:
                            i_obj = insts[0]
                            inst_id = clean_id(i_obj.get("id"))
                            inst_name = i_obj.get("display_name", "")
                            if inst_id and inst_id not in seen_institutions:
                                seen_institutions.add(inst_id)
                                institutions_records.append({
                                    "institution_id": inst_id,
                                    "institution_name": inst_name,
                                    "country_code": i_obj.get("country_code", ""),
                                    "type": i_obj.get("type", "education"),
                                })

                        if aid:
                            authors_records.append({
                                "work_id": wid,
                                "author_id": aid,
                                "author_name": aname,
                                "author_position": pos,
                                "institution_id": inst_id,
                                "institution_name": inst_name,
                            })

        print(f"Building DataFrames...")
        df_works = pd.DataFrame(works_records)
        df_counts = pd.DataFrame(counts_records)
        df_edges = pd.DataFrame(edges_records)
        df_authors = pd.DataFrame(authors_records)
        df_insts = pd.DataFrame(institutions_records)

        # Filter edges to in-corpus edges (where both source and target exist in corpus)
        all_corpus_wids = set(df_works["work_id"])
        in_corpus_mask = df_edges["target_work_id"].isin(all_corpus_wids)
        df_in_corpus_edges = df_edges[in_corpus_mask].copy()

        print(f"Summary Statistics:")
        print(f"  Works: {len(df_works):,}")
        print(f"  Counts by Year Rows: {len(df_counts):,}")
        print(f"  Total Raw References: {len(df_edges):,}")
        print(f"  In-Corpus Edges: {len(df_in_corpus_edges):,} ({len(df_in_corpus_edges)/max(1, len(df_edges))*100:.1f}% internal coverage)")
        print(f"  Authorships: {len(df_authors):,}")
        print(f"  Institutions: {len(df_insts):,}")

        # Save to parquet
        df_works.to_parquet(DENSE_DATA_DIR / "works.parquet", index=False)
        df_counts.to_parquet(DENSE_DATA_DIR / "counts_by_year.parquet", index=False)
        df_in_corpus_edges.to_parquet(DENSE_DATA_DIR / "edges.parquet", index=False)
        df_edges.to_parquet(DENSE_DATA_DIR / "raw_references.parquet", index=False)
        df_authors.to_parquet(DENSE_DATA_DIR / "authors.parquet", index=False)
        df_insts.to_parquet(DENSE_DATA_DIR / "institutions.parquet", index=False)

        print(f"[SUCCESS] Saved dense dataset tables to {DENSE_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download and build dense corpus for SciGraph AI 2.0")
    parser.add_argument("--subfield_id", type=str, default="1707", help="Subfield ID (default: 1707 Computer Vision)")
    parser.add_argument("--subfield_name", type=str, default="Computer Vision and Pattern Recognition")
    parser.add_argument("--start_year", type=int, default=2012)
    parser.add_argument("--end_year", type=int, default=2022)
    parser.add_argument("--works_per_year", type=int, default=3500, help="Target number of works per year")
    parser.add_argument("--process_only", action="store_true", help="Only process existing shards to parquet")
    args = parser.parse_args()

    downloader = DenseCorpusDownloader(
        subfield_id=args.subfield_id,
        subfield_name=args.subfield_name,
        start_year=args.start_year,
        end_year=args.end_year,
        works_per_year=args.works_per_year,
    )

    if not args.process_only:
        downloader.download()

    downloader.process_shards_to_parquet()


if __name__ == "__main__":
    main()
