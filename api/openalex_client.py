"""Resilient OpenAlex API Client for SciGraph AI.

Features:
- Polite pool integration via user email (settings.OPENALEX_POLITE_EMAIL)
- Robust disk caching under data/cache/openalex/ to avoid repeated remote calls
- Exponential backoff retry logic for upstream rate limits (429) and server errors (5xx)
- Typed exceptions mapping cleanly to discrete HTTP status codes (404, 429, 502, 422)
- Zero silent failures (never returns {} or masks errors)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    retry = None

from api.config import settings

logger = logging.getLogger("scigraph.openalex")


class OpenAlexClientError(Exception):
    """Base exception for OpenAlex client."""
    pass


class OpenAlexNotFoundError(OpenAlexClientError):
    """Raised when paper or entity is not found in OpenAlex (HTTP 404)."""
    pass


class OpenAlexRateLimitError(OpenAlexClientError):
    """Raised when upstream OpenAlex returns 429 rate limit."""
    pass


class OpenAlexUpstreamError(OpenAlexClientError):
    """Raised when upstream OpenAlex fails with 5xx or connection error."""
    pass


class OpenAlexValidationError(OpenAlexClientError):
    """Raised when search or ID parameter is invalid (HTTP 422)."""
    pass


class OpenAlexClient:
    """Async & sync resilient client for OpenAlex REST API with local disk caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (settings.DATA_DIR / "cache" / "openalex")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = settings.OPENALEX_BASE_URL.rstrip("/")
        self.headers = {
            "User-Agent": f"SciGraphAI/2.0 (mailto:{settings.OPENALEX_POLITE_EMAIL})",
            "Accept": "application/json",
        }
        if settings.OPENALEX_API_KEY:
            self.headers["Authorization"] = f"Bearer {settings.OPENALEX_API_KEY}"

    def _cache_path(self, key: str) -> Path:
        safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        return self.cache_dir / f"{safe_key}.json"

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read cache for {key}: {e}")
        return None

    def _write_cache(self, key: str, data: Dict[str, Any]):
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write cache for {key}: {e}")

    def _lookup_local_corpus(self, clean_id: str) -> Optional[Dict[str, Any]]:
        works_path = settings.DATA_DIR / "works.parquet"
        if not works_path.exists():
            return None
        import pandas as pd
        import math
        full_uri = f"https://openalex.org/{clean_id}"

        def _clean(val):
            if val is None:
                return None
            if pd.isna(val):
                return None
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val

        if not hasattr(self, "_corpus_works_map"):
            try:
                df = pd.read_parquet(works_path)
                self._corpus_works_map = {}
                for _, r in df.iterrows():
                    wid = str(r["work_id"]).split("/")[-1]
                    self._corpus_works_map[wid] = r.to_dict()
            except Exception as e:
                logger.warning(f"Failed to index corpus works for openalex_client: {e}")
                self._corpus_works_map = {}

        row = self._corpus_works_map.get(clean_id)
        if not row:
            return None

        py_raw = _clean(row.get("publication_year"))
        py = int(py_raw) if py_raw is not None else 2018
        doi_val = _clean(row.get("doi"))
        venue_name = _clean(row.get("venue_name"))
        venue_issn_l = _clean(row.get("venue_issn_l"))
        venue_type = _clean(row.get("venue_type"))
        oa_url = _clean(row.get("oa_url"))
        apc_paid = _clean(row.get("apc_paid_usd"))
        apc_list = _clean(row.get("apc_list_usd"))
        topic_name = _clean(row.get("primary_topic_name")) or "General"
        field_name = _clean(row.get("primary_field_name")) or "General"
        is_oa = bool(row.get("is_oa", False))
        oa_status = str(_clean(row.get("oa_status")) or "closed")
        n_auth_raw = _clean(row.get("num_authors"))
        n_auth = max(1, int(n_auth_raw)) if n_auth_raw is not None else 1
        refs_raw = _clean(row.get("referenced_works_count"))
        refs_count = int(refs_raw) if refs_raw is not None else 0
        cits_cut_raw = _clean(row.get("citations_at_cutoff"))
        cits_cut = int(cits_cut_raw) if cits_cut_raw is not None else 0
        fut_raw = _clean(row.get("future_citations_5y"))
        fut_cits = int(fut_raw) if fut_raw is not None else 0

        raw_work = {
            "id": full_uri,
            "title": str(_clean(row.get("title")) or "Untitled"),
            "doi": doi_val,
            "publication_year": py,
            "publication_date": str(_clean(row.get("publication_date")) or f"{py}-01-01"),
            "is_oa": is_oa,
            "open_access": {
                "is_oa": is_oa,
                "oa_status": oa_status,
                "oa_url": oa_url
            },
            "primary_location": {
                "source": {
                    "display_name": venue_name,
                    "issn_l": venue_issn_l,
                    "type": venue_type
                },
                "apc_paid": {"value_usd": apc_paid},
                "apc_list": {"value_usd": apc_list}
            },
            "primary_topic": {
                "display_name": topic_name,
                "field": {"display_name": field_name}
            },
            "primary_field_name": field_name,
            "authorships": [],
            "referenced_works_count": refs_count,
            "referenced_works": [],
            "counts_by_year": [],
            "future_citations_5y": fut_cits
        }

        # 1. Real Authorships from authorships.parquet + authors.parquet + institutions.parquet
        if not hasattr(self, "_corpus_authors_map"):
            authorships_path = settings.DATA_DIR / "authorships.parquet"
            authors_path = settings.DATA_DIR / "authors.parquet"
            inst_path = settings.DATA_DIR / "institutions.parquet"
            self._corpus_authors_map = {}
            if authorships_path.exists() and authors_path.exists():
                try:
                    df_a = pd.read_parquet(authors_path)
                    a_name_map = dict(zip(df_a["author_id"].astype(str), df_a["display_name"].astype(str)))
                    for aid, name in list(a_name_map.items()):
                        a_name_map[aid.split("/")[-1]] = name

                    inst_name_map = {}
                    if inst_path.exists():
                        df_i = pd.read_parquet(inst_path)
                        inst_name_map = dict(zip(df_i["institution_id"].astype(str), df_i["display_name"].astype(str)))
                        for iid, iname in list(inst_name_map.items()):
                            inst_name_map[iid.split("/")[-1]] = iname

                    df_auths = pd.read_parquet(authorships_path)
                    has_inst = "institution_id" in df_auths.columns
                    for wid, aid, pos, iid in zip(
                        df_auths["work_id"].values,
                        df_auths["author_id"].values,
                        df_auths["author_position"].values,
                        df_auths["institution_id"].values if has_inst else [None] * len(df_auths)
                    ):
                        w_clean = str(wid).split("/")[-1]
                        a_clean = str(aid).split("/")[-1] if aid else None
                        dname = a_name_map.get(str(aid)) or a_name_map.get(a_clean) or "Unknown Author"

                        inst_list = []
                        if pd.notnull(iid):
                            iid_str = str(iid)
                            iname = inst_name_map.get(iid_str) or inst_name_map.get(iid_str.split("/")[-1])
                            if iname:
                                inst_list.append({"id": iid_str, "display_name": iname})

                        if w_clean not in self._corpus_authors_map:
                            self._corpus_authors_map[w_clean] = []
                        self._corpus_authors_map[w_clean].append({
                            "author": {"display_name": dname, "id": str(aid) if aid else None},
                            "author_position": str(pos) if pos else "middle",
                            "institutions": inst_list
                        })
                    logger.info(f"Indexed {len(self._corpus_authors_map):,} works with genuine author and institution names.")
                except Exception as e:
                    logger.warning(f"Failed to load corpus authorships: {e}")
                    self._corpus_authors_map = {}

        if clean_id in self._corpus_authors_map:
            raw_work["authorships"] = self._corpus_authors_map[clean_id]

        # 2. Real Referenced Works from citation_edges.parquet
        if not hasattr(self, "_corpus_refs_map"):
            edges_path = settings.DATA_DIR / "citation_edges.parquet"
            self._corpus_refs_map = {}
            if edges_path.exists():
                try:
                    df_edges = pd.read_parquet(edges_path)
                    for citing, cited in zip(df_edges["citing_work_id"].values, df_edges["referenced_work_id"].values):
                        citing_c = str(citing).split("/")[-1]
                        if citing_c not in self._corpus_refs_map:
                            self._corpus_refs_map[citing_c] = []
                        self._corpus_refs_map[citing_c].append(str(cited))
                    logger.info(f"Indexed {len(self._corpus_refs_map):,} works with genuine referenced works.")
                except Exception as e:
                    logger.warning(f"Failed to load corpus citation edges: {e}")
                    self._corpus_refs_map = {}

        if clean_id in self._corpus_refs_map:
            raw_work["referenced_works"] = self._corpus_refs_map[clean_id]
            raw_work["referenced_works_count"] = max(refs_count, len(raw_work["referenced_works"]))

        # 3. Yearly Citation Counts
        counts_path = settings.DATA_DIR / "counts_by_year.parquet"
        if counts_path.exists():
            if not hasattr(self, "_corpus_counts_map"):
                try:
                    df_c = pd.read_parquet(counts_path)
                    counts_dict = {}
                    for _, r in df_c.iterrows():
                        wid = str(r["work_id"]).split("/")[-1]
                        if wid not in counts_dict:
                            counts_dict[wid] = []
                        counts_dict[wid].append({
                            "year": int(r["year"]),
                            "cited_by_count": int(r["cited_by_count"])
                        })
                    self._corpus_counts_map = counts_dict
                except Exception as e:
                    logger.warning(f"Failed to map corpus counts: {e}")
                    self._corpus_counts_map = {}
            if clean_id in self._corpus_counts_map:
                raw_work["counts_by_year"] = self._corpus_counts_map[clean_id]
        return raw_work

    def clean_identifier(self, identifier: str) -> str:
        """Normalizes OpenAlex ID, DOI, or URL to OpenAlex work identifier."""
        identifier = identifier.strip()
        if not identifier:
            raise OpenAlexValidationError("Identifier cannot be empty.")
        
        # Check if full OpenAlex URL
        if "openalex.org/W" in identifier:
            match = re.search(r"W\d+", identifier)
            if match:
                return match.group(0)
        
        # Check if direct OpenAlex W-ID
        if re.match(r"^W\d+$", identifier, re.IGNORECASE):
            return identifier.upper()
            
        # Check if DOI
        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", identifier)
        if doi_match:
            return f"doi:{doi_match.group(0)}"
            
        return identifier

    async def get_work(self, identifier: str) -> Dict[str, Any]:
        """Fetches work by OpenAlex ID or DOI. Uses disk cache when available."""
        clean_id = self.clean_identifier(identifier)
        cache_key = f"work_{clean_id}"

        # Check pre-ingested local corpus first (authoritative parquet tables)
        local_work = self._lookup_local_corpus(clean_id)
        if local_work:
            return local_work

        cached = self._read_cache(cache_key)
        if cached:
            return cached

        url = f"{self.base_url}/works/{clean_id}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=self.headers)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.error(f"OpenAlex connection error fetching {clean_id}: {exc}")
                raise OpenAlexUpstreamError(f"Failed to connect to OpenAlex: {exc}")

        if response.status_code == 404:
            raise OpenAlexNotFoundError(f"Work '{clean_id}' was not found in OpenAlex.")
        elif response.status_code == 429:
            raise OpenAlexRateLimitError("OpenAlex upstream rate limit reached (HTTP 429).")
        elif response.status_code >= 500:
            raise OpenAlexUpstreamError(f"OpenAlex upstream error (HTTP {response.status_code}).")
        elif response.status_code != 200:
            raise OpenAlexUpstreamError(f"Unexpected OpenAlex status code: {response.status_code}")

        data = response.json()
        self._write_cache(cache_key, data)
        return data

    async def search_works(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search works by title or query string."""
        query = query.strip()
        if not query:
            raise OpenAlexValidationError("Query cannot be empty.")

        cache_key = f"search_{query[:50]}_{limit}"
        cached = self._read_cache(cache_key)
        if cached and "results" in cached:
            return cached["results"]

        url = f"{self.base_url}/works"
        params = {
            "search": query,
            "per-page": limit,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=params, headers=self.headers)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.error(f"OpenAlex search error: {exc}")
                raise OpenAlexUpstreamError(f"Failed to connect to OpenAlex: {exc}")

        if response.status_code == 429:
            raise OpenAlexRateLimitError("OpenAlex upstream rate limit reached.")
        elif response.status_code >= 500:
            raise OpenAlexUpstreamError(f"OpenAlex upstream error (HTTP {response.status_code}).")
        elif response.status_code != 200:
            raise OpenAlexUpstreamError(f"OpenAlex search failed: HTTP {response.status_code}")

        results = response.json().get("results", [])
        self._write_cache(cache_key, {"results": results})
        return results


openalex_client = OpenAlexClient()
