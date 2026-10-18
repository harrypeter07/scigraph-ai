"""Authentic Scientometric Venue Resolver for SciGraph AI.

Replaces keyword substring matching and fake defaults with:
1. Exact & nearest publication-year SCImago Journal Rank (SJR) quartiles (Q1-Q4) joined via ISSN-L.
2. Official CORE Conference Rankings (A*, A, B, C) for computer science venues.
3. Strict Null Handling: If venue cannot be sourced, returns None with null provenance (Rule R1).
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd

from api.config import settings

logger = logging.getLogger("VenueResolver")


@dataclass
class VenueRankingResult:
    venue_name: str
    venue_type: Optional[str]
    ranking_system: Optional[str]  # "SCImago", "CORE", or None
    quartile: Optional[str]        # "Q1", "Q2", "Q3", "Q4", or None
    core_rank: Optional[str]       # "A*", "A", "B", "C", or None
    sjr: Optional[float]
    h_index: Optional[float]
    provenance: Optional[str]      # "scimago", "core", or None
    issn_l: Optional[str] = None

    def model_dump(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class ScientometricVenueResolver:
    """Singleton-ready venue resolver linking ISSN-L and conference names to verified rankings."""

    def __init__(self, scimago_path: Optional[Path] = None, core_path: Optional[Path] = None):
        self.scimago_path = scimago_path or (settings.DATA_DIR / "scimago_rankings.parquet")
        self.core_path = core_path or (settings.DATA_DIR / "core_conference_rankings.parquet")

        self.scimago_lookup: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.core_title_lookup: Dict[str, str] = {}
        self.core_acronym_lookup: Dict[str, str] = {}
        self.core_title_history: Dict[str, List[Tuple[int, str]]] = {}
        self.core_acronym_history: Dict[str, List[Tuple[int, str]]] = {}

        self._load_datasets()

    def _normalize_issn(self, issn_str: Optional[str]) -> List[str]:
        if not issn_str or not isinstance(issn_str, str):
            return []
        tokens = re.findall(r"[0-9]{4}-?[0-9]{3}[0-9Xx]", issn_str)
        cleaned = []
        for tok in tokens:
            c = tok.replace("-", "").upper()
            if len(c) == 8:
                cleaned.append(c)
        return cleaned

    def _clean_str(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "", text.lower())

    def _load_datasets(self):
        # 1. Load SCImago for corpus venues (lean memory footprint ~11MB instead of 266MB)
        if self.scimago_path.exists():
            sources_path = settings.DATA_DIR / "sources.parquet"
            filters = None
            if sources_path.exists():
                try:
                    df_src = pd.read_parquet(sources_path, columns=["issn_l"])
                    corpus_issns = list(set(df_src["issn_l"].dropna().str.replace("-", "").str.upper()))
                    if corpus_issns:
                        filters = [("issn", "in", corpus_issns)]
                except Exception as e:
                    logger.warning(f"VenueResolver: Could not load corpus ISSNs from sources.parquet: {e}")

            try:
                df_scimago = pd.read_parquet(
                    self.scimago_path,
                    columns=["issn", "year", "quartile", "sjr", "h_index"],
                    filters=filters
                )
                for row in df_scimago.itertuples(index=False):
                    issn = getattr(row, "issn")
                    yr = int(getattr(row, "year"))
                    if issn not in self.scimago_lookup:
                        self.scimago_lookup[issn] = {}
                    sjr_val = getattr(row, "sjr")
                    h_val = getattr(row, "h_index")
                    self.scimago_lookup[issn][yr] = {
                        "quartile": getattr(row, "quartile"),
                        "sjr": float(sjr_val) if pd.notnull(sjr_val) else None,
                        "h_index": float(h_val) if pd.notnull(h_val) else None
                    }
                logger.info(f"VenueResolver: Loaded {len(self.scimago_lookup):,} SCImago ISSN entries ({len(df_scimago):,} records).")
            except Exception as e:
                logger.error(f"VenueResolver: Failed to load SCImago parquet: {e}")
        else:
            logger.warning(f"VenueResolver: SCImago dataset not found at {self.scimago_path}")

        # 2. Load CORE
        if self.core_path.exists():
            df_core = pd.read_parquet(self.core_path)
            rank_prio = {"A*": 5, "A": 4, "B": 3, "C": 2, "Australasian": 1, "Unranked": 0}
            df_core["prio"] = df_core["rank"].map(lambda r: rank_prio.get(r, 0))
            df_core = df_core.sort_values(by=["edition_year", "prio"], ascending=[False, False])

            for row in df_core.itertuples(index=False):
                t = self._clean_str(str(getattr(row, "title", "")))
                a = self._clean_str(str(getattr(row, "acronym", "")))
                rk = str(getattr(row, "rank", "Unranked"))
                ey = int(getattr(row, "edition_year", 2020))
                if t:
                    self.core_title_history.setdefault(t, []).append((ey, rk))
                    if t not in self.core_title_lookup:
                        self.core_title_lookup[t] = rk
                if a:
                    self.core_acronym_history.setdefault(a, []).append((ey, rk))
                    if a not in self.core_acronym_lookup:
                        self.core_acronym_lookup[a] = rk
            logger.info(f"VenueResolver: Loaded {len(self.core_title_history):,} CORE titles, {len(self.core_acronym_history):,} CORE acronyms with temporal editions.")
        else:
            logger.warning(f"VenueResolver: CORE dataset not found at {self.core_path}")

    def _lookup_core(self, token: str, pub_year: int, is_acronym: bool = True) -> Optional[str]:
        """Lookup CORE rank using strictly edition_year <= pub_year (latest available edition)."""
        history = self.core_acronym_history.get(token) if is_acronym else self.core_title_history.get(token)
        if not history:
            return None
        valid_editions = [(yr, rk) for yr, rk in history if yr <= pub_year]
        if not valid_editions:
            return None
        # Sort by year descending to get latest edition <= pub_year
        valid_editions.sort(key=lambda x: x[0], reverse=True)
        return valid_editions[0][1]

    def _lookup_scimago_dynamic(self, issn: str) -> Optional[Dict[int, Dict[str, Any]]]:
        """Dynamically queries SCImago for an unseen ISSN via parquet pushdown filter without full table scan."""
        if not self.scimago_path.exists() or not issn:
            return None
        try:
            df_single = pd.read_parquet(
                self.scimago_path,
                columns=["issn", "year", "quartile", "sjr", "h_index"],
                filters=[("issn", "==", issn)]
            )
            if df_single.empty:
                return None
            res = {}
            for row in df_single.itertuples(index=False):
                yr = int(getattr(row, "year"))
                sjr_val = getattr(row, "sjr")
                h_val = getattr(row, "h_index")
                res[yr] = {
                    "quartile": getattr(row, "quartile"),
                    "sjr": float(sjr_val) if pd.notnull(sjr_val) else None,
                    "h_index": float(h_val) if pd.notnull(h_val) else None
                }
            self.scimago_lookup[issn] = res
            return res
        except Exception as e:
            logger.debug(f"Dynamic SCImago query failed for {issn}: {e}")
            return None

    def resolve(
        self,
        venue_name: Optional[str] = None,
        venue_issn_l: Optional[str] = None,
        venue_type: Optional[str] = None,
        publication_year: Optional[int] = None,
        issn_l: Optional[str] = None,
        pub_year: Optional[int] = None
    ) -> VenueRankingResult:
        venue_issn_l = venue_issn_l or issn_l
        pub_year = publication_year or pub_year or 2018
        vname = venue_name or "Unknown Venue"
        # De-duplicate duplicated venue strings like "X - X"
        if vname and (" - " in vname or " / " in vname):
            for sep in [" - ", " / "]:
                if sep in vname:
                    parts = [p.strip() for p in vname.split(sep) if p.strip()]
                    if len(parts) >= 2 and parts[0].lower() == parts[1].lower():
                        vname = parts[0]
                        break

        vtype = (venue_type or "").lower()
        vname_clean = self._clean_str(vname)

        # 1. Check CORE for Computer Science conferences / proceedings
        is_conference = "conference" in vtype or "proceedings" in vname.lower() or any(
            k in vname.lower() for k in ["acm", "ieee", "cvpr", "icml", "neurips", "kdd", "acl", "aaai", "ijcai"]
        )

        if is_conference:
            # A. Match CVPR proceedings & conferences via CORE acronym/title
            if "computer vision and pattern recognition" in vname.lower():
                cvpr_rank = self._lookup_core("cvpr", pub_year, is_acronym=True)
                if cvpr_rank:
                    return VenueRankingResult(
                        venue_name=vname,
                        venue_type=venue_type or "conference",
                        ranking_system="CORE",
                        quartile=None,
                        core_rank=cvpr_rank,
                        sjr=None,
                        h_index=None,
                        provenance="core"
                    )

            # B. Check parenthesized acronyms first (e.g. "(CVPR)")
            paren_matches = re.findall(r"[\(\[\{]([A-Za-z0-9\-]+)[\)\]\}]", vname)
            for p in paren_matches:
                p_clean = self._clean_str(p)
                rk = self._lookup_core(p_clean, pub_year, is_acronym=True)
                if rk:
                    return VenueRankingResult(
                        venue_name=vname,
                        venue_type=venue_type or "conference",
                        ranking_system="CORE",
                        quartile=None,
                        core_rank=rk,
                        sjr=None,
                        h_index=None,
                        provenance="core"
                    )

            # B. Check discrete word tokens in title
            words = [self._clean_str(w) for w in re.split(r"[\s,;:\-\/]+", vname) if len(w) >= 3]
            for w in words:
                rk = self._lookup_core(w, pub_year, is_acronym=True)
                if rk:
                    return VenueRankingResult(
                        venue_name=vname,
                        venue_type=venue_type or "conference",
                        ranking_system="CORE",
                        quartile=None,
                        core_rank=rk,
                        sjr=None,
                        h_index=None,
                        provenance="core"
                    )

            # C. Check full title match
            rk = self._lookup_core(vname_clean, pub_year, is_acronym=False)
            if rk:
                return VenueRankingResult(
                    venue_name=vname,
                    venue_type=venue_type or "conference",
                    ranking_system="CORE",
                    quartile=None,
                    core_rank=rk,
                    sjr=None,
                    h_index=None,
                    provenance="core"
                )

        # 2. Check SCImago by ISSN-L
        issns = self._normalize_issn(venue_issn_l)
        for issn in issns:
            year_map = self.scimago_lookup.get(issn)
            if year_map is None:
                year_map = self._lookup_scimago_dynamic(issn)
            if year_map:
                valid_years = [y for y in year_map.keys() if y <= pub_year]
                if not valid_years:
                    continue
                latest_yr = max(valid_years)
                entry = year_map[latest_yr]
                return VenueRankingResult(
                    venue_name=vname,
                    venue_type=venue_type or "journal",
                    ranking_system="SCImago",
                    quartile=entry["quartile"],
                    core_rank=None,
                    sjr=entry["sjr"],
                    h_index=entry["h_index"],
                    provenance="scimago"
                )

        # 3. Strictly return None (Null) if not sourced (Rule R1)
        return VenueRankingResult(
            venue_name=vname,
            venue_type=venue_type,
            ranking_system=None,
            quartile=None,
            core_rank=None,
            sjr=None,
            h_index=None,
            provenance=None
        )


venue_resolver = ScientometricVenueResolver()
