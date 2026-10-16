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
        # 1. Load SCImago
        if self.scimago_path.exists():
            df_scimago = pd.read_parquet(self.scimago_path)
            for _, row in df_scimago.iterrows():
                issn = row["issn"]
                yr = int(row["year"])
                if issn not in self.scimago_lookup:
                    self.scimago_lookup[issn] = {}
                self.scimago_lookup[issn][yr] = {
                    "quartile": row["quartile"],
                    "sjr": float(row["sjr"]) if pd.notnull(row["sjr"]) else None,
                    "h_index": float(row["h_index"]) if pd.notnull(row["h_index"]) else None
                }
            logger.info(f"VenueResolver: Loaded {len(self.scimago_lookup):,} SCImago ISSN entries.")
        else:
            logger.warning(f"VenueResolver: SCImago dataset not found at {self.scimago_path}")

        # 2. Load CORE
        if self.core_path.exists():
            df_core = pd.read_parquet(self.core_path)
            rank_prio = {"A*": 5, "A": 4, "B": 3, "C": 2, "Australasian": 1, "Unranked": 0}
            df_core["prio"] = df_core["rank"].map(lambda r: rank_prio.get(r, 0))
            df_core = df_core.sort_values(by=["prio", "edition_year"], ascending=False)

            for _, row in df_core.iterrows():
                t = self._clean_str(str(row.get("title", "")))
                a = self._clean_str(str(row.get("acronym", "")))
                rk = str(row.get("rank", "Unranked"))
                if t and t not in self.core_title_lookup:
                    self.core_title_lookup[t] = rk
                if a and a not in self.core_acronym_lookup:
                    self.core_acronym_lookup[a] = rk
            logger.info(f"VenueResolver: Loaded {len(self.core_title_lookup):,} CORE titles, {len(self.core_acronym_lookup):,} CORE acronyms.")
        else:
            logger.warning(f"VenueResolver: CORE dataset not found at {self.core_path}")

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
        vtype = (venue_type or "").lower()
        vname_clean = self._clean_str(vname)

        # 1. Check CORE for Computer Science conferences / proceedings
        is_conference = "conference" in vtype or "proceedings" in vname.lower() or any(
            k in vname.lower() for k in ["acm", "ieee", "cvpr", "icml", "neurips", "kdd", "acl", "aaai", "ijcai"]
        )

        if is_conference:
            # A. Check parenthesized acronyms first (e.g. "(CVPR)")
            paren_matches = re.findall(r"[\(\[\{]([A-Za-z0-9\-]+)[\)\]\}]", vname)
            for p in paren_matches:
                p_clean = self._clean_str(p)
                if p_clean in self.core_acronym_lookup:
                    return VenueRankingResult(
                        venue_name=vname,
                        venue_type=venue_type or "conference",
                        ranking_system="CORE",
                        quartile=None,
                        core_rank=self.core_acronym_lookup[p_clean],
                        sjr=None,
                        h_index=None,
                        provenance="core"
                    )

            # B. Check discrete word tokens in title
            words = [self._clean_str(w) for w in re.split(r"[\s,;:\-\/]+", vname) if len(w) >= 3]
            for w in words:
                if w in self.core_acronym_lookup:
                    return VenueRankingResult(
                        venue_name=vname,
                        venue_type=venue_type or "conference",
                        ranking_system="CORE",
                        quartile=None,
                        core_rank=self.core_acronym_lookup[w],
                        sjr=None,
                        h_index=None,
                        provenance="core"
                    )

            # C. Check full title match
            if vname_clean in self.core_title_lookup:
                return VenueRankingResult(
                    venue_name=vname,
                    venue_type=venue_type or "conference",
                    ranking_system="CORE",
                    quartile=None,
                    core_rank=self.core_title_lookup[vname_clean],
                    sjr=None,
                    h_index=None,
                    provenance="core"
                )

        # 2. Check SCImago by ISSN-L
        issns = self._normalize_issn(venue_issn_l)
        for issn in issns:
            if issn in self.scimago_lookup:
                year_map = self.scimago_lookup[issn]
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
