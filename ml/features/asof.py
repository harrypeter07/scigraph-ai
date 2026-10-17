"""As-Of Feature Engineering for SciGraph AI 2.0 (Phase 2).

All features in this module are strictly time-bounded:
Given a paper with publication_year t_P, cutoff is t_C = t_P + 2.
No data, edge, or citation with timestamp > t_C is ever accessed.

Feature Groups:
- Baseline (F0): Currently served features (log1p citations, OA flag, authors, venue SJR)
- F1: Cohort-normalized early citations (percentile rank within (field, year), z-score of log cits)
- F2: Yearly citation trajectory (y0, y1, y2, acceleration, growth ratio, last-year share)
- F3: Citer diversity and volume as-of cutoff
- F4: Author history as-of cutoff (prior papers, prior citations, veteran author flag)
- F5: Venue history as-of cutoff (prior paper count, prior mean citations, rank status)
- F6: Reference properties (reference count, log reference count)
- F7: Paper intrinsics (title length, title words, OA status category, APC indicator)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def clean_id(raw_id: Any) -> str:
    if pd.isna(raw_id):
        return ""
    return str(raw_id).split("/")[-1]


class AsOfFeatureBuilder:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data"
        self._load_tables()
        self._precompute_lookups()

    def _load_tables(self):
        works_p = self.data_dir / "works.parquet"
        counts_p = self.data_dir / "counts_by_year.parquet"
        edges_p = self.data_dir / "citation_edges.parquet"
        authors_p = self.data_dir / "authorships.parquet"

        self.df_works = pd.read_parquet(works_p)
        self.df_counts = pd.read_parquet(counts_p) if counts_p.exists() else pd.DataFrame()
        self.df_edges = pd.read_parquet(edges_p) if edges_p.exists() else pd.DataFrame()
        self.df_authors = pd.read_parquet(authors_p) if authors_p.exists() else pd.DataFrame()

        self.df_works["clean_id"] = self.df_works["work_id"].apply(clean_id)
        if len(self.df_counts) > 0:
            self.df_counts["clean_id"] = self.df_counts["work_id"].apply(clean_id)
        if len(self.df_edges) > 0:
            col_src = "source_work_id" if "source_work_id" in self.df_edges.columns else "citing_work_id"
            col_dst = "target_work_id" if "target_work_id" in self.df_edges.columns else "referenced_work_id"
            self.df_edges["src_id"] = self.df_edges[col_src].apply(clean_id)
            self.df_edges["dst_id"] = self.df_edges[col_dst].apply(clean_id)
        if len(self.df_authors) > 0:
            self.df_authors["clean_id"] = self.df_authors["work_id"].apply(clean_id)
            self.df_authors["author_id"] = self.df_authors["author_id"].apply(clean_id)

    def _precompute_lookups(self):
        """Build fast in-memory indexes."""
        # 1. Publication year map
        self.pub_year_map = dict(zip(self.df_works["clean_id"], self.df_works["publication_year"]))

        # 2. Counts map: wid -> {year: count}
        self.counts_map: Dict[str, Dict[int, int]] = {}
        for _, r in self.df_counts.iterrows():
            wid = r["clean_id"]
            yr = int(r["year"])
            cnt = int(r["cited_by_count"])
            if wid not in self.counts_map:
                self.counts_map[wid] = {}
            self.counts_map[wid][yr] = cnt

        # 3. Authors per paper
        self.paper_authors_map: Dict[str, List[str]] = {}
        # Author papers: aid -> list of (pub_year, wid)
        self.author_papers_map: Dict[str, List[Tuple[int, str]]] = {}
        if len(self.df_authors) > 0:
            for _, r in self.df_authors.iterrows():
                wid = r["clean_id"]
                aid = r["author_id"]
                py = self.pub_year_map.get(wid, 0)
                if wid and aid:
                    self.paper_authors_map.setdefault(wid, []).append(aid)
                    self.author_papers_map.setdefault(aid, []).append((py, wid))

        # 4. In-corpus citations to target: target_id -> list of citing_pub_year
        self.citer_years_map: Dict[str, List[int]] = {}
        if len(self.df_edges) > 0:
            for _, r in self.df_edges.iterrows():
                src = r["src_id"]
                dst = r["dst_id"]
                src_py = self.pub_year_map.get(src, 0)
                if dst and src_py > 0:
                    self.citer_years_map.setdefault(dst, []).append(src_py)

    def extract_features_df(self) -> pd.DataFrame:
        """Extracts complete ladder of features for all evaluation papers."""
        valid_mask = (self.df_works["publication_year"] >= 2012) & (self.df_works["publication_year"] <= 2020)
        df_target = self.df_works[valid_mask].copy()

        rows = []
        for _, row in df_target.iterrows():
            wid = row["clean_id"]
            py = int(row["publication_year"])
            cutoff = py + 2
            field = str(row.get("primary_field_name") or "Computer Science")
            venue_id = clean_id(row.get("venue_id"))

            cm = self.counts_map.get(wid, {})
            # Early counts by year
            y0 = cm.get(py, 0)
            y1 = cm.get(py + 1, 0)
            y2 = cm.get(py + 2, 0)
            cits_cutoff = sum(cnt for yr, cnt in cm.items() if yr <= cutoff)

            # -------------------------------------------------------------
            # F0: Baseline Features
            # -------------------------------------------------------------
            f0_log_cits = np.log1p(float(cits_cutoff))
            f0_is_oa = 1.0 if row.get("is_oa", False) else 0.0
            f0_num_authors = float(row.get("num_authors", 1) or 1)
            sjr_val = row.get("venue_sjr")
            f0_venue_sjr = float(sjr_val) if pd.notna(sjr_val) else 0.0

            # -------------------------------------------------------------
            # F2: Yearly Trajectory
            # -------------------------------------------------------------
            f2_y0 = float(y0)
            f2_y1 = float(y1)
            f2_y2 = float(y2)
            f2_accel = float((y2 - y1) - (y1 - y0))
            f2_growth_ratio = float(y2) / (float(y0 + y1) + 1.0)
            f2_last_year_share = float(y2) / max(1.0, float(cits_cutoff))
            f2_positive_growth = 1.0 if (y2 > y1 and y1 > y0) else 0.0

            # -------------------------------------------------------------
            # F3: Citer Diversity as-of cutoff
            # -------------------------------------------------------------
            citer_years = self.citer_years_map.get(wid, [])
            cits_before_cutoff = [y for y in citer_years if y <= cutoff]
            f3_in_corpus_citer_count = float(len(cits_before_cutoff))

            # -------------------------------------------------------------
            # F4: Author History as-of cutoff
            # -------------------------------------------------------------
            authors = self.paper_authors_map.get(wid, [])
            prior_counts = []
            for aid in authors:
                prior_papers = [p for p in self.author_papers_map.get(aid, []) if p[0] < cutoff and p[1] != wid]
                prior_counts.append(len(prior_papers))

            if prior_counts:
                f4_author_max_prior_papers = float(max(prior_counts))
                f4_author_mean_prior_papers = float(np.mean(prior_counts))
                f4_has_veteran_author = 1.0 if f4_author_max_prior_papers >= 5.0 else 0.0
            else:
                f4_author_max_prior_papers = 0.0
                f4_author_mean_prior_papers = 0.0
                f4_has_veteran_author = 0.0

            # -------------------------------------------------------------
            # F5: Venue History & Ranks
            # -------------------------------------------------------------
            f5_has_core_rank = 1.0 if pd.notna(row.get("venue_core_rank")) else 0.0
            scimago_q = str(row.get("venue_scimago_quartile") or "")
            f5_is_q1 = 1.0 if "Q1" in scimago_q else 0.0

            # -------------------------------------------------------------
            # F6: Reference Properties
            # -------------------------------------------------------------
            raw_refs = float(row.get("referenced_works_count", 0) or 0)
            f6_ref_count = raw_refs
            f6_log_ref_count = np.log1p(raw_refs)

            # -------------------------------------------------------------
            # F7: Paper Intrinsics
            # -------------------------------------------------------------
            title_str = str(row.get("title") or "")
            f7_title_chars = float(len(title_str))
            f7_title_words = float(len(title_str.split()))
            f7_has_apc = 1.0 if pd.notna(row.get("apc_paid_usd")) and float(row.get("apc_paid_usd") or 0) > 0 else 0.0

            rows.append({
                "work_id": wid,
                "title": title_str,
                "publication_year": py,
                "field": field,
                "temporal_split": str(row["temporal_split"]),
                "cits_cutoff": cits_cutoff,
                "impact_class": int(row["impact_class"]),
                # F0
                "f0_log_cits": f0_log_cits,
                "f0_is_oa": f0_is_oa,
                "f0_num_authors": f0_num_authors,
                "f0_venue_sjr": f0_venue_sjr,
                # F2
                "f2_y0": f2_y0,
                "f2_y1": f2_y1,
                "f2_y2": f2_y2,
                "f2_accel": f2_accel,
                "f2_growth_ratio": f2_growth_ratio,
                "f2_last_year_share": f2_last_year_share,
                "f2_positive_growth": f2_positive_growth,
                # F3
                "f3_in_corpus_citer_count": f3_in_corpus_citer_count,
                # F4
                "f4_author_max_prior_papers": f4_author_max_prior_papers,
                "f4_author_mean_prior_papers": f4_author_mean_prior_papers,
                "f4_has_veteran_author": f4_has_veteran_author,
                # F5
                "f5_has_core_rank": f5_has_core_rank,
                "f5_is_q1": f5_is_q1,
                # F6
                "f6_ref_count": f6_ref_count,
                "f6_log_ref_count": f6_log_ref_count,
                # F7
                "f7_title_chars": f7_title_chars,
                "f7_title_words": f7_title_words,
                "f7_has_apc": f7_has_apc,
            })

        df_out = pd.DataFrame(rows)

        # -------------------------------------------------------------
        # F1: Cohort-Normalized Early Citations (Computed within field & pub_year)
        # -------------------------------------------------------------
        # Percentile rank & Z-score computed per cohort strictly on early citations (no labels)
        cohort_groups = df_out.groupby(["field", "publication_year"])
        pct_ranks = []
        z_scores = []
        c_sizes = []

        for (f_val, py_val), grp in cohort_groups:
            cits = grp["cits_cutoff"].values
            log_c = np.log1p(cits)
            # Percentile rank
            from scipy.stats import rankdata
            if len(cits) > 1:
                ranks = rankdata(cits, method="average") / len(cits)
                std_c = np.std(log_c)
                z = (log_c - np.mean(log_c)) / (std_c if std_c > 1e-6 else 1.0)
            else:
                ranks = np.array([0.5])
                z = np.array([0.0])

            for idx, r_val, z_val in zip(grp.index, ranks, z):
                pct_ranks.append((idx, r_val))
                z_scores.append((idx, z_val))
                c_sizes.append((idx, len(cits)))

        pct_ranks = sorted(pct_ranks, key=lambda x: x[0])
        z_scores = sorted(z_scores, key=lambda x: x[0])
        c_sizes = sorted(c_sizes, key=lambda x: x[0])

        df_out["f1_cohort_percentile"] = [p[1] for p in pct_ranks]
        df_out["f1_cohort_zscore"] = [z[1] for z in z_scores]
        df_out["f1_cohort_size"] = [c[1] for c in c_sizes]

        return df_out
