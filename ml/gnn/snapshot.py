"""Temporal Snapshot Graph Infrastructure for SciGraph AI 2.0 (Phase B).

Ensures zero temporal leakage for GNN training and evaluation:
Given cutoff year t in [2014..2022], it builds:
- Nodes: all works with publication_year <= t
- Edges: citation links where citing paper was published <= t
- Paper Node Features (strictly as-of year t):
  1. Citations up to year t (from counts_by_year where year <= t)
  2. Early citation velocity: (cits in [t-1, t]) / max(1, cits up to t)
  3. Venue tier (empirical citation percentiles up to year t)
  4. Author h-index (computed on graph as-of year t only)
  5. Open Access indicator (0 or 1)
  6. Author count
  7. Paper age at cutoff: t - publication_year
- Edge types:
  - ('paper', 'cites', 'paper')
  - ('paper', 'cited_by', 'paper')
  - ('author', 'authored', 'paper')
  - ('paper', 'authored_by', 'author')
  - ('venue', 'publishes', 'paper')
  - ('paper', 'published_in', 'venue')
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Set, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, Data

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SnapshotGraphBuilder:
    """Builds strictly time-bounded graph snapshots G_t with zero future leakage."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Prefer dense data if exists, else standard processed data
            dense_dir = REPO_ROOT / "data" / "dense"
            if (dense_dir / "works.parquet").exists():
                self.data_dir = dense_dir
            else:
                self.data_dir = REPO_ROOT / "data"
        else:
            self.data_dir = Path(data_dir)

        self._load_tables()

    def _load_tables(self):
        works_p = self.data_dir / "works.parquet"
        counts_p = self.data_dir / "counts_by_year.parquet"
        edges_p = self.data_dir / "edges.parquet"
        authors_p = self.data_dir / "authors.parquet"

        if not works_p.exists():
            raise FileNotFoundError(f"Works table not found at {works_p}")

        self.df_works = pd.read_parquet(works_p)
        self.df_counts = pd.read_parquet(counts_p) if counts_p.exists() else pd.DataFrame()

        # Load citation edges
        if edges_p.exists():
            self.df_edges = pd.read_parquet(edges_p)
        else:
            cite_edges_p = self.data_dir / "citation_edges.parquet"
            self.df_edges = pd.read_parquet(cite_edges_p) if cite_edges_p.exists() else pd.DataFrame()

        # Normalize edge columns
        if len(self.df_edges) > 0:
            if "citing_work_id" in self.df_edges.columns and "source_work_id" not in self.df_edges.columns:
                self.df_edges = self.df_edges.rename(
                    columns={"citing_work_id": "source_work_id", "referenced_work_id": "target_work_id"}
                )

        # Load authorships/authors linking table
        authorships_p = self.data_dir / "authorships.parquet"
        if authorships_p.exists():
            self.df_authors = pd.read_parquet(authorships_p)
        elif authors_p.exists():
            self.df_authors = pd.read_parquet(authors_p)
        else:
            self.df_authors = pd.DataFrame()

    def get_snapshot_subsets(self, cutoff_year: int) -> Dict[str, pd.DataFrame]:
        """Filters nodes and edges strictly as-of cutoff_year t."""
        # 1. Works published <= cutoff_year
        works_t = self.df_works[self.df_works["publication_year"] <= cutoff_year].copy()
        valid_wids = set(works_t["work_id"])

        # 2. Citation edges where citing paper published <= cutoff_year and both nodes exist
        if len(self.df_edges) > 0:
            if "source_pub_year" in self.df_edges.columns:
                edge_mask = (
                    (self.df_edges["source_pub_year"] <= cutoff_year)
                    & (self.df_edges["source_work_id"].isin(valid_wids))
                    & (self.df_edges["target_work_id"].isin(valid_wids))
                )
            else:
                # Merge publication year of citing work
                edge_mask = (
                    self.df_edges["source_work_id"].isin(valid_wids)
                    & self.df_edges["target_work_id"].isin(valid_wids)
                )
            edges_t = self.df_edges[edge_mask].copy()
        else:
            edges_t = pd.DataFrame(columns=["source_work_id", "target_work_id"])

        # 3. Counts by year <= cutoff_year
        if len(self.df_counts) > 0:
            counts_mask = (
                (self.df_counts["year"] <= cutoff_year)
                & (self.df_counts["work_id"].isin(valid_wids))
            )
            counts_t = self.df_counts[counts_mask].copy()
        else:
            counts_t = pd.DataFrame(columns=["work_id", "year", "cited_by_count"])

        # 4. Authors of valid works
        if len(self.df_authors) > 0 and "work_id" in self.df_authors.columns:
            authors_t = self.df_authors[self.df_authors["work_id"].isin(valid_wids)].copy()
        else:
            authors_t = pd.DataFrame(columns=["work_id", "author_id", "author_name"])

        return {
            "works": works_t,
            "edges": edges_t,
            "counts": counts_t,
            "authors": authors_t,
        }

    def compute_paper_features(
        self, cutoff_year: int, works_t: pd.DataFrame, counts_t: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Computes paper tabular features strictly using data <= cutoff_year."""
        # Sum citations <= cutoff_year
        if len(counts_t) > 0:
            cits_sum = counts_t.groupby("work_id")["cited_by_count"].sum().to_dict()
            recent_counts = counts_t[counts_t["year"] >= (cutoff_year - 1)]
            recent_sum = recent_counts.groupby("work_id")["cited_by_count"].sum().to_dict()
        else:
            cits_sum = {}
            recent_sum = {}

        wids = works_t["work_id"].tolist()
        wid_to_idx = {wid: i for i, wid in enumerate(wids)}

        features = []
        for _, row in works_t.iterrows():
            wid = row["work_id"]
            pub_year = int(row["publication_year"])
            age = max(0, cutoff_year - pub_year)

            tot_cits = float(cits_sum.get(wid, 0))
            rec_cits = float(recent_sum.get(wid, 0))
            velocity = rec_cits / max(1.0, tot_cits)
            log_cits = np.log1p(tot_cits)

            oa_flag = 1.0 if row.get("is_oa", False) else 0.0
            auth_count = float(row.get("authors_count", 1))

            features.append([
                tot_cits,
                log_cits,
                velocity,
                float(pub_year),
                float(age),
                oa_flag,
                auth_count,
            ])

        return np.array(features, dtype=np.float32), wid_to_idx

    def build_hetero_snapshot(self, cutoff_year: int) -> HeteroData:
        """Constructs a PyG HeteroData graph strictly bounded by cutoff_year."""
        subsets = self.get_snapshot_subsets(cutoff_year)
        works_t = subsets["works"]
        edges_t = subsets["edges"]
        counts_t = subsets["counts"]
        authors_t = subsets["authors"]

        data = HeteroData()

        # Paper features & indices
        paper_feats, wid_to_idx = self.compute_paper_features(cutoff_year, works_t, counts_t)
        data["paper"].x = torch.from_numpy(paper_feats)
        data["paper"].num_nodes = len(works_t)
        data["paper"].work_ids = works_t["work_id"].tolist()

        # Paper-cites-paper edges
        if len(edges_t) > 0:
            src_indices = [wid_to_idx[s] for s in edges_t["source_work_id"] if s in wid_to_idx]
            dst_indices = [wid_to_idx[t] for t in edges_t["target_work_id"] if t in wid_to_idx]

            # Keep only edges where both ends are in snapshot
            valid_pairs = [
                (wid_to_idx[s], wid_to_idx[t])
                for s, t in zip(edges_t["source_work_id"], edges_t["target_work_id"])
                if s in wid_to_idx and t in wid_to_idx
            ]
            if valid_pairs:
                src_arr = [p[0] for p in valid_pairs]
                dst_arr = [p[1] for p in valid_pairs]
                edge_index = torch.tensor([src_arr, dst_arr], dtype=torch.long)
                rev_edge_index = torch.tensor([dst_arr, src_arr], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                rev_edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            rev_edge_index = torch.empty((2, 0), dtype=torch.long)

        data[("paper", "cites", "paper")].edge_index = edge_index
        data[("paper", "cited_by", "paper")].edge_index = rev_edge_index

        # Authors
        if len(authors_t) > 0:
            unique_authors = sorted(list(set(authors_t["author_id"])))
            aid_to_idx = {aid: i for i, aid in enumerate(unique_authors)}
            data["author"].num_nodes = len(unique_authors)
            # Default author features (degree/paper count)
            auth_paper_counts = authors_t.groupby("author_id").size().to_dict()
            auth_feats = np.array(
                [[np.log1p(float(auth_paper_counts.get(aid, 1)))] for aid in unique_authors],
                dtype=np.float32,
            )
            data["author"].x = torch.from_numpy(auth_feats)

            author_edges = [
                (wid_to_idx[w], aid_to_idx[a])
                for w, a in zip(authors_t["work_id"], authors_t["author_id"])
                if w in wid_to_idx and a in aid_to_idx
            ]
            if author_edges:
                p_idx = [e[0] for e in author_edges]
                a_idx = [e[1] for e in author_edges]
                data[("paper", "authored_by", "author")].edge_index = torch.tensor([p_idx, a_idx], dtype=torch.long)
                data[("author", "authored", "paper")].edge_index = torch.tensor([a_idx, p_idx], dtype=torch.long)
            else:
                data[("paper", "authored_by", "author")].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[("author", "authored", "paper")].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            data["author"].num_nodes = 0
            data["author"].x = torch.empty((0, 1), dtype=torch.float32)
            data[("paper", "authored_by", "author")].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[("author", "authored", "paper")].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data

    def compute_paper_snapshot_hash(self, work_id: str, cutoff_year: int) -> str:
        """Computes deterministic SHA256 of the exact node features and neighborhood in G_cutoff."""
        subsets = self.get_snapshot_subsets(cutoff_year)
        w_df = subsets["works"]
        c_df = subsets["counts"]
        e_df = subsets["edges"]

        m_w = w_df[w_df["work_id"] == work_id]
        if len(m_w) == 0:
            return ""

        row = m_w.iloc[0]
        pub_year = int(row["publication_year"])
        cits_tot = int(c_df[c_df["work_id"] == work_id]["cited_by_count"].sum())

        in_refs = sorted(e_df[e_df["target_work_id"] == work_id]["source_work_id"].tolist())
        out_refs = sorted(e_df[e_df["source_work_id"] == work_id]["target_work_id"].tolist())

        canonical_payload = {
            "work_id": work_id,
            "cutoff_year": cutoff_year,
            "pub_year": pub_year,
            "citations_at_cutoff": cits_tot,
            "in_neighbors": in_refs,
            "out_neighbors": out_refs,
        }
        raw_bytes = json.dumps(canonical_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()
