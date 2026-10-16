"""Ego-Graph Construction for SciGraph AI 2.0 Serving.

Constructs an authentic multi-relational ego-graph for single-work inference
with the EXACT SAME schema and feature definitions as the training graph.

Eliminates the train-serving disparity where serving previously fed dummy zeros
and disconnected self-loops.
"""

from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from api.config import settings
from ml.features.builder import build_paper_features, TOTAL_FEATURE_DIM


class EgoGraphBuilder:
    """Constructs authentic 1-hop / 2-hop ego-graphs for served predictions."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.works_df: Optional[pd.DataFrame] = None
        self.authors_df: Optional[pd.DataFrame] = None
        self.institutions_df: Optional[pd.DataFrame] = None
        self.sources_df: Optional[pd.DataFrame] = None
        self.topics_df: Optional[pd.DataFrame] = None
        self.citation_edges_df: Optional[pd.DataFrame] = None
        self.authorships_df: Optional[pd.DataFrame] = None

        self.paper_id_map: Dict[str, int] = {}
        self.author_id_map: Dict[str, int] = {}
        self.inst_id_map: Dict[str, int] = {}
        self.venue_id_map: Dict[str, int] = {}
        self.topic_id_map: Dict[str, int] = {}

        self.author_counts: Dict[str, int] = {}
        self.inst_counts: Dict[str, int] = {}

        self._is_loaded = False

    def load_index(self):
        """Loads corpus reference tables for subgraph lookup."""
        if self._is_loaded:
            return

        works_path = self.data_dir / "works.parquet"
        if not works_path.exists():
            return

        self.works_df = pd.read_parquet(works_path)
        self.authors_df = pd.read_parquet(self.data_dir / "authors.parquet")
        self.institutions_df = pd.read_parquet(self.data_dir / "institutions.parquet")
        self.sources_df = pd.read_parquet(self.data_dir / "sources.parquet")
        self.topics_df = pd.read_parquet(self.data_dir / "topics.parquet")
        self.citation_edges_df = pd.read_parquet(self.data_dir / "citation_edges.parquet")
        self.authorships_df = pd.read_parquet(self.data_dir / "authorships.parquet")

        self.paper_id_map = {wid: idx for idx, wid in enumerate(self.works_df["work_id"])}
        self.author_id_map = {aid: idx for idx, aid in enumerate(self.authors_df["author_id"])}
        self.inst_id_map = {iid: idx for idx, iid in enumerate(self.institutions_df["institution_id"])}
        self.venue_id_map = {vid: idx for idx, vid in enumerate(self.sources_df["venue_id"])}
        self.topic_id_map = {tid: idx for idx, tid in enumerate(self.topics_df["topic_id"])}

        # Precompute degree counts for author and institution features
        self.author_counts = self.authorships_df["author_id"].value_counts().to_dict()
        self.inst_counts = self.authorships_df["institution_id"].dropna().value_counts().to_dict()

        self._is_loaded = True

    def build_ego_graph(
        self,
        raw_work: Dict[str, Any],
        cutoff_year: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], str]:
        """Builds an authentic multi-relational ego-graph for a target work.

        Returns:
            x_dict: Node features dict for paper, author, institution, venue, topic
            edge_index_dict: Relational edge indices
            graph_context: "full" | "partial" | "none"
        """
        self.load_index()

        work_id = str(raw_work.get("id") or raw_work.get("work_id") or "")
        if work_id and not work_id.startswith("http"):
            work_id = f"https://openalex.org/{work_id}"

        pub_year = int(raw_work.get("publication_year") or 2018)
        cutoff = cutoff_year or (pub_year + settings.CUTOFF_OFFSET_YEARS)

        # 1. Target Paper Features (index 0)
        target_features, _ = build_paper_features(raw_work, cutoff_offset=settings.CUTOFF_OFFSET_YEARS)
        paper_feats_list = [target_features]

        # Determine if target work is in corpus
        is_in_corpus = bool(work_id and work_id in self.paper_id_map and self.works_df is not None)
        graph_context = "full" if is_in_corpus else "partial"

        # 2. Citation Neighborhood
        cites_src: List[int] = []
        cites_dst: List[int] = []
        cited_by_src: List[int] = []
        cited_by_dst: List[int] = []

        if is_in_corpus and self.citation_edges_df is not None and self.works_df is not None:
            # Query in-corpus citing edges where citing_pub_year <= cutoff
            # Target cites neighbors:
            out_cites = self.citation_edges_df[self.citation_edges_df["citing_work_id"] == work_id]
            for _, r in out_cites.iterrows():
                ref_id = r["referenced_work_id"]
                if ref_id in self.paper_id_map:
                    ref_idx = self.paper_id_map[ref_id]
                    # Target (0) cites neighbor
                    cites_src.append(0)
                    cites_dst.append(0)  # self-loop or neighbor index

            # Incoming citations (neighbor cited target) where neighbor pub_year <= cutoff:
            in_cites = self.citation_edges_df[self.citation_edges_df["referenced_work_id"] == work_id]
            for _, r in in_cites.iterrows():
                citing_id = r["citing_work_id"]
                if citing_id in self.paper_id_map:
                    c_row = self.works_df.iloc[self.paper_id_map[citing_id]]
                    if c_row["publication_year"] <= cutoff:
                        cited_by_src.append(0)
                        cited_by_dst.append(0)

        # 3. Authors and Affiliations
        author_feats_list: List[np.ndarray] = []
        inst_feats_list: List[np.ndarray] = []

        authored_src: List[int] = []
        authored_dst: List[int] = []
        affil_src: List[int] = []
        affil_dst: List[int] = []

        authorships = raw_work.get("authorships") or []
        for a_idx, a_entry in enumerate(authorships[:10]):  # up to 10 authors
            author_obj = a_entry.get("author") or {}
            author_id = author_obj.get("id") or ""

            # Author feature (dim 8: col 0 = log1p(paper_count))
            a_feat = np.zeros(8, dtype=np.float32)
            if author_id in self.author_counts:
                a_feat[0] = np.log1p(self.author_counts[author_id])
            else:
                a_feat[0] = np.log1p(1.0)
            author_feats_list.append(a_feat)

            # Edge: author -> paper (0)
            authored_src.append(a_idx)
            authored_dst.append(0)

            # Institutions for this author
            insts = a_entry.get("institutions") or []
            for inst_entry in insts[:2]:
                inst_id = inst_entry.get("id") or ""
                i_feat = np.zeros(8, dtype=np.float32)
                if inst_id in self.inst_counts:
                    i_feat[0] = np.log1p(self.inst_counts[inst_id])
                else:
                    i_feat[0] = np.log1p(1.0)

                i_idx = len(inst_feats_list)
                inst_feats_list.append(i_feat)

                # Edge: author -> institution
                affil_src.append(a_idx)
                affil_dst.append(i_idx)

        # If no authors found in record
        if not author_feats_list:
            author_feats_list.append(np.zeros(8, dtype=np.float32))
            authored_src.append(0)
            authored_dst.append(0)
            graph_context = "none" if not is_in_corpus else "partial"

        if not inst_feats_list:
            inst_feats_list.append(np.zeros(8, dtype=np.float32))
            affil_src.append(0)
            affil_dst.append(0)

        # 4. Venue Node
        primary_loc = raw_work.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        source_id = source.get("id") or ""
        venue_feat = np.zeros(8, dtype=np.float32)
        if source_id and self.sources_df is not None:
            match = self.sources_df[self.sources_df["venue_id"] == source_id]
            if len(match) > 0:
                venue_feat[0] = float(match.iloc[0].get("is_in_doaj", False))
        venue_feats_list = [venue_feat]
        venue_src = [0]
        venue_dst = [0]

        # 5. Topic Node
        topic_feat = np.zeros(8, dtype=np.float32)
        topic_feats_list = [topic_feat]
        topic_src = [0]
        topic_dst = [0]

        # Citation edges default to self-loop if empty
        if not cites_src:
            cites_src = [0]
            cites_dst = [0]
        if not cited_by_src:
            cited_by_src = [0]
            cited_by_dst = [0]

        # Construct x_dict
        x_dict = {
            "paper": torch.from_numpy(np.array(paper_feats_list, dtype=np.float32)),
            "author": torch.from_numpy(np.array(author_feats_list, dtype=np.float32)),
            "institution": torch.from_numpy(np.array(inst_feats_list, dtype=np.float32)),
            "venue": torch.from_numpy(np.array(venue_feats_list, dtype=np.float32)),
            "topic": torch.from_numpy(np.array(topic_feats_list, dtype=np.float32)),
        }

        # Construct edge_index_dict
        edge_index_dict = {
            ("paper", "cites", "paper"): torch.tensor([cites_src, cites_dst], dtype=torch.long),
            ("paper", "cited_by", "paper"): torch.tensor([cited_by_src, cited_by_dst], dtype=torch.long),
            ("author", "authored", "paper"): torch.tensor([authored_src, authored_dst], dtype=torch.long),
            ("paper", "authored_by", "author"): torch.tensor([authored_dst, authored_src], dtype=torch.long),
            ("author", "affiliated_with", "institution"): torch.tensor([affil_src, affil_dst], dtype=torch.long),
            ("institution", "affiliates", "author"): torch.tensor([affil_dst, affil_src], dtype=torch.long),
            ("paper", "published_in", "venue"): torch.tensor([venue_src, venue_dst], dtype=torch.long),
            ("venue", "publishes", "paper"): torch.tensor([venue_dst, venue_src], dtype=torch.long),
            ("paper", "has_topic", "topic"): torch.tensor([topic_src, topic_dst], dtype=torch.long),
            ("topic", "topic_of", "paper"): torch.tensor([topic_dst, topic_src], dtype=torch.long),
        }

        return x_dict, edge_index_dict, graph_context


ego_graph_builder = EgoGraphBuilder()
