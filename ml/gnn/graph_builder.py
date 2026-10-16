"""Heterogeneous Graph Construction for SciGraph AI using PyTorch Geometric.

Constructs an authentic HeteroData graph object from relational Parquet tables:
- Node types: paper, author, institution, venue, topic
- Edge types: cites, authored_by, affiliated_with, published_in, has_topic (and reverses)
- Features: strictly as-of-cutoff (T_cutoff = pub_year + 2), including early citations,
  temporal offset, venue SJR/tier, access status, APC, and title SVD embeddings.
- Temporal split masks:
  * train: <= 2016 (10,500 papers)
  * val:   == 2017 (1,500 papers)
  * test:  >= 2018 (4,500 papers)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import HeteroData

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from api.config import settings

logger = logging.getLogger("HeteroGraphBuilder")


def build_heterogeneous_graph(data_dir: Optional[Path] = None) -> HeteroData:
    data_dir = data_dir or settings.DATA_DIR
    logger.info("Loading relational tables for PyG HeteroData construction...")

    df_works = pd.read_parquet(data_dir / "works.parquet")
    df_authors = pd.read_parquet(data_dir / "authors.parquet")
    df_institutions = pd.read_parquet(data_dir / "institutions.parquet")
    df_authorships = pd.read_parquet(data_dir / "authorships.parquet")
    df_sources = pd.read_parquet(data_dir / "sources.parquet")
    df_topics = pd.read_parquet(data_dir / "topics.parquet")
    df_citation_edges = pd.read_parquet(data_dir / "citation_edges.parquet")

    logger.info(f"Loaded {len(df_works):,} works, {len(df_authors):,} authors, {len(df_institutions):,} institutions.")

    # 1. Map entity string IDs to continuous 0-indexed integers
    paper_id_map = {wid: idx for idx, wid in enumerate(df_works["work_id"])}
    author_id_map = {aid: idx for idx, aid in enumerate(df_authors["author_id"])}
    inst_id_map = {iid: idx for idx, iid in enumerate(df_institutions["institution_id"])}
    venue_id_map = {vid: idx for idx, vid in enumerate(df_sources["venue_id"])}
    topic_id_map = {tid: idx for idx, tid in enumerate(df_topics["topic_id"])}

    data = HeteroData()

    # 2. Construct Paper Node Features (strictly as-of-cutoff!)
    logger.info("Computing paper node features (strictly as-of-cutoff)...")
    # Title SVD embeddings (32-dim)
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_works["title"].fillna("Untitled"))
    svd = TruncatedSVD(n_components=32, random_state=42)
    title_embeddings = svd.fit_transform(tfidf_matrix)

    # Tabular cutoff features
    cits_cutoff = np.log1p(df_works["citations_at_cutoff"].clip(lower=0).values)[:, None]
    pub_offset = ((df_works["publication_year"].values - settings.MIN_PUB_YEAR) / 10.0)[:, None]
    ref_count = np.log1p(df_works["referenced_works_count"].clip(lower=0).values)[:, None]
    num_auth = (df_works["num_authors"].clip(lower=1, upper=50).values / 10.0)[:, None]
    is_oa = df_works["is_oa"].astype(float).values[:, None]

    # OA status one-hot
    oa_statuses = ["gold", "green", "hybrid", "bronze", "closed"]
    oa_one_hot = np.zeros((len(df_works), len(oa_statuses)), dtype=np.float32)
    for i, st in enumerate(oa_statuses):
        oa_one_hot[:, i] = (df_works["oa_status"].str.lower() == st).astype(float).values

    # APC and Venue metrics
    apc_paid = np.log1p(df_works["apc_paid_usd"].fillna(0.0).clip(lower=0).values)[:, None]
    sjr = df_works["venue_sjr"].fillna(0.0).values[:, None]
    is_top_tier = (
        (df_works["venue_scimago_quartile"] == "Q1") | 
        (df_works["venue_core_rank"].isin(["A*", "A"]))
    ).astype(float).values[:, None]

    paper_features = np.hstack([
        cits_cutoff,
        pub_offset,
        ref_count,
        num_auth,
        is_oa,
        oa_one_hot,
        apc_paid,
        sjr,
        is_top_tier,
        title_embeddings
    ]).astype(np.float32)

    data["paper"].x = torch.from_numpy(paper_features)
    data["paper"].y = torch.tensor(df_works["impact_class"].values, dtype=torch.long)
    data["paper"].y_reg = torch.tensor(np.log1p(df_works["future_citations_5y"].values), dtype=torch.float)

    # Split masks
    data["paper"].train_mask = torch.tensor((df_works["temporal_split"] == "train").values, dtype=torch.bool)
    data["paper"].val_mask = torch.tensor((df_works["temporal_split"] == "val").values, dtype=torch.bool)
    data["paper"].test_mask = torch.tensor((df_works["temporal_split"] == "test").values, dtype=torch.bool)

    # 3. Auxiliary Node Types (Vectorized)
    # Authors: 8-dim feature (log paper count)
    auth_counts = df_authorships["author_id"].value_counts()
    auth_features = np.zeros((len(df_authors), 8), dtype=np.float32)
    auth_features[:, 0] = np.log1p(df_authors["author_id"].map(auth_counts).fillna(1).values)
    data["author"].x = torch.from_numpy(auth_features)

    # Institutions: 8-dim feature (log affiliation count)
    inst_counts = df_authorships["institution_id"].dropna().value_counts()
    inst_features = np.zeros((len(df_institutions), 8), dtype=np.float32)
    inst_features[:, 0] = np.log1p(df_institutions["institution_id"].map(inst_counts).fillna(1).values)
    data["institution"].x = torch.from_numpy(inst_features)

    # Venues: 8-dim feature (DOAJ flag)
    venue_features = np.zeros((len(df_sources), 8), dtype=np.float32)
    venue_features[:, 0] = df_sources["is_in_doaj"].astype(float).values
    data["venue"].x = torch.from_numpy(venue_features)

    # Topics: 8-dim feature
    topic_features = np.zeros((len(df_topics), 8), dtype=np.float32)
    data["topic"].x = torch.from_numpy(topic_features)

    # 4. Construct Relational Edges (with bidirectional reverses)
    logger.info("Constructing heterogeneous edge indexes (vectorized)...")

    # (paper, cites, paper) internal citation edges
    s_mapped = df_citation_edges["citing_work_id"].map(paper_id_map)
    d_mapped = df_citation_edges["referenced_work_id"].map(paper_id_map)
    valid_cites = s_mapped.notnull() & d_mapped.notnull()
    cites_src = s_mapped[valid_cites].astype(int).values
    cites_dst = d_mapped[valid_cites].astype(int).values

    if len(cites_src) > 0:
        edge_cites = torch.from_numpy(np.vstack([cites_src, cites_dst])).long()
        edge_cited_by = torch.from_numpy(np.vstack([cites_dst, cites_src])).long()
    else:
        indices = np.arange(len(df_works))
        edge_cites = torch.from_numpy(np.vstack([indices, indices])).long()
        edge_cited_by = torch.from_numpy(np.vstack([indices, indices])).long()

    data["paper", "cites", "paper"].edge_index = edge_cites
    data["paper", "cited_by", "paper"].edge_index = edge_cited_by

    # (author, authored, paper) & reverse
    w_mapped = df_authorships["work_id"].map(paper_id_map)
    a_mapped = df_authorships["author_id"].map(author_id_map)
    valid_auth = w_mapped.notnull() & a_mapped.notnull()
    auth_src = a_mapped[valid_auth].astype(int).values
    auth_dst = w_mapped[valid_auth].astype(int).values

    data["author", "authored", "paper"].edge_index = torch.from_numpy(np.vstack([auth_src, auth_dst])).long()
    data["paper", "authored_by", "author"].edge_index = torch.from_numpy(np.vstack([auth_dst, auth_src])).long()

    # (author, affiliated_with, institution) & reverse
    df_inst_valid = df_authorships.dropna(subset=["institution_id"])
    a_inst_mapped = df_inst_valid["author_id"].map(author_id_map)
    i_mapped = df_inst_valid["institution_id"].map(inst_id_map)
    valid_inst = a_inst_mapped.notnull() & i_mapped.notnull()
    aff_src = a_inst_mapped[valid_inst].astype(int).values
    aff_dst = i_mapped[valid_inst].astype(int).values

    if len(aff_src) > 0:
        data["author", "affiliated_with", "institution"].edge_index = torch.from_numpy(np.vstack([aff_src, aff_dst])).long()
        data["institution", "affiliates", "author"].edge_index = torch.from_numpy(np.vstack([aff_dst, aff_src])).long()

    # (paper, published_in, venue) & reverse
    df_venue_valid = df_works.dropna(subset=["venue_id"])
    w_venue_mapped = df_venue_valid["work_id"].map(paper_id_map)
    v_mapped = df_venue_valid["venue_id"].map(venue_id_map)
    valid_venue = w_venue_mapped.notnull() & v_mapped.notnull()
    pub_src = w_venue_mapped[valid_venue].astype(int).values
    pub_dst = v_mapped[valid_venue].astype(int).values

    if len(pub_src) > 0:
        data["paper", "published_in", "venue"].edge_index = torch.from_numpy(np.vstack([pub_src, pub_dst])).long()
        data["venue", "publishes", "paper"].edge_index = torch.from_numpy(np.vstack([pub_dst, pub_src])).long()

    # (paper, has_topic, topic) & reverse
    df_topic_valid = df_works.dropna(subset=["primary_topic_id"])
    w_top_mapped = df_topic_valid["work_id"].map(paper_id_map)
    t_mapped = df_topic_valid["primary_topic_id"].map(topic_id_map)
    valid_top = w_top_mapped.notnull() & t_mapped.notnull()
    top_src = w_top_mapped[valid_top].astype(int).values
    top_dst = t_mapped[valid_top].astype(int).values

    if len(top_src) > 0:
        data["paper", "has_topic", "topic"].edge_index = torch.from_numpy(np.vstack([top_src, top_dst])).long()
        data["topic", "topic_of", "paper"].edge_index = torch.from_numpy(np.vstack([top_dst, top_src])).long()

    logger.info(f"PyG HeteroData assembled successfully: {data}")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    graph = build_heterogeneous_graph()
    print("Graph built:", graph)
