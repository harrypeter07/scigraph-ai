"""Phase 7 Unit Tests - Heterogeneous Graph Construction & Schema Verification."""

import pytest
import pandas as pd
from ml.graph.build_graph import HeteroGraphBuilder


def test_hetero_graph_construction(tmp_path):
    """Test HeteroGraphBuilder node and edge index creation from mock Parquets."""
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    interim_dir.mkdir()
    processed_dir.mkdir()

    papers_df = pd.DataFrame([
        {"id": "W1", "publication_year": 2018},
        {"id": "W2", "publication_year": 2019}
    ])
    authors_df = pd.DataFrame([{"id": "A1", "display_name": "Author 1"}])
    inst_df = pd.DataFrame([{"id": "I1", "display_name": "Inst 1", "country_code": "US", "type": "Edu"}])
    auth_df = pd.DataFrame([{"author_id": "A1", "paper_id": "W1", "institution_id": "I1", "author_position": 0}])
    cit_df = pd.DataFrame([{"citing_paper_id": "W2", "cited_paper_id": "W1", "citation_year": 2019}])

    papers_df.to_parquet(processed_dir / "labeled_papers.parquet", index=False)
    authors_df.to_parquet(interim_dir / "authors.parquet", index=False)
    inst_df.to_parquet(interim_dir / "institutions.parquet", index=False)
    auth_df.to_parquet(interim_dir / "authorships.parquet", index=False)
    cit_df.to_parquet(interim_dir / "paper_citations.parquet", index=False)

    builder = HeteroGraphBuilder(interim_dir=str(interim_dir), processed_dir=str(processed_dir))
    res = builder.build_time_consistent_graph()

    stats = res["stats"]
    assert stats["num_paper_nodes"] == 2
    assert stats["num_author_nodes"] == 1
    assert stats["num_writes_edges"] == 1
    assert stats["num_cites_edges"] == 1
