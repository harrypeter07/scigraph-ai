"""Phase 3 Unit Tests - Preprocessing & Interim Parquet Generation."""

import os
import pytest
import pandas as pd
from ml.preprocessing.cleaner import OpenAlexPreprocessor


def test_preprocessing_pipeline(tmp_path):
    """Test loading, processing, and Parquet export of mock raw records."""
    mock_raw_records = [
        {
            "id": "https://openalex.org/W100",
            "title": "Deep Learning Overview",
            "publication_year": 2018,
            "publication_date": "2018-05-10",
            "abstract_inverted_index": {"Deep": [0], "Learning": [1]},
            "cited_by_count": 500,
            "counts_by_year": [{"year": 2018, "cited_by_count": 50}, {"year": 2019, "cited_by_count": 150}],
            "referenced_works": ["https://openalex.org/W99"],
            "primary_topic": {"id": "https://openalex.org/T001", "display_name": "Deep Learning"},
            "concepts": [{"id": "C41008148", "display_name": "AI", "level": 1, "score": 0.9}],
            "authorships": [
                {
                    "author": {"id": "A1", "display_name": "Alice Smith"},
                    "institutions": [{"id": "I1", "display_name": "MIT", "country_code": "US"}]
                }
            ]
        }
    ]

    preprocessor = OpenAlexPreprocessor(raw_dir=str(tmp_path / "raw"), interim_dir=str(tmp_path / "interim"))
    papers_df, authors_df, institutions_df, authorships_df, citations_df = preprocessor.process_records(mock_raw_records)

    assert len(papers_df) == 1
    assert papers_df.iloc[0]["id"] == "https://openalex.org/W100"
    assert papers_df.iloc[0]["publication_year"] == 2018
    assert len(authors_df) == 1
    assert authors_df.iloc[0]["display_name"] == "Alice Smith"
    assert len(institutions_df) == 1
    assert institutions_df.iloc[0]["country_code"] == "US"
    assert len(authorships_df) == 1
    assert len(citations_df) == 1
