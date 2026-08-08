"""Phase 1/2 Tests - OpenAlex Acquisition Client & Record Schema Parsing."""

import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from ml.acquisition.openalex import (
    parse_abstract_inverted_index,
    parse_openalex_record,
    OpenAlexAcquisitionClient
)


def test_parse_abstract_inverted_index():
    """Verify reconstruction of abstract text from OpenAlex inverted index."""
    inverted_index = {
        "Graph": [0],
        "Neural": [1],
        "Networks": [2],
        "for": [3],
        "Citation": [4],
        "Prediction": [5]
    }
    reconstructed = parse_abstract_inverted_index(inverted_index)
    assert reconstructed == "Graph Neural Networks for Citation Prediction"


def test_parse_openalex_record_schema():
    """Verify parsed OpenAlex record complies with DATA_DICTIONARY.md structure."""
    raw_work = {
        "id": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.1145/3292500.3330961",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "publication_year": 2017,
        "publication_date": "2017-01-16",
        "abstract_inverted_index": {"Graph": [0], "Convolutions": [1]},
        "cited_by_count": 15000,  # Retained in raw, NOT used in feature engineering
        "counts_by_year": [
            {"year": 2017, "cited_by_count": 100},
            {"year": 2018, "cited_by_count": 1500},
            {"year": 2019, "cited_by_count": 3000}
        ],
        "referenced_works": ["https://openalex.org/W111", "https://openalex.org/W222"],
        "primary_topic": {
            "id": "https://openalex.org/T10001",
            "display_name": "Graph Neural Networks"
        },
        "concepts": [
            {"id": "https://openalex.org/C41008148", "display_name": "Artificial Intelligence", "level": 1, "score": 0.95}
        ],
        "authorships": [
            {
                "author": {"id": "https://openalex.org/A111", "display_name": "Thomas N. Kipf"},
                "institutions": [{"id": "https://openalex.org/I111", "display_name": "University of Amsterdam", "country_code": "NL"}]
            }
        ]
    }

    parsed = parse_openalex_record(raw_work)

    # Validate paper dictionary schema
    paper = parsed["paper"]
    assert paper["id"] == "https://openalex.org/W2741809807"
    assert paper["publication_year"] == 2017
    assert paper["abstract_text"] == "Graph Convolutions"
    assert len(paper["counts_by_year"]) == 3
    assert paper["referenced_works_count"] == 2
    assert "referenced_works" in paper

    # Validate authors & institutions
    assert len(parsed["authors"]) == 1
    assert parsed["authors"][0]["display_name"] == "Thomas N. Kipf"
    assert len(parsed["institutions"]) == 1
    assert parsed["institutions"][0]["country_code"] == "NL"
    assert len(parsed["authorships"]) == 1


@patch("ml.acquisition.openalex.requests.get")
def test_acquisition_client_mock_run(mock_get, tmp_path):
    """Test acquisition client batch fetching and JSONL writing with mocked API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "meta": {"next_cursor": "next_cursor_token_123"},
        "results": [
            {
                "id": "https://openalex.org/W001",
                "title": "Mock Paper 1",
                "publication_year": 2018,
                "abstract_inverted_index": {"Mock": [0]},
                "counts_by_year": [{"year": 2018, "cited_by_count": 2}],
                "referenced_works": []
            }
        ]
    }
    mock_get.return_value = mock_response

    # Create temporary config
    config_data = {
        "acquisition": {
            "concept_filter": {"value": "C41008148"},
            "year_range": {"min_year": 2012, "max_year": 2022},
            "sample_size": 1,
            "rate_limit_delay_seconds": 0.0,
            "batch_size": 1
        },
        "paths": {
            "raw_dir": str(tmp_path / "data" / "raw" / "openalex")
        }
    }
    config_file = tmp_path / "test_dataset.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml_str = "\n".join([f"{k}: {json.dumps(v)}" for k, v in config_data.items()])
        f.write(yaml_str)

    client = OpenAlexAcquisitionClient(config_path=str(config_file))
    total_ingested = client.run_acquisition(max_records=1)

    assert total_ingested == 1
    state_file = tmp_path / "data" / "raw" / "openalex" / "ingestion_state.json"
    assert state_file.exists()

    with open(state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
        assert state["total_ingested"] == 1
