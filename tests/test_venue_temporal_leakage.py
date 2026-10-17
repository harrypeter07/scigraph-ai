"""Temporal Leakage Unit Tests for SCImago and CORE Venue Resolution.

Verifies:
1. SCImago lookup strictly uses editions where year <= pub_year (no future edition leakage).
2. CORE conference lookup strictly uses editions where edition_year <= pub_year.
3. When no historical edition exists <= pub_year, returns None/unknown (-1.0, known_flag=0.0).
4. Querying a paper published in year Y never accesses scientometric metrics published after Y.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.scientometrics.venue_resolver import ScientometricVenueResolver, VenueRankingResult
from ml.features.builder import (
    build_paper_features,
    FEATURE_GROUPS,
    FEATURE_NAMES,
)


@pytest.fixture
def test_resolver(tmp_path):
    """Creates an isolated VenueResolver with synthetic historical and future editions."""
    # 1. Create synthetic SCImago dataset
    # Journal A: published in 2015, 2018, 2021
    # Journal B: only published in 2020 (future only for 2012-2016 papers)
    scimago_data = pd.DataFrame([
        {"issn": "11112222", "year": 2015, "quartile": "Q2", "sjr": 0.85, "h_index": 40.0},
        {"issn": "11112222", "year": 2018, "quartile": "Q1", "sjr": 1.45, "h_index": 65.0},
        {"issn": "11112222", "year": 2021, "quartile": "Q1", "sjr": 2.10, "h_index": 90.0},
        {"issn": "33334444", "year": 2020, "quartile": "Q1", "sjr": 3.00, "h_index": 120.0},
    ])
    scimago_path = tmp_path / "scimago.parquet"
    scimago_data.to_parquet(scimago_path)

    # 2. Create synthetic CORE dataset
    # Conf A: 2014 (Rank B), 2018 (Rank A), 2021 (Rank A*)
    # Conf B: 2020 only (Rank A*)
    core_data = pd.DataFrame([
        {"title": "International Conference on Machine Learning Test", "acronym": "ICMLTEST", "edition_year": 2014, "rank": "B"},
        {"title": "International Conference on Machine Learning Test", "acronym": "ICMLTEST", "edition_year": 2018, "rank": "A"},
        {"title": "International Conference on Machine Learning Test", "acronym": "ICMLTEST", "edition_year": 2021, "rank": "A*"},
        {"title": "Future Conference on Neural Systems", "acronym": "FCNS", "edition_year": 2020, "rank": "A*"},
    ])
    core_path = tmp_path / "core.parquet"
    core_data.to_parquet(core_path)

    return ScientometricVenueResolver(scimago_path=scimago_path, core_path=core_path)


def test_scimago_never_leaks_future_year(test_resolver):
    """Test that SCImago lookup rejects editions published after pub_year."""
    # Paper published in 2012: Journal A has editions only in 2015, 2018, 2021 -> must be unknown
    res_2012 = test_resolver.resolve(venue_issn_l="1111-2222", pub_year=2012)
    assert res_2012.ranking_system is None, "Leaked future edition for 2012 paper!"
    assert res_2012.sjr is None
    assert res_2012.quartile is None

    # Paper published in 2016: Journal A has 2015 edition -> must return 2015 (Q2, sjr=0.85), NOT 2018
    res_2016 = test_resolver.resolve(venue_issn_l="1111-2222", pub_year=2016)
    assert res_2016.ranking_system == "SCImago"
    assert res_2016.quartile == "Q2"
    assert res_2016.sjr == 0.85

    # Paper published in 2019: Journal A has 2018 edition -> must return 2018 (Q1, sjr=1.45), NOT 2021
    res_2019 = test_resolver.resolve(venue_issn_l="1111-2222", pub_year=2019)
    assert res_2019.ranking_system == "SCImago"
    assert res_2019.quartile == "Q1"
    assert res_2019.sjr == 1.45

    # Journal B only has 2020 edition: paper published in 2017 must return unknown
    res_b_2017 = test_resolver.resolve(venue_issn_l="3333-4444", pub_year=2017)
    assert res_b_2017.ranking_system is None, "Leaked 2020 edition for 2017 paper!"
    assert res_b_2017.sjr is None


def test_core_never_leaks_future_edition(test_resolver):
    """Test that CORE conference lookup rejects editions after pub_year."""
    # Paper published in 2012: Conf A has first edition in 2014 -> must be unranked
    res_2012 = test_resolver.resolve(venue_name="Conference on ICMLTEST", venue_type="conference", pub_year=2012)
    assert res_2012.core_rank is None, "Leaked 2014/2018/2021 CORE edition to 2012 paper!"

    # Paper published in 2015: Conf A had 2014 edition (Rank B) -> must return B, NOT A or A*
    res_2015 = test_resolver.resolve(venue_name="Conference on ICMLTEST", venue_type="conference", pub_year=2015)
    assert res_2015.core_rank == "B", f"Expected historical rank 'B', got '{res_2015.core_rank}'"

    # Paper published in 2019: Conf A had 2018 edition (Rank A) -> must return A, NOT 2021 A*
    res_2019 = test_resolver.resolve(venue_name="Conference on ICMLTEST", venue_type="conference", pub_year=2019)
    assert res_2019.core_rank == "A", f"Expected rank 'A', got '{res_2019.core_rank}'"

    # Conf B (2020 only): Paper in 2016 must be unranked
    res_b_2016 = test_resolver.resolve(venue_name="Annual FCNS Proceedings", venue_type="conference", pub_year=2016)
    assert res_b_2016.core_rank is None, "Leaked 2020 CORE rank to 2016 paper!"


def test_feature_builder_unknown_venue_indicator():
    """Test that an unknown venue produces explicit -1.0 SJR with venue_known = 0.0."""
    paper_unknown_venue = {
        "work_id": "W99999999",
        "title": "A Novel Algorithm Without Venue Data",
        "publication_year": 2015,
        "primary_field_name": "Computer Science",
        "counts_by_year": [{"year": 2016, "cited_by_count": 5}],
        "referenced_works_count": 10,
        "num_authors": 2,
        "is_oa": False,
        "oa_status": "closed",
        "venue_sjr": None,
        "venue_scimago_quartile": None,
        "venue_core_rank": None,
    }
    feats, names = build_paper_features(paper_unknown_venue)
    sjr_slice = FEATURE_GROUPS["sjr"]
    sjr_val = feats[sjr_slice.start]
    venue_known_val = feats[sjr_slice.start + 1]

    assert sjr_val == -1.0, f"Expected -1.0 for missing SJR, got {sjr_val}"
    assert venue_known_val == 0.0, f"Expected 0.0 for venue_known, got {venue_known_val}"
