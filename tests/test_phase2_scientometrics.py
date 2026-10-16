"""Verification Test Suite for Phase 2: Authentic Scientometrics & Teacher Suggestions.

Asserts:
1. SCImago Journal Rank (SJR) lookup via ISSN returns authentic publication-year quartiles.
2. CORE conference lookup returns authentic rankings (A*, A, B, C) for conferences.
3. Strict Null Handling: Unknown venues return None/null, never default to Q2 or fake strings (Rule R1).
4. Access economics computes authentic OA status, real APCs, and dynamically linked empirical advantage.
5. Bidirectional lineage reports true in/out degree counts and replaces simulated ratios.
6. Provenance tags ('scimago', 'core', 'openalex | computed') are rigorously populated.
"""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.scientometrics.venue_resolver import venue_resolver, VenueRankingResult
from ml.scientometrics.access_economics import access_service, AccessEconomicsResult
from ml.scientometrics.lineage_engine import lineage_engine, BidirectionalLineageResult


def test_venue_resolver_scimago_issn():
    """Verify that journals with valid ISSN resolve to authentic SCImago quartiles."""
    # Test Nature (ISSN: 0028-0836)
    res = venue_resolver.resolve(
        venue_name="Nature",
        venue_issn_l="0028-0836",
        venue_type="journal",
        publication_year=2015
    )
    assert isinstance(res, VenueRankingResult)
    assert res.ranking_system == "SCImago"
    assert res.quartile == "Q1"
    assert res.provenance == "scimago"
    assert res.sjr is not None and res.sjr > 0.0


def test_venue_resolver_core_conference():
    """Verify that top computer science conferences resolve to CORE rankings."""
    # Test CVPR (CORE Rank A*)
    res = venue_resolver.resolve(
        venue_name="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        venue_issn_l="1063-6919",
        venue_type="conference",
        publication_year=2016
    )
    assert isinstance(res, VenueRankingResult)
    assert res.ranking_system == "CORE"
    assert res.core_rank in ["A*", "A"]
    assert res.provenance == "core"


def test_venue_resolver_real_corpus_cvpr_proceedings_resolves_and_deduplicates():
    """Verify Task 1d: Real corpus CVPR proceedings string de-duplicates and resolves via CORE."""
    real_vname = "IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings"
    res = venue_resolver.resolve(
        venue_name=real_vname,
        venue_type="conference",
        publication_year=2016
    )
    assert isinstance(res, VenueRankingResult)
    assert res.venue_name == "IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings"
    assert res.ranking_system == "CORE"
    assert res.core_rank == "A*"
    assert res.provenance == "core"



def test_venue_resolver_unknown_returns_none():
    """Verify Rule R1: Unknown venues must return None/null, NEVER default to Q2 or fake strings."""
    res = venue_resolver.resolve(
        venue_name="Nonexistent Imaginary Journal of Unverified Research",
        venue_issn_l="9999-9999",
        venue_type="journal",
        publication_year=2015
    )
    assert isinstance(res, VenueRankingResult)
    assert res.ranking_system is None
    assert res.quartile is None, "Violated Rule R1: Unknown venue defaulted to a quartile!"
    assert res.core_rank is None
    assert res.provenance is None


def test_access_economics_gold_oa_with_apc():
    """Verify authentic access economics for Gold OA with real APC."""
    oa_obj = {
        "is_oa": True,
        "oa_status": "gold",
        "oa_url": "https://doi.org/10.1038/s41467-020-00000"
    }
    apc_paid = {"value_usd": 2790.0, "currency": "USD"}
    res = access_service.evaluate_access(
        open_access_obj=oa_obj,
        field_name="Computer Science",
        apc_paid_obj=apc_paid
    )
    assert isinstance(res, AccessEconomicsResult)
    assert res.is_oa is True
    assert res.oa_status == "gold"
    assert res.apc_paid_usd == 2790.0
    assert res.empirical_cohort_oa_advantage_pct > 0.0
    assert "Association, Not Causation" in res.scientometric_notice
    assert res.provenance == "openalex | computed"


def test_access_economics_closed_paywall():
    """Verify authentic access economics for closed subscription papers."""
    oa_obj = {
        "is_oa": False,
        "oa_status": "closed",
        "oa_url": None
    }
    res = access_service.evaluate_access(open_access_obj=oa_obj)
    assert isinstance(res, AccessEconomicsResult)
    assert res.is_oa is False
    assert res.oa_status == "closed"
    assert res.empirical_cohort_oa_advantage_pct == 0.0
    assert res.apc_paid_usd is None


def test_bidirectional_lineage_computation():
    """Verify real in/out degree counts and replacement of simulated metrics."""
    res = lineage_engine.compute_lineage(
        target_work_id="https://openalex.org/W2194775991",
        referenced_works=["https://openalex.org/W1", "https://openalex.org/W2"],
        raw_counts_by_year=[{"year": 2016, "cited_by_count": 50}, {"year": 2017, "cited_by_count": 120}],
        citation_normalized_percentile=99.8
    )
    assert isinstance(res, BidirectionalLineageResult)
    assert res.outflow_count == 2
    assert res.inflow_count == 170
    assert res.inflow_outflow_ratio == 85.0
    assert res.citation_normalized_percentile == 99.8
    assert res.provenance == "openalex | computed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
