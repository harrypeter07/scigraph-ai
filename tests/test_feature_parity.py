"""Feature Parity Test Suite for WP1.

Verifies:
1. For 200 corpus works, features built from parquet rows equal features built from
   raw OpenAlex-style dictionaries (np.allclose, atol=1e-5).
2. Exactly 14 FEATURE_GROUPS covering exactly 47 dimensions.
3. Unknown SJR and unknown APC explicitly do NOT equal 0.0 (assigned -1.0 with missingness indicator 0.0).
4. Text encoder is loaded from persisted checkpoint (never refit on boot).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features.builder import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    TOTAL_FEATURE_DIM,
    build_paper_features,
    build_paper_features_batch,
    get_text_encoder,
    DEFAULT_TEXT_ENCODER_PATH,
)


def test_feature_groups_structure():
    """Assert exactly 14 feature groups covering exactly 47 dimensions without gaps."""
    assert len(FEATURE_GROUPS) == 14, f"Expected exactly 14 feature groups, got {len(FEATURE_GROUPS)}"
    assert TOTAL_FEATURE_DIM == 47, f"Expected 47 dimensions, got {TOTAL_FEATURE_DIM}"
    assert len(FEATURE_NAMES) == 47, f"Expected 47 feature names, got {len(FEATURE_NAMES)}"

    # Check contiguous slices without overlap
    current_idx = 0
    for name, sl in FEATURE_GROUPS.items():
        assert sl.start == current_idx, f"Group {name} slice start {sl.start} != {current_idx}"
        current_idx = sl.stop
    assert current_idx == 47, f"Total slice coverage {current_idx} != 47"


def test_text_encoder_persistence():
    """Assert text encoder is loaded from disk and has valid TF-IDF and SVD fitted states."""
    assert DEFAULT_TEXT_ENCODER_PATH.exists(), "Text encoder checkpoint does not exist"
    encoder = get_text_encoder()
    assert "tfidf" in encoder and "svd" in encoder
    assert hasattr(encoder["tfidf"], "vocabulary_")
    assert hasattr(encoder["svd"], "components_")
    assert encoder["svd"].components_.shape == (32, 2500)


def test_missingness_not_zero():
    """Assert unknown SJR and unknown APC do not equal 0.0 (they equal -1.0 with flag 0.0)."""
    blank_record = {
        "title": "A Study in Scientometrics",
        "publication_year": 2018,
        "citations_at_cutoff": 5,
        "referenced_works_count": 10,
        "num_authors": 2,
        "is_oa": False,
        "oa_status": "closed",
        "apc_paid_usd": None,  # Unknown APC
        "venue_sjr": None,     # Unknown SJR
        "venue_scimago_quartile": None,
        "venue_core_rank": None,
    }
    feats, names = build_paper_features(blank_record)

    # Check APC group: [log_apc, apc_known]
    apc_slice = FEATURE_GROUPS["log_apc"]
    log_apc, apc_known = feats[apc_slice]
    assert apc_known == 0.0
    assert log_apc == -1.0, f"Unknown APC must NOT equal 0.0, got {log_apc}"

    # Check SJR group: [sjr, venue_known]
    sjr_slice = FEATURE_GROUPS["sjr"]
    sjr_val, venue_known = feats[sjr_slice]
    assert venue_known == 0.0
    assert sjr_val == -1.0, f"Unknown SJR must NOT equal 0.0, got {sjr_val}"


def test_200_works_parity_parquet_vs_raw_openalex():
    """Verify feature parity for 200 corpus works between parquet row and raw OpenAlex dict.

    Residual as-of-today fields documented per WP1 specification:
    - Open access status (oa_status, is_oa) reflects contemporary repository crawls.
    - Article Processing Charges (apc_paid_usd) reflects current institutional records.
    Both are tested for strict equality when mapped from raw record to parquet schema.
    """
    df_works = pd.read_parquet("data/works.parquet")
    sample_df = df_works.sample(n=200, random_state=42)

    batch_feats, _ = build_paper_features_batch(sample_df)

    for i, (_, row) in enumerate(sample_df.iterrows()):
        # 1. Feature vector built from parquet row
        row_feats, _ = build_paper_features(row)

        # 2. Assert vectorized batch matches row-by-row
        assert np.allclose(row_feats, batch_feats[i], atol=1e-5), (
            f"Batch vs row mismatch at work {row['work_id']}"
        )

        # 3. Build raw OpenAlex-style dictionary simulating API response
        pub_year = int(row["publication_year"])
        cutoff_year = pub_year + 2
        raw_dict = {
            "id": row["work_id"],
            "title": row["title"],
            "publication_year": pub_year,
            "counts_by_year": [
                {"year": cutoff_year - 1, "cited_by_count": int(row["citations_at_cutoff"]) // 2},
                {"year": cutoff_year, "cited_by_count": int(row["citations_at_cutoff"]) - (int(row["citations_at_cutoff"]) // 2)},
                {"year": cutoff_year + 1, "cited_by_count": 999},  # Future citations (must be ignored)
            ],
            "referenced_works": ["W_ref"] * int(row["referenced_works_count"]),
            "authorships": [{"author": {"display_name": "Author"}} for _ in range(int(row["num_authors"]))],
            "open_access": {
                "is_oa": bool(row["is_oa"]),
                "oa_status": str(row["oa_status"]),
            },
            "apc_paid": {"value_usd": row["apc_paid_usd"]} if pd.notnull(row["apc_paid_usd"]) else None,
            "venue_sjr": row["venue_sjr"] if pd.notnull(row["venue_sjr"]) else None,
            "venue_scimago_quartile": row["venue_scimago_quartile"] if pd.notnull(row["venue_scimago_quartile"]) else None,
            "venue_core_rank": row["venue_core_rank"] if pd.notnull(row["venue_core_rank"]) else None,
        }

        dict_feats, _ = build_paper_features(raw_dict)

        # Assert parity between parquet row and raw OpenAlex dictionary
        diff = np.abs(row_feats - dict_feats)
        max_diff = np.max(diff)
        assert max_diff < 1e-5, (
            f"Parity mismatch at work {row['work_id']}! Max diff: {max_diff:.6f}, "
            f"Mismatched indices: {np.where(diff >= 1e-5)[0]}"
        )
