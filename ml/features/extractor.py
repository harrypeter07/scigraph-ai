"""Tabular Feature Extraction Module.

Transforms snapshotted paper data into numerical feature matrices for baseline tabular models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def extract_tabular_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X, target vector y, and feature column names."""
    feat_df = df.copy()

    # Feature 1: Historical citation count at cutoff
    if "historical_citation_count_at_cutoff" not in feat_df.columns:
        feat_df["historical_citation_count_at_cutoff"] = 0

    # Feature 2: Referenced works count
    if "referenced_works_count" not in feat_df.columns:
        feat_df["referenced_works_count"] = 0

    # Feature 3: Abstract word count
    feat_df["abstract_word_count"] = feat_df["abstract_text"].apply(lambda t: len(str(t).split()) if t else 0)

    # Feature 4: Title word count
    feat_df["title_word_count"] = feat_df["title"].apply(lambda t: len(str(t).split()) if t else 0)

    # Feature 5: Publication year offset relative to 2012
    feat_df["pub_year_offset"] = feat_df["publication_year"] - 2012

    feature_cols = [
        "historical_citation_count_at_cutoff",
        "referenced_works_count",
        "abstract_word_count",
        "title_word_count",
        "pub_year_offset"
    ]

    X = feat_df[feature_cols].fillna(0).values.astype(np.float32)
    
    if "impact_label" in feat_df.columns:
        y = feat_df["impact_label"].values.astype(int)
    else:
        y = np.zeros(len(feat_df), dtype=int)

    return X, y, feature_cols
