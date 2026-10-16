"""Tabular Feature Extraction Module (SciGraph AI 2.0).

Delegates feature extraction to ml.features.builder.build_paper_features_batch,
maintaining backward compatibility while enforcing strict feature parity.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd

from ml.features.builder import build_paper_features_batch, FEATURE_NAMES


def extract_tabular_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X, target vector y, and feature column names via central builder."""
    X, feature_cols = build_paper_features_batch(df)

    if "impact_class" in df.columns:
        y = df["impact_class"].values.astype(int)
    elif "impact_label" in df.columns:
        y = df["impact_label"].values.astype(int)
    else:
        y = np.zeros(len(df), dtype=int)

    return X, y, feature_cols
