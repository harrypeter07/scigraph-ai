"""Cohort-Normalized Labeling Module.

Computes 3-class impact labels (Low: 0, Medium: 1, High: 2) using percentile-based
cohort normalization within (publication_year, primary_topic_id) subfield groups.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CohortLabeler")


class CohortImpactLabeler:
    """Labeler that assigns Low/Medium/High impact classes relative to cohort trajectory distributions."""

    def __init__(self, config_path: str = "configs/labels.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f).get("labeling", {})

        self.horizon_years = self.config.get("horizon_years", 5)
        self.cohort_keys = self.config.get("cohort_keys", ["publication_year", "primary_topic_id"])
        thresholds = self.config.get("thresholds", {})
        self.low_percentile = thresholds.get("low_percentile", 50.0)
        self.high_percentile = thresholds.get("high_percentile", 90.0)

    def compute_percentiles_in_cohort(self, group: pd.DataFrame) -> pd.Series:
        """Compute percentile rank of delta_citations_5y within cohort group."""
        values = group["delta_citations_5y"]
        if len(values) <= 1:
            return pd.Series(50.0, index=group.index)
        
        # Rank values (0 to 100)
        ranks = values.rank(method="average", pct=True) * 100.0
        return ranks

    def assign_class_from_percentile(self, pct: float) -> int:
        """Assign 0 (Low), 1 (Medium), or 2 (High) based on percentile score."""
        if pct < self.low_percentile:
            return 0  # Low
        elif pct < self.high_percentile:
            return 1  # Medium
        else:
            return 2  # High

    def label_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cohort percentile labeling to snapshotted papers dataframe."""
        labeled_df = df.copy()

        if "delta_citations_5y" not in labeled_df.columns:
            raise KeyError("Input dataframe missing required column 'delta_citations_5y'")

        # Fill missing primary_topic_id if needed
        if "primary_topic_id" not in labeled_df.columns:
            labeled_df["primary_topic_id"] = "UNKNOWN_TOPIC"

        # Apply cohort percentile calculation
        percentile_series = labeled_df.groupby(["publication_year", "primary_topic_id"], group_keys=False)["delta_citations_5y"].transform(
            lambda x: x.rank(method="average", pct=True) * 100.0 if len(x) > 1 else pd.Series(50.0, index=x.index)
        )
        
        labeled_df["impact_cohort_percentile"] = percentile_series

        # Assign class labels
        labeled_df["impact_label"] = labeled_df["impact_cohort_percentile"].apply(self.assign_class_from_percentile)

        # Log class distribution
        class_counts = labeled_df["impact_label"].value_counts().to_dict()
        total = len(labeled_df)
        logger.info(f"Labeling complete across {total} papers.")
        for label_cls, count in sorted(class_counts.items()):
            pct = (count / total) * 100.0 if total > 0 else 0.0
            label_name = {0: "Low (<50%)", 1: "Medium (50-90%)", 2: "High (>=90%)"}.get(label_cls, str(label_cls))
            logger.info(f"  Class {label_cls} ({label_name}): {count} papers ({pct:.2f}%)")

        return labeled_df


if __name__ == "__main__":
    snap_path = "data/interim/snapshotted_papers.parquet"
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "labeled_papers.parquet")

    if os.path.exists(snap_path):
        df = pd.read_parquet(snap_path)
        labeler = CohortImpactLabeler()
        labeled_df = labeler.label_dataset(df)
        
        # Convert list/dict columns if present before saving
        for col in ["counts_by_year", "referenced_works"]:
            if col in labeled_df.columns and not isinstance(labeled_df[col].iloc[0], str):
                import json
                labeled_df[col] = labeled_df[col].apply(json.dumps)

        labeled_df.to_parquet(out_path, index=False)
        print("Labeled papers saved successfully to:", out_path)
