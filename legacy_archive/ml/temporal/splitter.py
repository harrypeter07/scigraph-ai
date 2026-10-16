"""Dataset Splitting Module (Time-Consistent vs Naive Random).

Implements both time-consistent splitting by publication year and naive random splitting
for temporal leakage ablation studies.
"""

import os
import yaml
import logging
import pandas as pd
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetSplitter")


class TemporalDatasetSplitter:
    """Splitter handling time-consistent and naive random splits."""

    def __init__(self, config_path: str = "configs/splits.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f).get("splits", {})

        self.time_cfg = self.config.get("time_consistent", {})
        self.naive_cfg = self.config.get("naive_random", {})

    def split_time_consistent(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform strict publication year temporal split."""
        train_min = self.time_cfg.get("train_years", {}).get("min", 2012)
        train_max = self.time_cfg.get("train_years", {}).get("max", 2018)
        val_min = self.time_cfg.get("val_years", {}).get("min", 2019)
        val_max = self.time_cfg.get("val_years", {}).get("max", 2019)
        test_min = self.time_cfg.get("test_years", {}).get("min", 2020)
        test_max = self.time_cfg.get("test_years", {}).get("max", 2021)

        train_df = df[(df["publication_year"] >= train_min) & (df["publication_year"] <= train_max)].copy()
        val_df = df[(df["publication_year"] >= val_min) & (df["publication_year"] <= val_max)].copy()
        test_df = df[(df["publication_year"] >= test_min) & (df["publication_year"] <= test_max)].copy()

        # SPLIT INTEGRITY ASSERTION
        train_ids = set(train_df["id"])
        val_ids = set(val_df["id"])
        test_ids = set(test_df["id"])

        assert train_ids.isdisjoint(val_ids), "Train and Val paper IDs overlap!"
        assert train_ids.isdisjoint(test_ids), "Train and Test paper IDs overlap!"
        assert val_ids.isdisjoint(test_ids), "Val and Test paper IDs overlap!"

        logger.info(f"Time-Consistent Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def split_naive_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Perform naive random split across all publication years (DELIBERATELY FLAWED ABLATION CONDITION)."""
        train_ratio = self.naive_cfg.get("train_ratio", 0.70)
        val_ratio = self.naive_cfg.get("val_ratio", 0.15)
        seed = self.naive_cfg.get("seed", 42)

        shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        total = len(shuffled_df)

        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        train_df = shuffled_df.iloc[:n_train].copy()
        val_df = shuffled_df.iloc[n_train:n_train + n_val].copy()
        test_df = shuffled_df.iloc[n_train + n_val:].copy()

        logger.info(f"Naive Random Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def save_splits(self, df: pd.DataFrame, processed_dir: str = "data/processed") -> Dict[str, str]:
        """Execute and save both temporal and naive random splits."""
        os.makedirs(processed_dir, exist_ok=True)

        train_t, val_t, test_t = self.split_time_consistent(df)
        train_n, val_n, test_n = self.split_naive_random(df)

        paths = {
            "train_temporal": os.path.join(processed_dir, "train_temporal.parquet"),
            "val_temporal": os.path.join(processed_dir, "val_temporal.parquet"),
            "test_temporal": os.path.join(processed_dir, "test_temporal.parquet"),
            "train_naive": os.path.join(processed_dir, "train_naive.parquet"),
            "val_naive": os.path.join(processed_dir, "val_naive.parquet"),
            "test_naive": os.path.join(processed_dir, "test_naive.parquet"),
        }

        train_t.to_parquet(paths["train_temporal"], index=False)
        val_t.to_parquet(paths["val_temporal"], index=False)
        test_t.to_parquet(paths["test_temporal"], index=False)

        train_n.to_parquet(paths["train_naive"], index=False)
        val_n.to_parquet(paths["val_naive"], index=False)
        test_n.to_parquet(paths["test_naive"], index=False)

        return paths


if __name__ == "__main__":
    labeled_path = "data/processed/labeled_papers.parquet"
    if os.path.exists(labeled_path):
        df = pd.read_parquet(labeled_path)
        splitter = TemporalDatasetSplitter()
        paths = splitter.save_splits(df)
        print("Dataset splits saved successfully:", paths)
