"""Fit TF-IDF + TruncatedSVD on TRAIN titles only and save checkpoint."""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features.builder import fit_and_save_text_encoder, DEFAULT_TEXT_ENCODER_PATH


def main():
    print("Loading data/works.parquet...")
    df_works = pd.read_parquet("data/works.parquet")
    print(f"Total works: {len(df_works):,}")

    train_mask = df_works["temporal_split"] == "train"
    train_titles = df_works.loc[train_mask, "title"].fillna("Untitled").tolist()
    print(f"Fitting text encoder on {len(train_titles):,} TRAIN titles (temporal_split == 'train')...")

    out_path = fit_and_save_text_encoder(
        train_titles=train_titles,
        output_path=DEFAULT_TEXT_ENCODER_PATH,
        max_features=2500,
        n_components=32,
        random_state=42
    )
    print(f"Text encoder saved successfully to: {out_path}")
    print(f"File size: {out_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
