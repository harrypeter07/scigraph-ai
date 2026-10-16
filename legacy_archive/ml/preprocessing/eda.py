"""Exploratory Data Analysis (EDA) Generator.

Generates dataset statistics, summary figures, and writes reports/dataset_report.md.
Gracefully handles missing plotting packages.
"""

import os
import json
import pandas as pd
from typing import Dict, Any


def generate_eda(interim_dir: str = "data/interim", processed_dir: str = "data/processed", fig_dir: str = "figures", report_dir: str = "reports"):
    """Generate EDA summary stats and dataset_report.md from real ingested data."""
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    papers_path = os.path.join(processed_dir, "labeled_papers.parquet")
    authorships_path = os.path.join(interim_dir, "authorships.parquet")

    if not os.path.exists(papers_path):
        print(f"Skipping EDA: {papers_path} does not exist.")
        return

    df = pd.read_parquet(papers_path)
    auth_df = pd.read_parquet(authorships_path) if os.path.exists(authorships_path) else pd.DataFrame()

    total_papers = len(df)
    min_year = int(df["publication_year"].min()) if not df.empty else 0
    max_year = int(df["publication_year"].max()) if not df.empty else 0

    year_counts = df["publication_year"].value_counts().sort_index().to_dict()
    class_counts = df["impact_label"].value_counts().sort_index().to_dict()

    # Try plotting with matplotlib if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.bar(list(year_counts.keys()), list(year_counts.values()), color="skyblue")
        plt.title("OpenAlex AI/ML Publications per Year (2012–2022)")
        plt.xlabel("Publication Year")
        plt.ylabel("Number of Papers")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "papers_per_year.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.bar(["0: Low", "1: Med", "2: High"][:len(class_counts)], list(class_counts.values()), color="teal")
        plt.title("Impact Label Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
        plt.close()
        print("Matplotlib figures generated successfully.")
    except Exception as e:
        print(f"Plotting notice ({e}). Generated text statistics.")

    # Write reports/dataset_report.md
    report_file = os.path.join(report_dir, "dataset_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# SciGraph AI — Dataset Exploratory Data Analysis (EDA) Report\n\n")
        f.write(f"- **Total Papers Ingested & Labeled**: {total_papers}\n")
        f.write(f"- **Publication Year Range**: {min_year} – {max_year}\n")
        f.write(f"- **Total Authorship Relations**: {len(auth_df)}\n\n")
        f.write("## Impact Label Class Distribution\n")
        for cls, count in class_counts.items():
            f.write(f"- Class {cls}: {count} papers ({count/total_papers*100:.2f}%)\n")
        f.write("\n## Publication Counts by Year\n")
        for yr, count in year_counts.items():
            f.write(f"- {yr}: {count} papers\n")

    print("EDA dataset report written to:", report_file)


if __name__ == "__main__":
    generate_eda()
