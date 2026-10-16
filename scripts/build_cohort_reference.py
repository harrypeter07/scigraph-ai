"""Build Empirical Cohort Citation Thresholds from TRAIN+VAL Works (<= 2017).

Generates data/reference/cohort_thresholds.json storing empirical p50 and p90
citations_window (citations in cutoff+1..pub+5) per (field, year) cohort,
with fallback hierarchies for field-level and global levels.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings


def build_cohort_reference(output_path: Path = Path("data/reference/cohort_thresholds.json")):
    print("Loading data/works.parquet for cohort thresholds...")
    df_works = pd.read_parquet("data/works.parquet")

    # Strictly use TRAIN + VAL works (publication_year <= 2017)
    train_val_mask = df_works["publication_year"] <= 2017
    df_tv = df_works[train_val_mask].copy()
    print(f"Total TRAIN+VAL works (<=2017): {len(df_tv):,}")

    cits_col = "future_citations_5y"  # strictly citations in (cutoff, pub+5]

    # 1. Global train+val distribution
    global_vals = df_tv[cits_col].dropna().values
    global_p50 = float(np.percentile(global_vals, 50))
    global_p90 = float(np.percentile(global_vals, 90))

    thresholds_data = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": "data/works.parquet",
            "filter": "publication_year <= 2017 (TRAIN+VAL only)",
            "total_sample_size": len(df_tv),
            "label_window": "citations_window: (pub_year + 2, pub_year + 5]",
        },
        "global": {
            "p50": round(global_p50, 2),
            "p90": round(global_p90, 2),
            "sample_size": len(global_vals),
        },
        "by_field": {},
        "by_field_year": {}
    }

    # 2. Per-Field distribution
    for field_name, grp in df_tv.groupby("primary_field_name"):
        f_vals = grp[cits_col].dropna().values
        if len(f_vals) > 0:
            thresholds_data["by_field"][str(field_name)] = {
                "p50": round(float(np.percentile(f_vals, 50)), 2),
                "p90": round(float(np.percentile(f_vals, 90)), 2),
                "sample_size": len(f_vals)
            }

    # 3. Per-(Field, Year) distribution with min-sample threshold (>= 50)
    for (field_name, year), grp in df_tv.groupby(["primary_field_name", "publication_year"]):
        fy_vals = grp[cits_col].dropna().values
        cohort_key = f"{field_name}::{year}"
        if len(fy_vals) >= 50:
            thresholds_data["by_field_year"][cohort_key] = {
                "p50": round(float(np.percentile(fy_vals, 50)), 2),
                "p90": round(float(np.percentile(fy_vals, 90)), 2),
                "sample_size": len(fy_vals),
                "cohort_level": "field_year"
            }
        else:
            # Fall back to field level
            f_stats = thresholds_data["by_field"].get(str(field_name), thresholds_data["global"])
            thresholds_data["by_field_year"][cohort_key] = {
                "p50": f_stats["p50"],
                "p90": f_stats["p90"],
                "sample_size": len(fy_vals),
                "cohort_level": "field_fallback"
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_data, f, indent=2)

    print(f"Cohort reference written to: {output_path}")
    print(f"Global p50: {global_p50:.2f}, p90: {global_p90:.2f}")
    print(f"Fields indexed: {len(thresholds_data['by_field'])}")
    print(f"Field-Year cohorts indexed: {len(thresholds_data['by_field_year'])}")
    return output_path


if __name__ == "__main__":
    build_cohort_reference()
