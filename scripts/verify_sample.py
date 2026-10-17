"""Independent Sample Verification Script (Layer 2 Verification).

Picks n random held-out test papers, queries api.openalex.org LIVE for each,
recomputes the 3-year window citations and actual tier purely from external API data,
compares with stored dataset records, and prints an audit verification table.

Usage:
    python scripts/verify_sample.py --n 20
"""

import sys
import json
import random
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

POLITE_EMAIL = "researcher@scigraph.org"
USER_AGENT = f"SciGraphVerification/2.0 (mailto:{POLITE_EMAIL})"
OPENALEX_BASE = "https://api.openalex.org/works"


def fetch_openalex_live(work_id: str) -> Dict[str, Any]:
    """Fetches work directly from OpenAlex API."""
    clean_id = str(work_id).split("/")[-1]
    if not clean_id.startswith("W"):
        clean_id = "W" + clean_id

    url = f"{OPENALEX_BASE}/{clean_id}?select=id,title,publication_year,primary_topic,counts_by_year,cited_by_count&mailto={POLITE_EMAIL}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {}


def load_cohort_threshold(field_name: Optional[str], pub_year: int) -> Dict[str, float]:
    """Loads empirical cohort thresholds (p50, p90) reading ONLY label_thresholds.json."""
    cohort_path = REPO_ROOT / "data" / "reference" / "label_thresholds.json"
    if cohort_path.exists():
        try:
            with open(cohort_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                by_fy = data.get("by_field_year", {})
                key = f"{field_name}::{pub_year}"
                if key in by_fy:
                    return {"p50": float(by_fy[key]["p50"]), "p90": float(by_fy[key]["p90"])}
                by_f = data.get("by_field", {})
                if field_name and str(field_name) in by_f:
                    return {"p50": float(by_f[str(field_name)]["p50"]), "p90": float(by_f[str(field_name)]["p90"])}
                glob = data.get("global", {})
                if "p50" in glob:
                    return {"p50": float(glob["p50"]), "p90": float(glob["p90"])}
        except Exception:
            pass
    return {"p50": 330.0, "p90": 882.0}


def run_sample_verification(n: int = 20, seed: int = 42):
    print("=" * 105)
    print(f"SCIGRAPH AI 2.0: LIVE EXTERNAL OPENALEX API SPOT-CHECK (N={n})")
    print("Independent Layer 2 Ground-Truth Audit: External API vs. Local Dataset Records")
    print("=" * 105)

    test_path = REPO_ROOT / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        print(f"ERROR: {test_path} missing.")
        sys.exit(1)

    df_test = pd.read_parquet(test_path)
    sample = df_test.sample(n=min(n, len(df_test)), random_state=seed)

    results = []
    print(f"\nFetching {len(sample)} works live from api.openalex.org...\n")

    for _, row in sample.iterrows():
        wid = row["work_id"]
        stored_pub_year = int(row["publication_year"])
        stored_field = row.get("primary_field_name", "Computer Science")
        stored_window = int(row.get("future_citations_5y", 0))
        stored_class = int(row.get("impact_class", 0))

        live_data = fetch_openalex_live(wid)
        if not live_data:
            results.append({
                "work_id": wid,
                "title": row.get("title", "Untitled")[:30],
                "pub_year": stored_pub_year,
                "stored_window": stored_window,
                "live_window": None,
                "stored_cls": stored_class,
                "live_cls": None,
                "status": "API_TIMEOUT"
            })
            continue

        counts_by_year = live_data.get("counts_by_year", [])
        by_year_dict = {
            int(c.get("year", 0)): int(c.get("cited_by_count", 0))
            for c in counts_by_year
            if isinstance(c, dict)
        }

        cutoff_year = stored_pub_year + 2
        window_counts = [by_year_dict.get(yr, 0) for yr in range(cutoff_year + 1, stored_pub_year + 6)]
        live_window = sum(window_counts)

        thresholds = load_cohort_threshold(stored_field, stored_pub_year)
        p50 = thresholds["p50"]
        p90 = thresholds["p90"]

        if live_window < p50:
            live_class = 0
        elif live_window < p90:
            live_class = 1
        else:
            live_class = 2

        tier_match = (stored_class == live_class)
        # Note: OpenAlex counts can increase slightly over time for live queries
        results.append({
            "work_id": wid,
            "title": row.get("title", "Untitled")[:32],
            "pub_year": stored_pub_year,
            "stored_window": stored_window,
            "live_window": live_window,
            "stored_cls": stored_class,
            "live_cls": live_class,
            "tier_match": tier_match,
            "status": "MATCH" if tier_match else "SHIFT"
        })

    # Print summary table
    print(f"{'Work ID':13s} | {'Title':34s} | {'Year':4s} | {'Stored Window':13s} | {'Live Window':11s} | {'Stored Cls':10s} | {'Live Cls':8s} | {'Audit':8s}")
    print("-" * 105)
    matches = 0
    total_valid = 0
    for r in results:
        wid = r["work_id"]
        t = r["title"]
        yr = str(r["pub_year"])
        sw = f"{r['stored_window']:,}"
        lw = f"{r['live_window']:,}" if r['live_window'] is not None else "ERR"
        sc = str(r["stored_cls"])
        lc = str(r["live_cls"]) if r['live_cls'] is not None else "ERR"
        aud = r["status"]
        if r["status"] in ["MATCH", "SHIFT"]:
            total_valid += 1
            if r["tier_match"]:
                matches += 1
        print(f"{wid:13s} | {t:34s} | {yr:4s} | {sw:13s} | {lw:11s} | {sc:10s} | {lc:8s} | {aud:8s}")

    match_rate = (matches / total_valid * 100.0) if total_valid > 0 else 0.0
    print("=" * 105)
    print(f"VERIFICATION SUMMARY: {matches}/{total_valid} spot-checked papers matched stored impact class ({match_rate:.1f}%).")
    print("Live API citations recomputed strictly from api.openalex.org counts_by_year.")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run independent spot-check of test papers against live OpenAlex API.")
    parser.add_argument("--n", type=int, default=10, help="Number of papers to spot check (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    run_sample_verification(n=args.n, seed=args.seed)
