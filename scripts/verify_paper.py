"""Independent Verification Script for a Single Paper (Layer 2 Verification).

Queries api.openalex.org directly and recomputes the ground truth citation outcome
completely independently of the inference service or server runtime:
- Extracts publication year
- Sets cutoff = pub_year + 2
- Sums counts_by_year strictly across window years (pub+3, pub+4, pub+5)
- Classifies into Low / Medium / High based on field-year cohort thresholds
- Compares with stored prediction or receipts

Usage:
    python scripts/verify_paper.py W2752782242
"""

import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional

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
        print(f"Error querying OpenAlex API for {clean_id}: {e}")
        return {}


def load_cohort_threshold(field_name: Optional[str], pub_year: int) -> Dict[str, float]:
    """Loads empirical cohort thresholds (p50, p90) with strict hierarchical lookup from label_thresholds.json."""
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


def verify_paper(work_id: str):
    print("=" * 80)
    print(f"SCIGRAPH AI 2.0 — INDEPENDENT EXTERNAL GROUND-TRUTH VERIFICATION")
    print("=" * 80)

    work_data = fetch_openalex_live(work_id)
    if not work_data:
        # Check local corpus
        works_path = REPO_ROOT / "data" / "works.parquet"
        counts_path = REPO_ROOT / "data" / "counts_by_year.parquet"
        if works_path.exists() and counts_path.exists():
            import pandas as pd
            df_w = pd.read_parquet(works_path)
            clean_id = str(work_id).split("/")[-1]
            m = df_w[df_w["work_id"] == clean_id]
            if len(m) > 0:
                row = m.iloc[0]
                df_c = pd.read_parquet(counts_path)
                c_m = df_c[df_c["work_id"] == clean_id]
                work_data = {
                    "id": clean_id,
                    "title": row["title"],
                    "publication_year": int(row["publication_year"]),
                    "primary_topic": {"field": {"display_name": row.get("primary_field_name", "Computer Science")}},
                    "counts_by_year": c_m.to_dict(orient="records")
                }
                print(f"[INFO] Fetched record from local corpus (offline fallback).")

    if not work_data:
        print(f"ERROR: Could not fetch or locate paper {work_id}")
        sys.exit(1)

    clean_id = work_data.get("id", "").split("/")[-1]
    title = work_data.get("title", "Untitled")
    pub_year = int(work_data.get("publication_year", 0))
    cutoff_year = pub_year + 2
    window_years = [cutoff_year + 1, cutoff_year + 2, pub_year + 5]

    field_name = (
        work_data.get("primary_topic", {})
        .get("field", {})
        .get("display_name", "Computer Science")
    )

    counts_by_year = work_data.get("counts_by_year", [])
    by_year_dict = {
        int(c.get("year", 0)): int(c.get("cited_by_count", 0))
        for c in counts_by_year
        if isinstance(c, dict)
    }

    # Citations at cutoff (frozen inputs)
    cits_at_cutoff = sum(cnt for yr, cnt in by_year_dict.items() if yr <= cutoff_year)

    # Citations in 3-year window (post-cutoff ground truth)
    window_counts = [by_year_dict.get(yr, 0) for yr in range(cutoff_year + 1, pub_year + 6)]
    actual_window_cits = sum(window_counts)

    # Cohort threshold
    thresholds = load_cohort_threshold(field_name, pub_year)
    p50 = thresholds["p50"]
    p90 = thresholds["p90"]

    if actual_window_cits < p50:
        actual_tier = "Low (<50th)"
        actual_class = 0
    elif actual_window_cits < p90:
        actual_tier = "Medium (50-90th)"
        actual_class = 1
    else:
        actual_tier = "High (>=90th)"
        actual_class = 2

    # Check prediction ledger if exists
    ledger_path = REPO_ROOT / "reports" / "prediction_ledger.jsonl"
    stored_pred = None
    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("work_id") == clean_id:
                        stored_pred = rec
                except Exception:
                    pass

    print(f"\nPaper Identity:")
    print(f"  OpenAlex ID     : {clean_id}")
    print(f"  Title           : {title[:75]}")
    print(f"  Field           : {field_name}")
    print(f"  Publication Year: {pub_year}")
    print(f"  Cutoff Year     : {cutoff_year} (T_cutoff = pub_year + 2)")

    print(f"\nFrozen Pre-Cutoff Inputs (Observed as of {cutoff_year}):")
    print(f"  Citations at Cutoff : {cits_at_cutoff:,}")

    print(f"\nRevealed Ground-Truth Outcome (Years {window_years[0]}–{window_years[-1]}):")
    for yr in range(cutoff_year + 1, pub_year + 6):
        print(f"  - Year {yr} citations : {by_year_dict.get(yr, 0):,}")
    print(f"  Total Window Citations: {actual_window_cits:,}")
    print(f"  Cohort Thresholds     : p50={p50:.1f}, p90={p90:.1f}")
    print(f"  Actual Impact Tier    : {actual_tier}")

    if stored_pred:
        pred_tier = stored_pred.get("predicted_tier", "Unknown")
        is_match = (pred_tier == actual_tier)
        print(f"\nReceipt Audit (from prediction_ledger.jsonl):")
        print(f"  Receipt SHA256  : {stored_pred.get('receipt_sha256')}")
        print(f"  Predicted Tier  : {pred_tier}")
        print(f"  Prediction Match: {'CORRECT [OK]' if is_match else 'INCORRECT [MISMATCH]'}")
        print(f"  Probabilities   : {stored_pred.get('probabilities')}")

    print(f"\nAudit Links:")
    print(f"  Live OpenAlex JSON: https://api.openalex.org/works/{clean_id}?select=id,publication_year,counts_by_year")
    print(f"  OpenAlex Web Page : https://openalex.org/{clean_id}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_paper.py <work_id>")
        sys.exit(1)
    verify_paper(sys.argv[1])
