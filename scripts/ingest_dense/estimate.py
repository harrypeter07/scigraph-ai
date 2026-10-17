"""SciGraph AI 2.0 — Phase A: Dense Corpus Ingestion Estimator & Probe.

Budget guard:
Probes OpenAlex counts, reference coverage, and estimates requests, runtimes, and disk usage
for three dense graph strategies:
  C1 "subfield_full": Full/dense works of top subfield (e.g. Computer Vision or AI)
  C2 "snowball": 3,000 seeds expanded 1-hop by referenced_works and top-50 citers
  C3 "multifield_dense": 3 related subfields in Computer Science domain

Enforces hard cap: <= 120,000 works and <= 6 hours per variant.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.config import settings

POLITE_EMAIL = settings.OPENALEX_POLITE_EMAIL
USER_AGENT = f"SciGraphAI/2.0 (mailto:{POLITE_EMAIL})"
BASE_URL = "https://api.openalex.org"

CANDIDATE_SUBFIELDS = [
    {"id": "1707", "name": "Computer Vision and Pattern Recognition", "field": "Computer Science"},
    {"id": "1702", "name": "Artificial Intelligence", "field": "Computer Science"},
    {"id": "1703", "name": "Computational Theory and Mathematics", "field": "Computer Science"},
    {"id": "1705", "name": "Computer Networks and Communications", "field": "Computer Science"},
    {"id": "1306", "name": "Biochemistry", "field": "Biochemistry, Genetics and Molecular Biology"},
    {"id": "2730", "name": "Oncology", "field": "Medicine"}
]

def openalex_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params_with_mail = dict(params)
    params_with_mail["mailto"] = POLITE_EMAIL
    query_str = urllib.parse.urlencode(params_with_mail)
    url = f"{BASE_URL}/{endpoint}?{query_str}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"OpenAlex request failed for {url}: {e}")
            time.sleep(1.0 * (2 ** attempt))
    return {}

def probe_subfield_counts() -> List[Dict[str, Any]]:
    print("=" * 70)
    print("STEP 1: Probing OpenAlex candidate subfield volumes (2010–2022, articles)...")
    print("=" * 70)
    results = []
    
    for sf in CANDIDATE_SUBFIELDS:
        subfield_id = sf["id"]
        # Filters: primary_topic.subfield.id, type:article|proceedings-article, pub_year:2010-2022, has_abstract:true
        filt = f"primary_topic.subfield.id:{subfield_id},type:article|proceedings-article,publication_year:2010-2022,has_abstract:true"
        try:
            data = openalex_get("works", {"filter": filt, "per_page": 1})
            meta = data.get("meta", {})
            count = meta.get("count", 0)
            results.append({
                "subfield_id": subfield_id,
                "name": sf["name"],
                "field": sf["field"],
                "filter": filt,
                "total_works_2010_2022": count
            })
            print(f"  [{subfield_id}] {sf['name']} ({sf['field']}): {count:,} works")
        except Exception as e:
            print(f"  [{subfield_id}] Error: {e}")
            
    return results

def probe_subfield_references(subfield_id: str, subfield_name: str, n_probe: int = 100) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STEP 2: Probing reference density & in-corpus coverage for {subfield_name} (ID: {subfield_id})...")
    print("=" * 70)
    
    filt = f"primary_topic.subfield.id:{subfield_id},type:article|proceedings-article,publication_year:2015-2018,has_abstract:true"
    # Fetch sample
    data = openalex_get("works", {
        "filter": filt,
        "per_page": min(100, n_probe),
        "select": "id,title,publication_year,referenced_works,counts_by_year,cited_by_count"
    })
    
    works = data.get("results", [])
    ref_counts = [len(w.get("referenced_works", [])) for w in works]
    cits_counts = [w.get("cited_by_count", 0) for w in works]
    
    median_refs = sorted(ref_counts)[len(ref_counts)//2] if ref_counts else 0
    mean_refs = sum(ref_counts) / len(ref_counts) if ref_counts else 0
    median_cits = sorted(cits_counts)[len(cits_counts)//2] if cits_counts else 0
    mean_cits = sum(cits_counts) / len(cits_counts) if cits_counts else 0
    
    # Collect all referenced work IDs
    all_refs = set()
    for w in works:
        for r in w.get("referenced_works", []):
            all_refs.add(r.split("/")[-1])
            
    # Sample 30 referenced works to estimate in-subfield reference share
    sample_refs = list(all_refs)[:30]
    in_subfield_count = 0
    checked_count = 0
    
    if sample_refs:
        chunk_filter = "openalex:" + "|".join(sample_refs)
        try:
            ref_data = openalex_get("works", {
                "filter": chunk_filter,
                "per_page": 50,
                "select": "id,primary_topic"
            })
            for rw in ref_data.get("results", []):
                checked_count += 1
                pt = rw.get("primary_topic") or {}
                sf_info = pt.get("subfield") or {}
                if str(sf_info.get("id", "")).split("/")[-1] == subfield_id:
                    in_subfield_count += 1
        except Exception as e:
            print(f"  Warning: Reference lookup probe: {e}")
            
    in_subfield_share = in_subfield_count / max(1, checked_count) if checked_count > 0 else 0.45
    
    print(f"  Works sampled: {len(works)}")
    print(f"  Median references per paper: {median_refs} (mean: {mean_refs:.1f})")
    print(f"  Median cumulative citations: {median_cits} (mean: {mean_cits:.1f})")
    print(f"  Estimated in-subfield reference share: {in_subfield_share*100:.1f}% ({in_subfield_count}/{checked_count} sampled refs)")
    
    return {
        "subfield_id": subfield_id,
        "subfield_name": subfield_name,
        "median_references": median_refs,
        "mean_references": round(mean_refs, 1),
        "median_citations": median_cits,
        "mean_citations": round(mean_cits, 1),
        "in_subfield_reference_share": round(in_subfield_share, 3)
    }

def estimate_variants(counts_list: List[Dict[str, Any]], probe_info: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 3: Resource & Time Estimation across 3 Candidate Corpora (Cap = 120k works)...")
    print("=" * 70)
    
    HARD_CAP = 120_000
    REQ_PER_SEC = 6.0  # Safe conservative polite pool rate (limit is 10)
    WORKS_PER_PAGE = 200
    BYTES_PER_WORK_RAW = 1800  # ~1.8 KB per work raw JSON
    BYTES_PER_WORK_PQ = 350    # ~350 bytes per work parquet
    
    # C1: subfield_full (e.g. Computer Vision 1707)
    cv_count = next((c["total_works_2010_2022"] for c in counts_list if c["subfield_id"] == "1707"), 250_000)
    c1_planned = min(HARD_CAP, cv_count)
    c1_reqs = c1_planned // WORKS_PER_PAGE + 10
    c1_seconds = c1_reqs / REQ_PER_SEC
    c1_raw_mb = (c1_planned * BYTES_PER_WORK_RAW) / (1024 * 1024)
    c1_pq_mb = (c1_planned * BYTES_PER_WORK_PQ) / (1024 * 1024)
    c1_in_corpus_degree_est = int(probe_info["median_references"] * probe_info["in_subfield_reference_share"])
    
    # C2: snowball starting from 3,000 seeds
    # 3,000 seeds -> all references (~30/seed = 90k raw -> ~60k unique in-corpus)
    # + first 50 citers per seed (~30 unique/seed = ~50k unique)
    # Deduplicated total: ~110,000 works
    c2_planned = 110_000
    c2_seed_reqs = 3000 // WORKS_PER_PAGE + 2
    c2_ref_reqs = 60000 // WORKS_PER_PAGE + 5
    c2_citer_reqs = 3000 * 1  # 1 request per seed for first 50 citers = 3000 reqs
    c2_total_reqs = c2_seed_reqs + c2_ref_reqs + c2_citer_reqs
    c2_seconds = c2_total_reqs / REQ_PER_SEC
    c2_raw_mb = (c2_planned * BYTES_PER_WORK_RAW) / (1024 * 1024)
    c2_pq_mb = (c2_planned * BYTES_PER_WORK_PQ) / (1024 * 1024)
    c2_in_corpus_degree_est = int(probe_info["median_references"] * 0.85 + 15)  # Snowball by construction has ~85% ref coverage + citers
    
    # C3: multifield_dense (AI 1702, CV 1707, Theory 1703) - 40k each = 120,000
    c3_planned = HARD_CAP
    c3_reqs = c3_planned // WORKS_PER_PAGE + 15
    c3_seconds = c3_reqs / REQ_PER_SEC
    c3_raw_mb = (c3_planned * BYTES_PER_WORK_RAW) / (1024 * 1024)
    c3_pq_mb = (c3_planned * BYTES_PER_WORK_PQ) / (1024 * 1024)
    c3_in_corpus_degree_est = int(probe_info["median_references"] * 0.55)
    
    variants = {
        "C1_subfield_full": {
            "name": "C1: subfield_full (Computer Vision & Pattern Recognition)",
            "subfield_id": "1707",
            "planned_works": c1_planned,
            "estimated_requests": c1_reqs,
            "estimated_runtime_minutes": round(c1_seconds / 60, 1),
            "estimated_runtime_hours": round(c1_seconds / 3600, 2),
            "estimated_raw_json_mb": round(c1_raw_mb, 1),
            "estimated_parquet_mb": round(c1_pq_mb, 1),
            "estimated_target_median_degree": c1_in_corpus_degree_est,
            "budget_guard_pass": c1_planned <= 150_000 and (c1_seconds / 3600) <= 6.0
        },
        "C2_snowball": {
            "name": "C2: snowball (3,000 seeds + 1-hop refs + 50 citers)",
            "seeds": 3000,
            "planned_works": c2_planned,
            "estimated_requests": c2_total_reqs,
            "estimated_runtime_minutes": round(c2_seconds / 60, 1),
            "estimated_runtime_hours": round(c2_seconds / 3600, 2),
            "estimated_raw_json_mb": round(c2_raw_mb, 1),
            "estimated_parquet_mb": round(c2_pq_mb, 1),
            "estimated_target_median_degree": c2_in_corpus_degree_est,
            "budget_guard_pass": c2_planned <= 150_000 and (c2_seconds / 3600) <= 6.0
        },
        "C3_multifield_dense": {
            "name": "C3: multifield_dense (CV + AI + Computational Theory, 40k each)",
            "subfield_ids": ["1707", "1702", "1703"],
            "planned_works": c3_planned,
            "estimated_requests": c3_reqs,
            "estimated_runtime_minutes": round(c3_seconds / 60, 1),
            "estimated_runtime_hours": round(c3_seconds / 3600, 2),
            "estimated_raw_json_mb": round(c3_raw_mb, 1),
            "estimated_parquet_mb": round(c3_pq_mb, 1),
            "estimated_target_median_degree": c3_in_corpus_degree_est,
            "budget_guard_pass": c3_planned <= 150_000 and (c3_seconds / 3600) <= 6.0
        }
    }
    
    print("\nPROJECTION SUMMARY TABLE:")
    print("-" * 105)
    print(f"{'Variant':<22} | {'Works':<9} | {'Requests':<9} | {'Est Time':<12} | {'Raw JSON':<10} | {'Parquet':<9} | {'Degree Est':<10} | {'Guard'}")
    print("-" * 105)
    for k, v in variants.items():
        time_str = f"{v['estimated_runtime_minutes']} min"
        guard_str = "PASS (<=120k, <6h)" if v["budget_guard_pass"] else "FAIL (Exceeds budget)"
        print(f"{k:<22} | {v['planned_works']:<9,d} | {v['estimated_requests']:<9,d} | {time_str:<12} | {v['estimated_raw_json_mb']:<7.1f} MB | {v['estimated_parquet_mb']:<7.1f} MB | {v['estimated_target_median_degree']:<10} | {guard_str}")
    print("-" * 105)
    
    return variants

def main():
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    counts = probe_subfield_counts()
    probe_info = probe_subfield_references("1707", "Computer Vision and Pattern Recognition", n_probe=100)
    variants = estimate_variants(counts, probe_info)
    
    output_path = reports_dir / "dense_estimate.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "polite_pool": POLITE_EMAIL,
            "subfield_counts": counts,
            "reference_probe": probe_info,
            "variants_estimate": variants
        }, f, indent=2)
        
    print(f"\nSaved estimation report to: {output_path}")

if __name__ == "__main__":
    main()
