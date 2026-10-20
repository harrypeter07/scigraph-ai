"""Acquire unlabelled real reference neighbors without changing target labels.

The selection uses IDs only, never citation impact labels or future performance.
Histories must reconcile to the API total before being used as graph features.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import pandas as pd
import requests


def enrich(data_dir, maximum=40000):
    works = pd.read_parquet(data_dir / "works.parquet")
    edges = pd.read_parquet(data_dir / "citation_edges.parquet")
    known = set(works.work_id)
    # Give each root up to two chances of receiving an observed reference node.
    candidates = set()
    for _, group in edges.groupby("citing_work_id", sort=True):
        ids = [i for i in group.referenced_work_id if i not in known]
        ids.sort(key=lambda i: hashlib.sha256(i.encode()).digest())
        candidates.update(ids[:2])
    ids = sorted(candidates, key=lambda i: hashlib.sha256(i.encode()).digest())[:maximum]
    raw_dir = data_dir / "neighbor_shards"
    raw_dir.mkdir(exist_ok=True)
    key = os.environ.get("OPENALEX_API_KEY")
    def fetch(start):
        subset = ids[start:start+100]
        signature = hashlib.sha256("|".join(subset).encode()).hexdigest()[:20]
        destination = raw_dir / f"{signature}.json"
        if destination.exists():
            return json.loads(destination.read_text(encoding="utf-8"))
        params = dict(filter="openalex:"+"|".join(i.split("/")[-1] for i in subset), per_page=100,
                      select="id,title,publication_year,primary_topic,counts_by_year,cited_by_count,referenced_works")
        if key:
            params["api_key"] = key
        for attempt in range(5):
            try:
                response = requests.get("https://api.openalex.org/works", params=params, timeout=45)
                if response.status_code == 200:
                    results = response.json()["results"]
                    destination.write_text(json.dumps(results), encoding="utf-8")
                    return results
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"OpenAlex HTTP {response.status_code}; credentials/URL withheld")
            except (requests.RequestException, ValueError):
                pass
            time.sleep(min(16, 2**attempt))
        raise RuntimeError("Neighbor request failed; rerun to resume saved shards")
    raw = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, start) for start in range(0, len(ids), 100)]
        for count, future in enumerate(as_completed(futures), 1):
            raw.extend(future.result())
            if count % 50 == 0:
                print("Fetched neighbor batches", count, "of", len(futures), flush=True)
    records, counts, extra_edges = [], [], []
    excluded = 0
    for work in raw:
        if work.get("publication_year") is None:
            excluded += 1
            continue
        history = work.get("counts_by_year") or []
        if sum(c["cited_by_count"] for c in history) != work.get("cited_by_count", 0):
            excluded += 1
            continue
        topic = work.get("primary_topic") or {}
        records.append(dict(work_id=work["id"], title=work.get("title"), publication_year=work["publication_year"],
                            primary_field_name=(topic.get("field") or {}).get("display_name") or "Unknown",
                            num_authors=None, referenced_works_count=len(work.get("referenced_works") or [])))
        # Include a zero sentinel so a verified uncited neighbor has known history.
        if not history:
            history = [{"year": int(work["publication_year"]), "cited_by_count": 0}]
        counts.extend(dict(work_id=work["id"], year=c["year"], cited_by_count=c["cited_by_count"]) for c in history)
        extra_edges.extend(dict(citing_work_id=work["id"], referenced_work_id=i) for i in work.get("referenced_works") or [] if i in known)
    pd.DataFrame(records).drop_duplicates("work_id").to_parquet(data_dir / "neighbor_works.parquet", index=False)
    pd.DataFrame(counts).drop_duplicates(["work_id", "year"]).to_parquet(data_dir / "neighbor_counts_by_year.parquet", index=False)
    pd.DataFrame(extra_edges, columns=["citing_work_id", "referenced_work_id"]).drop_duplicates().to_parquet(data_dir / "neighbor_edges.parquet", index=False)
    audit = dict(requested=len(ids), returned=len(raw), accepted=len(records), unreconciled_history_excluded=excluded,
                 selection="up to two hash-selected references per root; globally capped by ID hash",
                 labels="none; neighbors are not additional supervised training examples")
    (data_dir / "neighbor_audit.json").write_text(json.dumps(audit, indent=2))
    print(audit, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/expanded_forecast"))
    parser.add_argument("--maximum", type=int, default=40000)
    args = parser.parse_args()
    enrich(args.data_dir, args.maximum)
