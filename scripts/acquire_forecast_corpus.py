"""Resumable random field/year sampling; keeps the original corpus untouched.

Accepts only histories whose annual sum reconciles to current citation totals.
This permits explicit zero years without guessing missing historical counts.
Reports exclusions and tie-driven class imbalance. No citation-ranked sampling.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata
from concurrent.futures import ThreadPoolExecutor, as_completed

FIELDS = {17: "Computer Science", 27: "Medicine", 31: "Physics and Astronomy",
          13: "Biochemistry, Genetics and Molecular Biology", 22: "Engineering"}


def fetch_cohort(output, field_id, year, per_cohort, api_key):
    path = output / f"sample_{field_id}_{year}.jsonl"
    if path.exists():
        return
    session = requests.Session()
    raw = []
    for page in range(1, (per_cohort+99)//100 + 1):
        params = dict(filter=f"primary_topic.field.id:{field_id},publication_year:{year}",
                      sample=per_cohort, seed=42, per_page=100, page=page,
                      select="id,title,doi,publication_year,counts_by_year,cited_by_count,referenced_works")
        if api_key:
            params["api_key"] = api_key
        for attempt in range(5):
            try:
                response = session.get("https://api.openalex.org/works", params=params, timeout=45)
                if response.status_code == 200:
                    payload = response.json()
                    break
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"OpenAlex HTTP {response.status_code}; credentials/URL withheld")
            except (requests.RequestException, ValueError):
                pass
            time.sleep(min(16, 2**attempt))
        else:
            raise RuntimeError(f"Retry limit for field {field_id}/{year}/page {page}; rerun to resume")
        raw.extend(payload.get("results", []))
        time.sleep(.15)
    path.write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    print("Downloaded cohort", field_id, year, len(raw), flush=True)


def acquire(output, per_cohort=1200, exclude=None):
    output.mkdir(parents=True, exist_ok=True)
    forbidden = set()
    if exclude and exclude.exists():
        forbidden = set(pd.read_parquet(exclude, columns=["work_id"]).work_id)
    api_key = os.environ.get("OPENALEX_API_KEY")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_cohort, output, f, year, per_cohort, api_key)
                   for year in range(2012, 2021) for f in FIELDS]
        for future in as_completed(futures):
            future.result()
    accepted, annual, edges, audit = [], [], [], []
    for year in range(2012, 2021):
        for field_id, field_name in FIELDS.items():
            raw_path = output / f"sample_{field_id}_{year}.jsonl"
            raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
            excluded = dict(previous_corpus=0, incomplete_history=0, duplicate=0)
            cohort = []
            seen = set()
            for work in raw:
                wid = work["id"]
                if wid in forbidden:
                    excluded["previous_corpus"] += 1
                    continue
                if wid in seen:
                    excluded["duplicate"] += 1
                    continue
                seen.add(wid)
                counts = work.get("counts_by_year") or []
                if sum(int(c.get("cited_by_count", 0)) for c in counts) != int(work.get("cited_by_count", 0)):
                    excluded["incomplete_history"] += 1
                    continue
                lookup = {int(c["year"]): int(c["cited_by_count"]) for c in counts}
                early = sum(c for y, c in lookup.items() if y <= year+2)
                future = sum(c for y, c in lookup.items() if year+2 < y <= year+5)
                source = (work.get("primary_location") or {}).get("source") or {}
                cohort.append(dict(work_id=wid, title=work.get("title") or "", doi=work.get("doi"),
                                   publication_year=year, primary_field_name=field_name,
                                   num_authors=len(work["authorships"]) if "authorships" in work else None,
                                   referenced_works_count=len(work.get("referenced_works") or []),
                                   venue_name=source.get("display_name"), oa_status=(work.get("open_access") or {}).get("oa_status"),
                                   citations_at_cutoff=early, future_citations_5y=future,
                                   temporal_split="train" if year<=2016 else "val" if year==2017 else "test"))
                # Explicit zeros are supported by reconciled complete history.
                annual.extend(dict(work_id=wid, year=y, cited_by_count=lookup.get(y, 0)) for y in range(year, 2027))
                edges.extend(dict(citing_work_id=wid, referenced_work_id=ref) for ref in work.get("referenced_works") or [])
            if len(cohort) < 30:
                raise ValueError(f"Too few verified works in {field_name}/{year}; received {len(cohort)}")
            percentiles = rankdata([r["future_citations_5y"] for r in cohort], method="average")/len(cohort)
            for row, percentile in zip(cohort, percentiles):
                row["impact_class"] = 0 if percentile < .5 else 1 if percentile < .9 else 2
            accepted.extend(cohort)
            audit.append(dict(field=field_name, year=year, sampled=len(raw), accepted=len(cohort), excluded=excluded,
                              classes={str(k): sum(r["impact_class"]==k for r in cohort) for k in range(3)}))
            print(field_name, year, len(cohort), "accepted", flush=True)
    pd.DataFrame(accepted).to_parquet(output / "works.parquet", index=False)
    pd.DataFrame(annual).to_parquet(output / "counts_by_year.parquet", index=False)
    pd.DataFrame(edges, columns=["citing_work_id", "referenced_work_id"]).drop_duplicates().to_parquet(output / "citation_edges.parquet", index=False)
    (output / "acquisition_audit.json").write_text(json.dumps(dict(
        sampling="seeded random field/year samples; includes uncited papers; original corpus IDs excluded",
        label_definition="rank of citations during publication+3 through publication+5 within sampled field/year; average ranks for ties",
        limitation="sample-relative citation tiers, not population-wide disciplinary percentiles; exclusions may introduce bias",
        cohorts=audit), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/expanded_forecast"))
    parser.add_argument("--per-cohort", type=int, default=1200)
    parser.add_argument("--exclude", type=Path, default=Path("data/works.parquet"))
    args = parser.parse_args()
    if not 200 <= args.per_cohort <= 10000:
        parser.error("per-cohort must be between 200 and 10000")
    acquire(args.output, args.per_cohort, args.exclude)
