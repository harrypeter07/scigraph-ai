"""Empirical API Latency & Robustness Benchmark (WP11).

Executes 200 requests across diverse endpoints:
- In-corpus cached works (ResNet, TON_IoT, etc.)
- In-corpus sampled works from works.parquet (150 papers)
- Out-of-corpus / unknown works (testing 404/422 status code correctness)

Measures exact wall-clock response latencies and computes empirical p50, p90, p99.
Outputs: reports/api_latency.json
"""

import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.config import settings


def run_latency_benchmark(n_requests: int = 200, server_url: str = "http://127.0.0.1:8080"):
    print("=" * 80, flush=True)
    print(f"API SERVING LATENCY & ROBUSTNESS BENCHMARK ({n_requests} REQUESTS)", flush=True)
    print("=" * 80, flush=True)

    # 1. Collect candidate paper identifiers
    # Cached works
    cached_ids = ["W2194775991", "W3085955590", "W3118615836", "W2142159465"]
    
    # In-corpus works from works.parquet
    works_path = settings.DATA_DIR / "works.parquet"
    if works_path.exists():
        df_works = pd.read_parquet(works_path, columns=["work_id"])
        corpus_sample = [str(x).split("/")[-1] for x in df_works["work_id"].sample(n=min(140, len(df_works)), random_state=42)]
    else:
        corpus_sample = []

    # Unknown / edge-case identifiers (should return 404 or 422, not 500)
    unknown_ids = [
        "W999999999999",
        "W000000000000",
        "invalid-id-xyz",
        "10.9999/nonexistent.doi.123",
        "W888888888888",
    ]

    # Combine into 200 requests pool
    request_pool = []
    # Repeat cached works
    while len(request_pool) < 40:
        request_pool.extend(cached_ids)
    
    # Add corpus sample
    request_pool.extend(corpus_sample[:130])

    # Add unknown works
    while len(request_pool) < n_requests:
        request_pool.extend(unknown_ids)
    
    request_pool = request_pool[:n_requests]
    print(f"Constructed request pool with {len(request_pool)} items.", flush=True)

    latencies_ms = []
    status_counts = {}

    print("\nSending requests to /api/v1/paper/predict/...", flush=True)
    t_start_total = time.time()

    for idx, pid in enumerate(request_pool):
        url = f"{server_url}/api/v1/paper/predict/{pid}"
        t0 = time.perf_counter()
        status_code = None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SciGraphLatencyBenchmark/2.0"})
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status_code = resp.getcode()
                _ = resp.read()
        except urllib.error.HTTPError as e:
            status_code = e.code
        except Exception as e:
            status_code = 500
            print(f"Request {idx} failed: {e}", flush=True)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        status_counts[status_code] = status_counts.get(status_code, 0) + 1

        if (idx + 1) % 25 == 0 or idx == n_requests - 1:
            print(f"  Processed {idx + 1}/{n_requests} requests (last: {dt_ms:.1f}ms, status: {status_code})", flush=True)
            print(f"  Processed {idx + 1}/{n_requests} requests (last: {dt_ms:.1f}ms, status: {status_code})")

    total_duration = time.time() - t_start_total
    latencies_arr = np.array(latencies_ms)

    p50 = float(np.percentile(latencies_arr, 50))
    p90 = float(np.percentile(latencies_arr, 90))
    p99 = float(np.percentile(latencies_arr, 99))
    p_mean = float(np.mean(latencies_arr))
    p_min = float(np.min(latencies_arr))
    p_max = float(np.max(latencies_arr))

    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Requests Total:   {len(latencies_ms)}")
    print(f"Total Duration:   {total_duration:.2f} s")
    print(f"Throughput:       {len(latencies_ms) / total_duration:.1f} req/s")
    print(f"Status Codes:     {status_counts}")
    print(f"Latency p50:      {p50:.2f} ms")
    print(f"Latency p90:      {p90:.2f} ms")
    print(f"Latency p99:      {p99:.2f} ms")
    print(f"Latency Mean:     {p_mean:.2f} ms (Min: {p_min:.2f} ms, Max: {p_max:.2f} ms)")

    # Save reports/api_latency.json
    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_requests": len(latencies_ms),
            "server_url": server_url,
            "test_endpoint": "/api/v1/paper/predict/{id}"
        },
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(p_mean, 2),
        "min_ms": round(p_min, 2),
        "max_ms": round(p_max, 2),
        "status_code_distribution": status_counts,
        "throughput_req_per_sec": round(len(latencies_ms) / max(1e-4, total_duration), 2)
    }

    out_path = settings.BASE_DIR / "reports" / "api_latency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved latency report to {out_path}")


if __name__ == "__main__":
    run_latency_benchmark()
