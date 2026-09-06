"""Simple concurrent load test against a running API instance.

Fires requests at fixed concurrency for a fixed wall-clock duration,
cycling through the evaluation query set, and reports throughput and
latency percentiles.

Usage (API must already be running, e.g. `uvicorn api.main:app`):
    python scripts/benchmark.py --url http://localhost:8000 --mode hybrid \
        --concurrency 8 --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

DEFAULT_QUERIES_FILE = "eval/eval_queries.json"


def load_queries(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["query"] for item in data]


async def worker(client: httpx.AsyncClient, queries: list[str], mode: str, top_k: int,
                  end_time: float, latencies: list[float], errors: list[int], idx: int):
    i = idx
    while time.perf_counter() < end_time:
        query = queries[i % len(queries)]
        i += 1
        start = time.perf_counter()
        try:
            resp = await client.get("/search", params={"q": query, "mode": mode, "top_k": top_k}, timeout=30.0)
            resp.raise_for_status()
        except Exception:
            errors[0] += 1
            continue
        latencies.append(time.perf_counter() - start)


async def run_benchmark(url: str, queries: list[str], mode: str, top_k: int,
                         concurrency: int, duration: float) -> dict:
    latencies: list[float] = []
    errors = [0]
    async with httpx.AsyncClient(base_url=url) as client:
        # warm up once so the first real request isn't paying for e.g. lazy model init
        await client.get("/search", params={"q": queries[0], "mode": mode, "top_k": top_k}, timeout=30.0)

        start_time = time.perf_counter()
        end_time = start_time + duration
        tasks = [
            asyncio.create_task(worker(client, queries, mode, top_k, end_time, latencies, errors, idx))
            for idx in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_time

    n = len(latencies)
    sorted_lat = sorted(latencies)

    def pct(p):
        if not sorted_lat:
            return float("nan")
        k = min(len(sorted_lat) - 1, int(len(sorted_lat) * p))
        return sorted_lat[k]

    return {
        "mode": mode,
        "concurrency": concurrency,
        "duration_s": round(wall_time, 2),
        "n_requests": n,
        "n_errors": errors[0],
        "qps": round(n / wall_time, 2) if wall_time > 0 else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 2) if latencies else None,
            "p50": round(pct(0.50) * 1000, 2),
            "p95": round(pct(0.95) * 1000, 2),
            "p99": round(pct(0.99) * 1000, 2),
            "max": round(max(latencies) * 1000, 2) if latencies else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--queries-file", default=DEFAULT_QUERIES_FILE)
    parser.add_argument("--out", default="benchmark_results/results.json")
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    print(f"Loaded {len(queries)} queries. Running {args.mode} mode, "
          f"concurrency={args.concurrency}, duration={args.duration}s ...")

    result = asyncio.run(run_benchmark(args.url, queries, args.mode, args.top_k, args.concurrency, args.duration))

    print(json.dumps(result, indent=2))

    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(result)
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Appended result to {out_path}")


if __name__ == "__main__":
    main()
