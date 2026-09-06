"""Run the hand-labeled evaluation query set through all three retrieval
modes (bm25 / vector / hybrid), compute precision@k and recall@k for each,
print a comparison table, and write eval/results.json.

Usage:
    python scripts/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import precision_at_k, recall_at_k
from retrieval.engine import VALID_MODES, SearchEngine

K_VALUES = (5, 10)


def load_eval_queries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_mode(engine: SearchEngine, queries: list[dict], mode: str, top_k: int) -> dict:
    per_query = []
    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_ids"])
        results = engine.search(query, mode=mode, top_k=top_k)
        retrieved_ids = [r.doc_id for r in results]

        row = {"id": item["id"], "query": query, "retrieved": retrieved_ids}
        for k in K_VALUES:
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        per_query.append(row)

    aggregate = {}
    for k in K_VALUES:
        aggregate[f"precision@{k}"] = sum(r[f"precision@{k}"] for r in per_query) / len(per_query)
        aggregate[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_query) / len(per_query)

    return {"mode": mode, "aggregate": aggregate, "per_query": per_query}


def print_comparison_table(results_by_mode: dict[str, dict]):
    header = ["mode"] + [f"{metric}@{k}" for metric in ("precision", "recall") for k in K_VALUES]
    rows = []
    for mode in VALID_MODES:
        agg = results_by_mode[mode]["aggregate"]
        row = [mode] + [f"{agg[f'{metric}@{k}']:.3f}" for metric in ("precision", "recall") for k in K_VALUES]
        rows.append(row)

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    def fmt_row(row):
        return " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths))

    print(fmt_row(header))
    print("-|-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))

    print("\nMarkdown:\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default="index/store")
    parser.add_argument("--queries", default="eval/eval_queries.json")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--top-k", type=int, default=max(K_VALUES))
    args = parser.parse_args()

    print(f"Loading search engine from {args.store_dir} ...")
    engine = SearchEngine.load(args.store_dir)

    queries = load_eval_queries(args.queries)
    print(f"Loaded {len(queries)} evaluation queries.\n")

    results_by_mode = {}
    for mode in VALID_MODES:
        t0 = time.perf_counter()
        results_by_mode[mode] = evaluate_mode(engine, queries, mode, args.top_k)
        print(f"  evaluated mode={mode} in {time.perf_counter() - t0:.2f}s")

    print()
    print_comparison_table(results_by_mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_by_mode, f, indent=2)
    print(f"\nFull per-query results written to {out_path}")


if __name__ == "__main__":
    main()
