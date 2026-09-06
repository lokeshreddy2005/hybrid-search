"""Small helper for authoring honest eval labels: search the corpus by
substring in title/text and print matching (id, title) pairs, so query
authors can find the real doc_ids that are actually relevant to a query
instead of guessing.

Usage:
    python scripts/inspect_corpus.py "Albert Einstein"
    python scripts/inspect_corpus.py --field title "Einstein"
"""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("term")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--field", choices=["title", "text", "both"], default="both")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    term = args.term.lower()
    n = 0
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            haystack = ""
            if args.field in ("title", "both"):
                haystack += doc["title"].lower()
            if args.field in ("text", "both"):
                haystack += " " + doc["text"].lower()
            if term in haystack:
                print(f"{doc['id']}\t{doc['title']}")
                n += 1
                if n >= args.limit:
                    break
    print(f"\n{n} matches shown.")


if __name__ == "__main__":
    main()
