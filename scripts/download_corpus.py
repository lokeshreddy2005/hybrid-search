"""Download a public corpus and write data/corpus.jsonl.

Source: Simple English Wikipedia (wikimedia/wikipedia, config
20231101.simple), streamed from Hugging Face. Chosen over full English
Wikipedia because Simple English articles are short, self-contained, and
cover recognizable general-knowledge topics (countries, historical
figures, science concepts, etc.) — which makes it realistic to author
honest, hand-labeled relevance judgments for the evaluation set later
(vs. MS MARCO passages, which are decontextualized fragments that are
harder to relevance-judge by inspection).

Each output row is {"id": ..., "title": ..., "text": ...} where `text`
is the article's lead content, truncated to an abstract-length excerpt.

Usage:
    python scripts/download_corpus.py --n-docs 30000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

MIN_TEXT_CHARS = 200      # skip stubs/redirect-only pages
MAX_ABSTRACT_CHARS = 1500  # keep documents abstract-sized


def make_abstract(text: str, max_chars: int = MAX_ABSTRACT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return truncated.rsplit(" ", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=30000)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.simple")
    parser.add_argument("--out", default="data/corpus.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {args.dataset} ({args.config}) from Hugging Face...")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    n_written = 0
    n_seen = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            n_seen += 1
            text = row.get("text", "") or ""
            title = row.get("title", "") or ""
            abstract = make_abstract(text)
            if len(abstract) < MIN_TEXT_CHARS or not title:
                continue
            doc = {"id": f"simplewiki-{row['id']}", "title": title, "text": abstract}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 5000 == 0:
                print(f"  written {n_written} docs (scanned {n_seen})...")
            if n_written >= args.n_docs:
                break

    print(f"Done. Wrote {n_written} documents to {out_path} (scanned {n_seen} source rows).")


if __name__ == "__main__":
    main()
