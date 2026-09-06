"""Build the inverted index (BM25) and the dense vector index (FAISS) from
data/corpus.jsonl, and write everything the API/eval need to index/store/.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from index.inverted_index import InvertedIndex
from retrieval.embedder import DEFAULT_MODEL_NAME, Embedder
from retrieval.vector_index import VectorIndex


def load_corpus(path: str) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--store-dir", default="index/store")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {args.corpus} ...")
    docs = load_corpus(args.corpus)
    print(f"Loaded {len(docs)} documents.")

    # ---- BM25 / inverted index ----
    print("Building inverted index (tokenization + postings)...")
    t0 = time.perf_counter()
    inv_index = InvertedIndex()
    triples = [(d["id"], d["title"], d["text"]) for d in docs]
    inv_index.build(triples)
    print(f"  {len(inv_index.postings)} unique terms, "
          f"avg doc length {inv_index.avg_doc_length:.1f} tokens, "
          f"took {time.perf_counter() - t0:.1f}s")
    inv_index.save(store_dir / "inverted_index.pkl")

    # ---- Dense embeddings + FAISS ----
    print(f"Loading embedding model {args.embedding_model} ...")
    embedder = Embedder(args.embedding_model)

    print("Encoding documents (title + text) ...")
    t0 = time.perf_counter()
    texts = [f"{d['title']}. {d['text']}" for d in docs]
    embeddings = embedder.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    print(f"  encoded {embeddings.shape[0]} docs -> dim {embeddings.shape[1]}, "
          f"took {time.perf_counter() - t0:.1f}s")

    vector_index = VectorIndex(dim=embedder.dim)
    vector_index.build(embeddings, doc_ids=[d["id"] for d in docs], doc_titles=[d["title"] for d in docs])
    vector_index.save(store_dir / "vector_index")

    # ---- doc store for API result rendering ----
    doc_texts = {d["id"]: {"title": d["title"], "text": d["text"]} for d in docs}
    with open(store_dir / "doc_texts.json", "w", encoding="utf-8") as f:
        json.dump(doc_texts, f, ensure_ascii=False)

    with open(store_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_docs": len(docs),
                "embedding_model": args.embedding_model,
                "embedding_dim": embedder.dim,
                "avg_doc_length": inv_index.avg_doc_length,
                "n_terms": len(inv_index.postings),
            },
            f,
            indent=2,
        )

    print(f"Index build complete. Artifacts written to {store_dir}/")


if __name__ == "__main__":
    main()
