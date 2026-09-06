"""FAISS-backed dense vector index.

Uses IndexFlatIP (exact brute-force inner-product search) over
L2-normalized embeddings, which is exact cosine similarity. Corpus size
here (tens of thousands of docs, 384-dim) is small enough that an
approximate index (IVF/HNSW) would only trade accuracy for a speedup we
don't need — see README limitations for scaling notes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids: list[str] = []       # internal position -> external doc id
        self.doc_titles: list[str] = []

    def build(self, embeddings: np.ndarray, doc_ids: list[str], doc_titles: list[str]) -> None:
        assert embeddings.shape[0] == len(doc_ids) == len(doc_titles)
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings)
        self.doc_ids = list(doc_ids)
        self.doc_titles = list(doc_titles)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Returns [(internal_doc_id, cosine_similarity), ...] top_k results."""
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def save(self, path: str | Path) -> None:
        import faiss
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        meta = {"dim": self.dim, "doc_ids": self.doc_ids, "doc_titles": self.doc_titles}
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @staticmethod
    def load(path: str | Path) -> "VectorIndex":
        import faiss
        import json

        path = Path(path)
        with open(path.with_suffix(".meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        vi = VectorIndex(dim=meta["dim"])
        vi.index = faiss.read_index(str(path.with_suffix(".faiss")))
        vi.doc_ids = meta["doc_ids"]
        vi.doc_titles = meta["doc_titles"]
        return vi
