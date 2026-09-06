"""Unified search engine: wraps BM25, dense vector search, and RRF fusion
behind one `search(query, mode, top_k)` interface used by both the API
and the evaluation script.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from index.bm25 import BM25
from index.inverted_index import InvertedIndex
from retrieval.embedder import DEFAULT_MODEL_NAME, Embedder
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.vector_index import VectorIndex

VALID_MODES = ("bm25", "vector", "hybrid")


@dataclass
class SearchResult:
    doc_id: str
    title: str
    snippet: str
    score: float
    rank: int


def _snippet(text: str, max_chars: int = 240) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


class SearchEngine:
    def __init__(
        self,
        inverted_index: InvertedIndex,
        vector_index: VectorIndex,
        embedder: Embedder,
        doc_texts: dict[str, dict],
        candidate_k: int = 100,
        rrf_k: int = 60,
    ):
        self.inverted_index = inverted_index
        self.bm25 = BM25(inverted_index)
        self.vector_index = vector_index
        self.embedder = embedder
        self.doc_texts = doc_texts
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    # -- individual retrievers, returning results keyed by external doc_id --

    def _bm25_ranked(self, query: str, top_k: int) -> list[tuple[str, float]]:
        hits = self.bm25.search(query, top_k=top_k)
        return [(self.inverted_index.doc_ids[i], score) for i, score in hits]

    def _vector_ranked(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_emb = self.embedder.encode([query])[0]
        hits = self.vector_index.search(query_emb, top_k=top_k)
        return [(self.vector_index.doc_ids[i], score) for i, score in hits]

    def _to_results(self, ranked: list[tuple[str, float]], top_k: int) -> list[SearchResult]:
        results = []
        for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
            doc = self.doc_texts.get(doc_id, {})
            title = doc.get("title", doc_id)
            text = doc.get("text", "")
            results.append(SearchResult(doc_id=doc_id, title=title, snippet=_snippet(text), score=score, rank=rank))
        return results

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10) -> list[SearchResult]:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        if mode == "bm25":
            ranked = self._bm25_ranked(query, top_k)
            return self._to_results(ranked, top_k)

        if mode == "vector":
            ranked = self._vector_ranked(query, top_k)
            return self._to_results(ranked, top_k)

        # hybrid: fetch a larger candidate pool from each retriever, then RRF-fuse
        bm25_ranked = self._bm25_ranked(query, self.candidate_k)
        vector_ranked = self._vector_ranked(query, self.candidate_k)
        fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=self.rrf_k)
        return self._to_results(fused, top_k)

    @staticmethod
    def load(store_dir: str | Path, model_name: str | None = None) -> "SearchEngine":
        import json

        store_dir = Path(store_dir)
        inverted_index = InvertedIndex.load(store_dir / "inverted_index.pkl")
        vector_index = VectorIndex.load(store_dir / "vector_index")
        with open(store_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        embedder = Embedder(model_name or meta.get("embedding_model") or DEFAULT_MODEL_NAME)
        with open(store_dir / "doc_texts.json", "r", encoding="utf-8") as f:
            doc_texts = json.load(f)
        return SearchEngine(inverted_index, vector_index, embedder, doc_texts)
