"""Reciprocal Rank Fusion (RRF) for combining ranked lists from BM25 and
dense vector retrieval.

Why RRF and not a weighted sum of raw scores:
BM25 scores are unbounded and corpus/query-dependent (a query with rare
terms can produce scores of 20+, a query with only common terms may top
out near 2). Cosine similarity from the embedding model is bounded in
[-1, 1] and tends to cluster in a much narrower band (~0.3-0.8) for this
kind of short-text corpus. Averaging or summing these two directly
requires per-query min-max (or z-score) normalization, and the result is
sensitive to outliers and to how that normalization is done — it's easy
to end up with one signal silently dominating.

RRF sidesteps the scale problem entirely by using each system's *rank*
rather than its raw score:

    RRF(d) = sum_{s in systems} 1 / (k + rank_s(d))

A document that doesn't appear in a system's top-N contributes 0 from
that system. k (default 60, the value used in the original RRF paper
and widely reused, e.g. Elasticsearch's hybrid search) dampens the
influence of very-top-ranked documents so the fusion isn't dominated by
whichever single system happens to rank one document 1st.
"""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of (doc_id, score) into one ranked list.

    Only rank position within each list is used, not the scores. Lists
    may have different lengths and need not overlap in doc_ids.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    assert len(weights) == len(ranked_lists)

    fused: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            fused[doc_id] += weight * (1.0 / (k + rank))

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
