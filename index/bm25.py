"""BM25 (Okapi) scoring implemented from scratch on top of InvertedIndex.

Formula (Robertson/Sparck-Jones, "+1" smoothed idf as used by Lucene/ES so
idf never goes negative for very common terms):

    score(D, Q) = sum_{t in Q} idf(t) * (f(t,D) * (k1 + 1))
                                          -----------------------------
                                          f(t,D) + k1 * (1 - b + b * |D|/avgdl)

    idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

where:
    f(t,D)  = raw term frequency of t in document D
    |D|     = length of D in tokens
    avgdl   = average document length across the corpus
    N       = total number of documents
    df(t)   = number of documents containing t
    k1, b   = free parameters (defaults 1.5, 0.75 — standard Okapi defaults)
"""
from __future__ import annotations

import math

from index.inverted_index import InvertedIndex
from index.tokenizer import tokenize


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self._idf_cache: dict[str, float] = {}

    def idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        n = self.index.n_docs
        df = self.index.doc_freq(term)
        value = math.log((n - df + 0.5) / (df + 0.5) + 1)
        self._idf_cache[term] = value
        return value

    def score_term(self, term: str, tf: int, doc_len: int) -> float:
        """BM25 contribution of a single query term given its raw tf in a doc."""
        avgdl = self.index.avg_doc_length or 1.0
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        return self.idf(term) * (numerator / denominator)

    def score(self, query_terms: list[str], internal_doc_id: int) -> float:
        """Full BM25 score of one document for a (already tokenized) query.

        O(|query|) via a term -> tf lookup per query term rather than
        scanning the whole postings list; used for tests / spot checks.
        Retrieval-time scoring uses `search`, which is postings-driven.
        """
        doc_len = self.index.doc_lengths[internal_doc_id]
        total = 0.0
        for term in set(query_terms):
            postings = self.index.get_postings(term)
            tf = next((tf for doc_id, tf in postings if doc_id == internal_doc_id), 0)
            if tf == 0:
                continue
            total += self.score_term(term, tf, doc_len)
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Term-at-a-time retrieval: walk only the postings lists of query
        terms and accumulate scores per candidate document, exactly what a
        real inverted-index search engine does (never scans non-matching docs).
        """
        query_terms = tokenize(query)
        scores: dict[int, float] = {}
        for term in set(query_terms):
            postings = self.index.get_postings(term)
            if not postings:
                continue
            idf = self.idf(term)
            avgdl = self.index.avg_doc_length or 1.0
            for internal_doc_id, tf in postings:
                doc_len = self.index.doc_lengths[internal_doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                scores[internal_doc_id] = scores.get(internal_doc_id, 0.0) + idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
