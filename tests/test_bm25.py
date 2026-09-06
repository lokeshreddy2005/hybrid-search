"""BM25 correctness verified against a hand-computed example.

Corpus (after tokenization, all doc lengths = 3, so avgdl = 3):
    d0: "cat sat mat"  -> [cat, sat, mat]
    d1: "dog sat log"  -> [dog, sat, log]
    d2: "cat cat cat"  -> [cat, cat, cat]   (tf(cat, d2) = 3)

Query: "cat", k1=1.5, b=0.75 (defaults).

df(cat) = 2 (d0, d2), N = 3
idf(cat) = ln((N - df + 0.5)/(df + 0.5) + 1) = ln((3-2+0.5)/(2+0.5) + 1) = ln(1.6) ~= 0.4700036292

d0: tf=1, |D|=3
    denom = tf + k1*(1 - b + b*|D|/avgdl) = 1 + 1.5*(0.25 + 0.75*1) = 1 + 1.5 = 2.5
    numer = tf*(k1+1) = 1*2.5 = 2.5
    score = idf * (2.5/2.5) = idf ~= 0.4700036292

d2: tf=3, |D|=3
    denom = 3 + 1.5*1.0 = 4.5
    numer = 3*2.5 = 7.5
    score = idf * (7.5/4.5) = idf * 1.66666... ~= 0.7833393821

d1 does not contain "cat" -> score 0, not returned by search().
"""
import math

import pytest

from index.bm25 import BM25
from index.inverted_index import InvertedIndex

DOCS = [
    ("d0", "", "cat sat mat"),
    ("d1", "", "dog sat log"),
    ("d2", "", "cat cat cat"),
]

EXPECTED_IDF_CAT = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
EXPECTED_SCORE_D0 = EXPECTED_IDF_CAT * (2.5 / 2.5)
EXPECTED_SCORE_D2 = EXPECTED_IDF_CAT * (7.5 / 4.5)


def build_bm25() -> BM25:
    idx = InvertedIndex()
    idx.build(DOCS)
    return BM25(idx, k1=1.5, b=0.75)


def test_idf_matches_hand_computation():
    bm25 = build_bm25()
    assert bm25.idf("cat") == pytest.approx(EXPECTED_IDF_CAT, rel=1e-9)


def test_score_matches_hand_computation():
    bm25 = build_bm25()
    query_terms = ["cat"]
    score_d0 = bm25.score(query_terms, internal_doc_id=0)
    score_d2 = bm25.score(query_terms, internal_doc_id=2)
    assert score_d0 == pytest.approx(EXPECTED_SCORE_D0, rel=1e-9)
    assert score_d2 == pytest.approx(EXPECTED_SCORE_D2, rel=1e-9)


def test_doc_without_term_scores_zero():
    bm25 = build_bm25()
    assert bm25.score(["cat"], internal_doc_id=1) == 0.0


def test_search_ranks_higher_term_frequency_first():
    bm25 = build_bm25()
    results = bm25.search("cat", top_k=10)
    # d2 (tf=3) should outrank d0 (tf=1); d1 (no "cat") should not appear
    assert [doc_id for doc_id, _ in results] == [2, 0]
    scores = dict(results)
    assert scores[0] == pytest.approx(EXPECTED_SCORE_D0, rel=1e-9)
    assert scores[2] == pytest.approx(EXPECTED_SCORE_D2, rel=1e-9)


def test_search_and_score_agree():
    """search() (postings-driven) and score() (term-lookup) must agree exactly."""
    bm25 = build_bm25()
    query_terms = ["cat"]
    for doc_id, search_score in bm25.search("cat", top_k=10):
        assert search_score == pytest.approx(bm25.score(query_terms, doc_id), rel=1e-12)


def test_query_with_no_matches_returns_empty():
    bm25 = build_bm25()
    assert bm25.search("zzz_nonexistent_term", top_k=10) == []
