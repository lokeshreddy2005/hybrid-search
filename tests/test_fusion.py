import pytest

from retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed():
    """
    bm25 ranks:   [A, B, C]  (rank 1, 2, 3)
    vector ranks: [B, A, D]  (rank 1, 2, 3)
    k = 60

    RRF(A) = 1/(60+1) + 1/(60+2) = 0.0163934426 + 0.0161290323 = 0.0325224749
    RRF(B) = 1/(60+2) + 1/(60+1) = same as A by symmetry = 0.0325224749
    RRF(C) = 1/(60+3)                                      = 0.0158730159
    RRF(D) =                       1/(60+3)                = 0.0158730159

    A and B tie; both outrank C and D, which also tie.
    """
    bm25_ranked = [("A", 5.0), ("B", 4.0), ("C", 3.0)]
    vector_ranked = [("B", 0.9), ("A", 0.8), ("D", 0.7)]

    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
    fused_scores = dict(fused)

    expected_ab = 1 / 61 + 1 / 62
    expected_cd = 1 / 63

    assert fused_scores["A"] == pytest.approx(expected_ab, rel=1e-9)
    assert fused_scores["B"] == pytest.approx(expected_ab, rel=1e-9)
    assert fused_scores["C"] == pytest.approx(expected_cd, rel=1e-9)
    assert fused_scores["D"] == pytest.approx(expected_cd, rel=1e-9)

    top_two = {doc_id for doc_id, _ in fused[:2]}
    assert top_two == {"A", "B"}


def test_rrf_ignores_raw_scores_only_uses_rank():
    """A huge raw score shouldn't matter if the rank is the same."""
    list_a = [("X", 1000.0), ("Y", 1.0)]
    list_b = [("X", 0.001), ("Y", 0.0009)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0][0] == "X"


def test_rrf_doc_only_in_one_list():
    bm25_ranked = [("A", 1.0)]
    vector_ranked = [("B", 1.0)]
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60)
    fused_scores = dict(fused)
    assert fused_scores["A"] == pytest.approx(1 / 61)
    assert fused_scores["B"] == pytest.approx(1 / 61)


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_rrf_weights():
    bm25_ranked = [("A", 1.0), ("B", 1.0)]
    vector_ranked = [("B", 1.0), ("A", 1.0)]
    fused = reciprocal_rank_fusion([bm25_ranked, vector_ranked], k=60, weights=[2.0, 1.0])
    fused_scores = dict(fused)
    # A: rank1 in bm25 (weight 2), rank2 in vector (weight 1)
    expected_a = 2.0 * (1 / 61) + 1.0 * (1 / 62)
    expected_b = 2.0 * (1 / 62) + 1.0 * (1 / 61)
    assert fused_scores["A"] == pytest.approx(expected_a)
    assert fused_scores["B"] == pytest.approx(expected_b)
    assert fused_scores["A"] > fused_scores["B"]
