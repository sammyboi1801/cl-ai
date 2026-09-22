"""Tests for reciprocal-rank fusion.

The property that justifies RRF over score averaging is that it reads ONLY the
order, so two rankers on wildly different numeric scales cannot be mis-weighted
against each other. Several tests below assert exactly that.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cl_ai.retrieval.fuse import RRF_K, fuse_ranked_ids, reciprocal_rank_fusion


def test_single_ranking_is_preserved() -> None:
    """With one list, RRF is a monotone transform of that list's order.

    This is what makes the embedder optional at no cost: the no-vector path
    needs no special case, because fusing one list changes nothing.
    """
    assert reciprocal_rank_fusion([[3, 1, 2]]) == [3, 1, 2]


def test_agreement_reinforces() -> None:
    assert reciprocal_rank_fusion([[1, 2, 3], [1, 2, 3]]) == [1, 2, 3]


def test_a_document_ranked_top_by_one_ranker_rises() -> None:
    """The case RRF exists for: either ranker can rescue a good document.

    Document 90 is buried at rank 9 by the first ranker but ranked first by the
    second; document 20 sits mid-list in the first and is absent from the
    second. Fusion must prefer 90 -- that is what lets the lexical half rescue
    a semantic miss and vice versa.
    """
    lexical = [10, 11, 12, 13, 20, 14, 15, 16, 17, 90]
    semantic = [90]
    fused = reciprocal_rank_fusion([lexical, semantic])
    assert fused.index(90) < fused.index(20)


def test_only_order_matters_not_scores() -> None:
    """Two rankers on different scales fuse identically given the same order.

    BM25 is unbounded; cosine sits near 1.0. Averaging would let whichever has
    the larger range win every time.
    """
    a = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
    b = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
    assert a == b


def test_disjoint_rankings_interleave() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]])
    assert set(fused) == {1, 2, 3, 4}
    assert fused.index(1) < fused.index(2)
    assert fused.index(3) < fused.index(4)


def test_weights_scale_influence() -> None:
    unweighted = reciprocal_rank_fusion([[1], [2]])
    weighted = reciprocal_rank_fusion([[1], [2]], weights=[0.1, 10.0])
    assert unweighted[0] == 1          # tie broken by id
    assert weighted[0] == 2            # the heavy ranker wins


def test_zero_weight_excludes_a_ranker() -> None:
    fused = reciprocal_rank_fusion([[1, 2], [3, 4]], weights=[1.0, 0.0])
    assert fused == [1, 2]


def test_weight_count_must_match() -> None:
    with pytest.raises(ValueError, match="weights must match"):
        fuse_ranked_ids([[1], [2]], weights=[1.0])


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_empty_ranking_alongside_a_real_one() -> None:
    """A ranker that declined contributes nothing and breaks nothing."""
    assert reciprocal_rank_fusion([[1, 2, 3], []]) == [1, 2, 3]


def test_scores_follow_the_formula() -> None:
    fused = dict(fuse_ranked_ids([[5, 6]]))
    assert fused[5] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[6] == pytest.approx(1.0 / (RRF_K + 2))


def test_larger_k_flattens_the_top() -> None:
    """k damps how much first place dominates."""
    sharp = dict(fuse_ranked_ids([[1, 2]], k=1))
    flat = dict(fuse_ranked_ids([[1, 2]], k=1000))
    assert sharp[1] / sharp[2] > flat[1] / flat[2]


def test_duplicate_ids_within_one_ranking_accumulate() -> None:
    """Not expected input, but it must not crash or lose the document."""
    fused = dict(fuse_ranked_ids([[1, 1]]))
    assert fused[1] > 1.0 / (RRF_K + 1)


def test_ties_break_by_id_deterministically() -> None:
    """RRF scores come from a small set of discrete ranks, so ties are common."""
    assert reciprocal_rank_fusion([[3, 1, 2], [3, 1, 2]]) == [3, 1, 2]
    assert reciprocal_rank_fusion([[7], [9]]) == [7, 9]


def test_accepts_generators() -> None:
    assert reciprocal_rank_fusion(iter([iter([1, 2])])) == [1, 2]


@given(
    rankings=st.lists(
        st.lists(st.integers(min_value=0, max_value=20), max_size=10, unique=True),
        min_size=1,
        max_size=4,
    )
)
@settings(max_examples=400, deadline=None)
def test_fusion_is_a_permutation_of_the_union(rankings: list[list[int]]) -> None:
    """Never invents or loses a document, and never repeats one."""
    fused = reciprocal_rank_fusion(rankings)
    union: set[int] = set()
    for ranking in rankings:
        union |= set(ranking)
    assert set(fused) == union
    assert len(fused) == len(union)


@given(
    rankings=st.lists(
        st.lists(st.integers(min_value=0, max_value=15), max_size=8, unique=True),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=300, deadline=None)
def test_fusion_scores_are_descending(rankings: list[list[int]]) -> None:
    scores = [score for _, score in fuse_ranked_ids(rankings)]
    assert scores == sorted(scores, reverse=True)


@given(
    rankings=st.lists(
        st.lists(st.integers(min_value=0, max_value=15), max_size=8, unique=True),
        min_size=1,
        max_size=3,
    )
)
@settings(max_examples=200, deadline=None)
def test_fusion_is_deterministic(rankings: list[list[int]]) -> None:
    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)
