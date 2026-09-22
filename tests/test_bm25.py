"""Tests for the BM25 index.

The two tests that matter most here encode bugs that this corpus guarantees:
IDF must never go negative, and length normalisation must not be allowed to
rank the best-documented tool last.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cl_ai.retrieval.bm25 import Bm25Index, Bm25Params, Field

FIELDS = (Field(name="name", weight=2.0, b=0.4), Field(name="body", weight=1.0, b=0.5))


def build(*docs: tuple[str, str]) -> Bm25Index:
    return Bm25Index.build(
        [{"name": n, "body": b} for n, b in docs], FIELDS
    )


# --------------------------------------------------------------------------
# IDF
# --------------------------------------------------------------------------

def test_idf_is_never_negative_for_a_common_term() -> None:
    """The textbook formula goes negative above 50% document frequency.

    In this corpus that is guaranteed -- "file" and "directory" appear in a
    large fraction of command descriptions. A negative weight means matching a
    common word LOWERS a tool's score, so a tool whose description says "file"
    ranks below one that never mentions files, for the query "file".
    """
    # "file" in every document: df == N, the worst case for the naive formula.
    index = build(*[(f"tool{i}", "file") for i in range(10)])
    assert index.idf("file") > 0.0


@given(
    total=st.integers(min_value=1, max_value=5000),
    df=st.integers(min_value=1, max_value=5000),
)
@settings(max_examples=300, deadline=None)
def test_idf_formula_is_positive_everywhere(total: int, df: int) -> None:
    df = min(df, total)
    value = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
    assert value > 0.0


def test_rarer_terms_weigh_more() -> None:
    index = build(
        ("a", "common rare"),
        ("b", "common"),
        ("c", "common"),
        ("d", "common"),
    )
    assert index.idf("rare") > index.idf("common")


def test_unknown_term_has_zero_idf() -> None:
    """Zero, not the maximum: an unknown term is no evidence at all.

    Treating it as maximally informative would let a typo dominate a ranking.
    """
    assert build(("a", "x")).idf("nonexistent") == 0.0


def test_document_frequency_counts_across_fields() -> None:
    """Rarity is a property of the corpus, not of which field a word is in.

    Per-field IDF would make a word common in descriptions look rare in names,
    and rank a coincidental name match above a real description match.
    """
    both = build(("shared", "shared"), ("other", "other"))
    assert both.idf("shared") == pytest.approx(both.idf("other"))


# --------------------------------------------------------------------------
# Length normalisation
# --------------------------------------------------------------------------

def test_low_b_does_not_punish_a_well_documented_tool() -> None:
    """Measured regression: `git commit` ranked 287th for `git com`.

    It lost to `git commit-graph` purely because it had eight examples and
    commit-graph had three, so its fields were longer and normalised down. In
    this corpus more documentation means more useful, so b stays low.
    """
    fields = (Field(name="name", weight=1.0, b=0.0),)
    documents = [
        {"name": "git commit " + "extra " * 40},   # well documented
        {"name": "git commit-graph"},              # sparse
    ]
    index = Bm25Index.build(documents, fields)
    results = index.score(["git", "commit"])
    assert results[0].doc == 0

    # With full normalisation the sparse document wins -- the bug.
    punishing = Bm25Index.build(documents, (Field(name="name", weight=1.0, b=1.0),))
    assert punishing.score(["git", "commit"])[0].doc == 1


def test_b_zero_ignores_length_entirely() -> None:
    fields = (Field(name="name", weight=1.0, b=0.0),)
    index = Bm25Index.build(
        [{"name": "x"}, {"name": "x " + "pad " * 50}], fields
    )
    results = {s.doc: s.score for s in index.score(["x"])}
    assert results[0] == pytest.approx(results[1])


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_matching_more_query_terms_scores_higher() -> None:
    index = build(("alpha beta", ""), ("alpha", ""))
    results = index.score(["alpha", "beta"])
    assert results[0].doc == 0


def test_field_weight_is_applied() -> None:
    """The same term is worth more in a higher-weighted field."""
    fields = (Field(name="name", weight=10.0, b=0.5), Field(name="body", weight=1.0, b=0.5))
    index = Bm25Index.build(
        [{"name": "target", "body": ""}, {"name": "", "body": "target"}], fields
    )
    assert index.score(["target"])[0].doc == 0


def test_matched_terms_are_reported() -> None:
    """The score floor is computed from which terms matched, not the score."""
    index = build(("alpha", "beta"), ("alpha", ""))
    by_doc = {s.doc: s.matched for s in index.score(["alpha", "beta", "gamma"])}
    assert by_doc[0] == {"alpha", "beta"}
    assert by_doc[1] == {"alpha"}


def test_term_frequency_saturates() -> None:
    """k1 caps the benefit of repetition, so keyword stuffing cannot win."""
    fields = (Field(name="name", weight=1.0, b=0.0),)
    index = Bm25Index.build(
        [{"name": "x"}, {"name": "x x x x x x x x x x"}], fields
    )
    scores = {s.doc: s.score for s in index.score(["x"])}
    assert scores[1] > scores[0]
    assert scores[1] < scores[0] * (Bm25Params().k1 + 1.0)


def test_only_matching_documents_are_returned() -> None:
    """Cost scales with posting lists, not corpus size."""
    index = build(("alpha", ""), ("beta", ""), ("gamma", ""))
    assert [s.doc for s in index.score(["alpha"])] == [0]


def test_limit_truncates() -> None:
    index = build(*[(f"x{i}", "shared") for i in range(10)])
    assert len(index.score(["shared"], limit=3)) == 3


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

def test_empty_corpus() -> None:
    index = Bm25Index.build([], FIELDS)
    assert index.doc_count == 0
    assert index.score(["anything"]) == []


def test_empty_query() -> None:
    assert build(("a", "b")).score([]) == []


def test_documents_with_empty_fields() -> None:
    index = Bm25Index.build([{"name": "", "body": ""}], FIELDS)
    assert index.score(["anything"]) == []
    assert index.doc_count == 1


def test_missing_field_in_a_document() -> None:
    """A document need not supply every field."""
    index = Bm25Index.build([{"name": "only"}], FIELDS)
    assert [s.doc for s in index.score(["only"])] == [0]


def test_unknown_terms_only() -> None:
    assert build(("a", "b")).score(["zzz", "qqq"]) == []


def test_vocabulary_and_doc_count() -> None:
    index = build(("alpha", "beta"), ("gamma", ""))
    assert index.doc_count == 2
    assert index.vocabulary == 3


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_results_are_deterministic() -> None:
    """Without a tie-break on doc id, candidates shuffle between runs."""
    index = build(*[("same text", "same body") for _ in range(20)])
    first = [s.doc for s in index.score(["same"])]
    second = [s.doc for s in index.score(["same"])]
    assert first == second == sorted(first)


def test_ties_break_by_document_id() -> None:
    index = build(("x", ""), ("x", ""), ("x", ""))
    assert [s.doc for s in index.score(["x"])] == [0, 1, 2]


def test_scores_are_descending() -> None:
    index = build(("alpha beta", ""), ("alpha", ""), ("alpha", "beta"))
    scores = [s.score for s in index.score(["alpha", "beta"])]
    assert scores == sorted(scores, reverse=True)


@given(
    docs=st.lists(
        st.tuples(
            st.text(alphabet="abc ", max_size=12), st.text(alphabet="abc ", max_size=12)
        ),
        min_size=1,
        max_size=12,
    ),
    query=st.lists(st.sampled_from(["a", "b", "c", "z"]), max_size=4),
)
@settings(max_examples=250, deadline=None)
def test_scoring_never_raises_and_scores_are_finite(
    docs: list[tuple[str, str]], query: list[str]
) -> None:
    index = build(*docs)
    for scored in index.score(query):
        assert math.isfinite(scored.score)
        assert scored.score > 0.0
        assert 0 <= scored.doc < len(docs)
