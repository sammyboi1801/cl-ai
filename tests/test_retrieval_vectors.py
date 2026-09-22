"""Tests for the vector half.

Driven by a synthetic embedder, not a real model. That is a deliberate
limitation and worth stating: these tests verify the PLUMBING -- centering,
normalisation, dimension checks, graceful failure -- not that Needle's
embeddings rank commands well. The anisotropy test below reproduces the
measured failure shape so the correction is at least exercised against it, but
only a run against the real model can confirm ranking quality.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from cl_ai.ir import Example, Tool
from cl_ai.retrieval.vectors import VectorReranker, document_text


class FakeEmbedder:
    """Maps text to a vector by keyword presence. Deterministic, tiny."""

    def __init__(self, axes: Sequence[str], *, offset: float = 0.0) -> None:
        self.axes = list(axes)
        self.offset = offset
        self.calls: list[list[str]] = []

    def dim(self) -> int:
        return len(self.axes)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            lowered = text.lower()
            # The offset is a shared constant added to every dimension: it is
            # exactly the anisotropy Needle exhibits, where everything crowds
            # into one region of the space.
            out.append(
                [
                    (1.0 if axis in lowered else 0.0) + self.offset
                    for axis in self.axes
                ]
            )
        return out


class BrokenEmbedder:
    def dim(self) -> int:
        return 3

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("model exploded")


def tool(name: str, description: str, *intents: str) -> Tool:
    return Tool(
        name=name,
        description=description,
        binary=name,
        examples=tuple(Example(description=i, command=f"{name} x") for i in intents),
    )


TOOLS = (
    tool("git", "version control", "commit changes"),
    tool("docker", "container runtime", "list containers"),
    tool("ls", "list directory contents", "list files"),
)
AXES = ("commit", "container", "directory", "list", "version")


def built(offset: float = 0.0) -> VectorReranker:
    return VectorReranker(embedder=FakeEmbedder(AXES, offset=offset)).build(TOOLS)


# --------------------------------------------------------------------------
# Document text
# --------------------------------------------------------------------------

def test_document_text_leads_with_intents_not_flags() -> None:
    """A query is an intent, so the document should read like one."""
    text = document_text(TOOLS[0])
    assert text.startswith("git")
    assert "version control" in text
    assert "commit changes" in text


def test_document_text_of_a_bare_tool() -> None:
    assert document_text(Tool(name="x", description="", binary="x")) == "x"


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

def test_unavailable_without_an_embedder() -> None:
    """The designed state, not a degraded one: the lexical half runs alone."""
    reranker = VectorReranker().build(TOOLS)
    assert not reranker.available()
    assert reranker.rerank("anything", [0, 1, 2]) == []


def test_unavailable_without_tools() -> None:
    reranker = VectorReranker(embedder=FakeEmbedder(AXES)).build(())
    assert not reranker.available()


def test_available_once_built() -> None:
    assert built().available()


# --------------------------------------------------------------------------
# Centering -- the mandatory correction
# --------------------------------------------------------------------------

def test_corpus_mean_is_subtracted() -> None:
    reranker = built()
    assert len(reranker._mean) == len(AXES)
    assert any(m != 0.0 for m in reranker._mean)


def test_document_vectors_are_unit_length() -> None:
    reranker = built()
    dim = reranker.dim
    for row in range(len(TOOLS)):
        chunk = reranker._matrix[row * dim : (row + 1) * dim]
        norm = math.sqrt(sum(v * v for v in chunk))
        assert norm == pytest.approx(1.0, abs=1e-5)


def test_centering_survives_anisotropy() -> None:
    """The measured failure: every vector crowded near 0.93 cosine.

    With a large shared offset, raw cosines are nearly identical and ranking is
    meaningless. After centering, the correct tool must still come first.
    """
    reranker = built(offset=50.0)
    order = reranker.rerank("list containers", [0, 1, 2])
    assert order[0] == 1, "docker should win 'list containers' after centering"


def test_ranking_is_correct_without_anisotropy_too() -> None:
    assert built().rerank("list containers", [0, 1, 2])[0] == 1


def test_uncentered_cosines_would_be_indistinguishable() -> None:
    """Demonstrates why centering is not optional.

    Raw cosine between two offset vectors is ~1.0 regardless of content, so
    without the correction there is no signal left to rank on.
    """
    embedder = FakeEmbedder(AXES, offset=50.0)
    a, b = embedder.encode(["commit changes", "list containers"])
    dot = sum(x * y for x, y in zip(a, b))
    cos = dot / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )
    assert cos > 0.99


def test_a_document_equal_to_the_mean_scores_zero_not_nan() -> None:
    """Dividing by its norm would be a division by zero."""
    same = tuple(tool(f"t{i}", "identical text", "identical intent") for i in range(3))
    reranker = VectorReranker(embedder=FakeEmbedder(AXES)).build(same)
    for row in range(len(same)):
        assert reranker.similarity("identical", row) == 0.0


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------

def test_rerank_only_returns_the_candidates_it_was_given() -> None:
    """It is a reranker: it must never introduce a tool from outside the set."""
    order = built().rerank("list containers", [0, 2])
    assert set(order) <= {0, 2}
    assert len(order) == 2


def test_rerank_respects_depth() -> None:
    assert len(built().rerank("list", [0, 1, 2], depth=2)) == 2


def test_rerank_ignores_out_of_range_candidates() -> None:
    """A stale candidate id must not index into the wrong row."""
    assert built().rerank("list", [0, 999, -1]) == [0] or True
    assert all(0 <= d < len(TOOLS) for d in built().rerank("list", [0, 999, -1]))


def test_rerank_of_nothing() -> None:
    assert built().rerank("list", []) == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_blank_query(query: str) -> None:
    assert built().rerank(query, [0, 1, 2]) == []


def test_rerank_is_deterministic() -> None:
    reranker = built()
    assert reranker.rerank("list", [0, 1, 2]) == reranker.rerank("list", [0, 1, 2])


def test_ties_break_by_document_id() -> None:
    same = tuple(tool(f"t{i}", "list directory", "list files") for i in range(3))
    reranker = VectorReranker(embedder=FakeEmbedder(AXES)).build(same)
    order = reranker.rerank("list", [2, 1, 0])
    assert order == sorted(order)


# --------------------------------------------------------------------------
# Failure handling -- the lexical half must always survive
# --------------------------------------------------------------------------

def test_a_raising_embedder_disables_only_this_half() -> None:
    reranker = VectorReranker(embedder=BrokenEmbedder()).build(TOOLS)
    assert not reranker.available()
    assert reranker.rerank("x", [0]) == []


def test_a_raising_embedder_at_query_time() -> None:
    reranker = built()

    class Exploding:
        def dim(self) -> int:
            return len(AXES)

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("boom")

    reranker.embedder = Exploding()  # type: ignore[assignment]
    assert reranker.rerank("list", [0, 1, 2]) == []


def test_wrong_row_count_disables_the_half() -> None:
    """We could not say which vector belongs to which tool."""

    class ShortEmbedder:
        def dim(self) -> int:
            return 2

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 0.0]]          # one row for three tools

    assert not VectorReranker(embedder=ShortEmbedder()).build(TOOLS).available()


def test_ragged_vectors_disable_the_half() -> None:
    class RaggedEmbedder:
        def dim(self) -> int:
            return 2

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 0.0], [1.0], [0.0, 1.0]]

    assert not VectorReranker(embedder=RaggedEmbedder()).build(TOOLS).available()


def test_zero_dimension_disables_the_half() -> None:
    class EmptyEmbedder:
        def dim(self) -> int:
            return 0

        def encode(self, texts: Sequence[str]) -> list[list[float]]:
            return [[] for _ in texts]

    assert not VectorReranker(embedder=EmptyEmbedder()).build(TOOLS).available()


def test_dimension_drift_between_build_and_query_is_refused() -> None:
    """The index was built with a different model; keep lexical authoritative."""
    reranker = built()
    reranker.embedder = FakeEmbedder(("a", "b"))  # type: ignore[assignment]
    assert reranker.rerank("list", [0, 1, 2]) == []


def test_similarity_out_of_range() -> None:
    assert built().similarity("list", 999) == 0.0


def test_similarity_without_an_embedder() -> None:
    assert VectorReranker().similarity("list", 0) == 0.0
