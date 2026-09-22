"""Vector half.

MANDATORY: mean-center every query and document. Raw Needle embeddings are
anisotropic -- measured, everything lands at ~0.93 cosine and `docker run`
outranked `git reset` for a git query. Centering fixes the ranking completely.
Skip it and retrieval returns noise at high confidence.

A RERANKER, NOT A SEARCHER
This scores a bounded candidate set handed to it by the lexical half; it never
scans the corpus. That started as a performance constraint and turned out to be
the better design.

The constraint: the package has no third-party dependencies, and a full scan
means 5,000 tools x 3,072 dimensions -- about 15 million multiply-adds per
query, seconds in CPython, behind a key the user expects to feel instant.
Pulling in numpy to fix that would add a dependency to the daemon's startup
path for one component. Reranking the top ~120 candidates is roughly 400,000
operations, a few milliseconds, and needs nothing but the standard library.

The design win: retrieve-then-rerank is how this is normally done anyway.
Lexical retrieval has high recall and mediocre ordering; embeddings have the
opposite. Letting each do the half it is good at beats asking either to do
both, and it means a semantic near-miss can never bury a tool the user's own
words clearly matched.

WHO CENTERS
The `Embedder` port says the adapter returns vectors already centered and
L2-normalised, because the correction is provider-specific. This module does
NOT trust that. Core programs against the weakest backend, so it centers and
normalises itself. Doing it twice is harmless -- centering an already-centered
corpus subtracts approximately zero -- while not doing it at all produces a
confidently wrong ranking, which is the worst failure in the system.

Centering is a property of a CORPUS, not of a vector, so it happens once at
build time and the same corpus mean is subtracted from every query. Centering
a query against itself would be a no-op that quietly does nothing.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cl_ai.ir import Tool

__all__ = ["VectorReranker", "document_text"]

#: How many lexical candidates to rerank. Large enough that a tool the lexical
#: half ranked poorly can still be promoted, small enough to stay in the
#: millisecond budget.
DEFAULT_RERANK_DEPTH = 120


class _Embedder(Protocol):
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def document_text(tool: Tool) -> str:
    """The text embedded for one tool.

    Intent phrasings lead, because a user's query is an intent. Embedding the
    flag soup of a command line instead puts the document in a different region
    of the space from every query that could match it.
    """
    parts = [tool.invocation]
    if tool.description:
        parts.append(tool.description)
    parts.extend(e.description for e in tool.examples if e.description)
    return ". ".join(parts)


def _dot(a: array[float], offset: int, b: Sequence[float], dim: int) -> float:
    total = 0.0
    for i in range(dim):
        total += a[offset + i] * b[i]
    return total


@dataclass
class VectorReranker:
    """Cosine reranking over centered embeddings. Optional half of the hybrid."""

    embedder: _Embedder | None = None
    dim: int = 0
    #: Flat row-major matrix of unit-length centered document vectors.
    _matrix: array[float] = field(default_factory=lambda: array("f"))
    _mean: list[float] = field(default_factory=list)
    _rows: int = 0

    def available(self) -> bool:
        return self.embedder is not None and self._rows > 0 and self.dim > 0

    # -- building ---------------------------------------------------------

    def build(self, tools: Sequence[Tool]) -> VectorReranker:
        """Encode and center the corpus. Safe to call with no embedder."""
        if self.embedder is None or not tools:
            return self
        try:
            vectors = self.embedder.encode([document_text(t) for t in tools])
        except Exception:  # noqa: BLE001 - a failing backend disables this half
            # A model that raises costs us the vector half, not the whole of
            # retrieval. The lexical half is the one that must always work.
            return self
        if not vectors or len(vectors) != len(tools):
            # Wrong row count means we cannot say which vector belongs to which
            # tool. Disabling is correct; guessing would rank tools against
            # other tools' vectors.
            return self

        dim = len(vectors[0])
        if dim == 0 or any(len(v) != dim for v in vectors):
            return self

        mean = [0.0] * dim
        for vector in vectors:
            for i, value in enumerate(vector):
                mean[i] += value
        count = float(len(vectors))
        mean = [m / count for m in mean]

        matrix: array[float] = array("f")
        for vector in vectors:
            centered = [vector[i] - mean[i] for i in range(dim)]
            norm = math.sqrt(sum(c * c for c in centered))
            if norm == 0.0:
                # Equal to the corpus mean exactly. Left as zero, it scores 0.0
                # against every query, which is the honest answer; dividing
                # would be a division by zero.
                matrix.extend([0.0] * dim)
            else:
                matrix.extend([c / norm for c in centered])

        self.dim = dim
        self._mean = mean
        self._matrix = matrix
        self._rows = len(vectors)
        return self

    # -- querying ---------------------------------------------------------

    def _encode_query(self, query: str) -> list[float] | None:
        if self.embedder is None or not query.strip():
            return None
        try:
            encoded = self.embedder.encode([query])
        except Exception:  # noqa: BLE001 - see build()
            return None
        if not encoded or len(encoded[0]) != self.dim:
            # Dimension drift means the index was built with a different model.
            # Returning nothing keeps the lexical half authoritative rather
            # than fusing scores computed in the wrong space.
            return None
        centered = [encoded[0][i] - self._mean[i] for i in range(self.dim)]
        norm = math.sqrt(sum(c * c for c in centered))
        if norm == 0.0:
            return None
        return [c / norm for c in centered]

    def rerank(
        self, query: str, candidates: Sequence[int], *, depth: int | None = None
    ) -> list[int]:
        """Reorder `candidates` by cosine similarity, best first.

        Returns [] when unavailable, so the caller can treat "no vector half"
        and "vector half declined" identically.
        """
        if not self.available() or not candidates:
            return []
        vector = self._encode_query(query)
        if vector is None:
            return []

        limit = DEFAULT_RERANK_DEPTH if depth is None else depth
        considered = [c for c in candidates[:limit] if 0 <= c < self._rows]
        scored = [
            (_dot(self._matrix, doc * self.dim, vector, self.dim), doc)
            for doc in considered
        ]
        # Descending score, then ascending id so ties are deterministic.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, doc in scored]

    def similarity(self, query: str, doc: int) -> float:
        """One raw cosine, for diagnosing a ranking rather than for fusion.

        Fusion uses only the ORDER from rerank(): these numbers sit in a narrow
        band and are not comparable with BM25 scores.
        """
        if not self.available() or not 0 <= doc < self._rows:
            return 0.0
        vector = self._encode_query(query)
        if vector is None:
            return 0.0
        return _dot(self._matrix, doc * self.dim, vector, self.dim)
