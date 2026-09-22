"""Reciprocal-rank fusion of the lexical and vector rankings.

RRF rather than score averaging, and the reason is concrete: BM25 scores are
unbounded and corpus-dependent, cosine similarities sit in a narrow band near
1.0 for Needle embeddings even after centering. Averaging two such scales means
whichever ranker happens to have the larger numeric range wins every time, and
tuning a weight to fix that re-breaks whenever the corpus changes.

RRF reads only the ORDER each ranker produced, so the two cannot be mis-scaled
relative to each other. A document ranked first by either ranker scores
1/(k+1); the constant k damps how much the top of a list dominates, and 60 is
the value from the original Cormack et al. work.

The second property that matters here: fusing is defined for one input list.
Embeddings are an optional dependency, so the lexical ranker must work alone --
and with a single list, RRF is a monotone transform of that list's order, i.e.
it changes nothing. The no-embedder path costs nothing and needs no special
case.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = ["RRF_K", "fuse_ranked_ids", "reciprocal_rank_fusion"]

#: From Cormack, Clarke & Buettcher (2009). Not tuned here: tuning it against
#: this corpus without held-out queries would be fitting noise.
RRF_K = 60


def fuse_ranked_ids(
    rankings: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several ranked id lists into one, best first.

    `weights` scales each ranker's contribution. It exists for the case where
    one ranker is known to be weaker -- not as a tuning knob to be fitted, and
    the default treats every ranker equally.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[int, float] = {}
    for position, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[position]
        if weight == 0.0:
            continue
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)

    fused = list(scores.items())
    # Descending score, then ascending id, so the result is deterministic for
    # documents that tie -- which happens constantly, because RRF scores come
    # from a small set of discrete rank positions.
    fused.sort(key=lambda item: (-item[1], item[0]))
    return fused


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[int]],
    *,
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[int]:
    """Convenience wrapper returning just the fused order."""
    materialised = [list(r) for r in rankings]
    return [doc for doc, _ in fuse_ranked_ids(materialised, weights=weights, k=k)]
