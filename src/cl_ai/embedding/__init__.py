"""Embedding backends and their cache.

Separate from `retrieval/` on purpose: `retrieval` is the algorithm and must
keep working with no backend at all, while this package is where a specific
model's quirks are allowed to live. The seam between them is the `Embedder`
port in `cl_ai.ports`.

`build_embedder()` returns None unless CL_AI_EMBEDDER explicitly names a
backend. That is not caution for its own sake: the only backend implemented
today is Needle, and Needle's embeddings are MEASURED to make retrieval worse
(hit@1 0.704 -> 0.444; see needle_embedder.py). An opt-in default is what
stops a future caller from wiring in a component whose own docstring says not
to use it.
"""

from __future__ import annotations

import os

from .cache import CachingEmbedder, VectorCache, default_vector_path, model_key
from .needle_embedder import NeedleEmbedder, NeedleUnavailable, default_weights

__all__ = [
    "CachingEmbedder",
    "NeedleEmbedder",
    "NeedleUnavailable",
    "VectorCache",
    "build_embedder",
    "default_vector_path",
    "default_weights",
    "model_key",
]


def build_embedder(
    weights: str | os.PathLike[str] | None = None,
    *,
    cache: bool = True,
    path: str | os.PathLike[str] | None = None,
    backend: str | None = None,
) -> CachingEmbedder | NeedleEmbedder | None:
    """The configured embedder, or None. Never raises.

    None rather than an exception because having no embedder is the ordinary
    case -- retrieval is a hybrid whose lexical half stands alone -- and a
    caller that must treat it as fatal can check for None itself.

    `backend` defaults to $CL_AI_EMBEDDER, and an unset or unrecognised value
    yields None. See the module docstring for why this is opt-in.
    """
    chosen = (backend if backend is not None else os.environ.get("CL_AI_EMBEDDER", "")).strip()
    if chosen.lower() not in {"needle", "needle3"}:
        return None
    embedder = NeedleEmbedder(weights)
    if not embedder.available():
        return None
    if not cache:
        return embedder
    key = model_key(embedder.backend, embedder.weights)
    return CachingEmbedder(embedder, key=key, path=path)
