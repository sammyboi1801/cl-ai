"""Needle 3 as an `Embedder`.

Owns every Needle-specific quirk so core never sees one:
  - `embed()` takes a single string, so batching is this module's problem
  - the model costs ~0.6s to construct and is therefore loaded lazily, on the
    first text rather than at import
  - anisotropy (see CENTERING below)

DO NOT ENABLE THIS FOR RETRIEVAL. MEASURED, IT MAKES RANKING WORSE.
Measured over 878 installed tools and the 27 queries in
tests/eval_retrieval.py:

    semantic only                       hit@1 0.037
    lexical + prefix                    hit@1 0.704
    lexical + prefix + these vectors    hit@1 0.444

Fusing it in costs 26 points of hit@1 and doubles query latency. `search()`
takes the vector half as an explicit argument and defaults to None, so nothing
turns this on by accident, and `build_embedder()` refuses to hand it back
unless CL_AI_EMBEDDER names it.

WHY, PRECISELY
Not "the embeddings are noise" -- that was my first conclusion and it is
wrong. Given fifty documents whose distractors are unrelated (fonts,
certificates, packets), this model puts `mkdir`, `curl` and `ps` first for the
matching queries. It separates distant things perfectly well.

What it cannot do is separate NEAR things, and a CLI catalog is nothing but
near things. Every tldr page is one imperative sentence about files,
processes or the network, so the distinctions retrieval actually needs are
exactly the ones this model does not make. At corpus scale the effect is
severe hubness: `tty`, `exec` and `sleep` top the ranking for queries they
have nothing to do with, because short generic documents land nearest the
centroid and score well against everything.

That is the expected behaviour of a tool-CALLING model's hidden state. It was
never trained against a contrastive retrieval objective, so proximity in that
space means "similar kind of text", not "answers this query".

HOW I GOT THIS WRONG, SO THE NEXT PERSON DOES NOT
The justification for building this was a four-document probe:

    0.9650  mkdir - create directories        <- correct, and ranked first
    0.9358  cmake - build system generator
    0.9228  curl - transfer data
    0.9155  rm - remove files

That looks like a working embedder and is worth nothing. Four candidates
means 25% by chance, the spread is 0.05, and the example was hand-picked.
Both probes are kept as tests, the small one green and the corpus-scale
finding recorded, so the trap is visible rather than folklore.

The lesson is about the probe, not the model: a semantic ranker must be
measured ALONE and at corpus scale before it is fused with anything. Fused
first, its failure looks like a bad weight and invites weeks of tuning.

CENTERING IS NOT THE PROBLEM
Checked, because it was the obvious suspect. Centering works exactly as
intended -- median cosine between random unrelated pairs goes from 0.967 raw
to -0.002 centered, which is textbook. hit@1 is 0.037 both before and after.
The geometry is fixed; the ranking is not.

WHAT THIS IS STILL FOR
The adapter and its cache are backend-agnostic and correct, and the `Embedder`
port is the seam a real embedding model plugs into. The semantic gap this was
written to close is real -- "make a directory" shares no token with `mkdir`,
so no lexical method can bridge it -- and closing it needs a model trained for
retrieval. This file is then a 30-line sibling, not a rewrite.

CENTERING, AND THE BATCH-OF-ONE TRAP
Centering is a property of a CORPUS. The obvious implementation -- subtract
the mean of whatever batch was passed to `encode()` -- is silently catastrophic
at query time, where the batch is one text: subtracting a single vector's own
mean yields the zero vector, whose cosine against everything is zero. That is
not a degraded ranking, it is no ranking, and nothing raises.

So this adapter never derives a mean from the batch in front of it. It centers
only against a mean fixed once by `fit()`, and uses that same mean for every
subsequent call regardless of batch size. With no `fit()`, it normalises and
does not center.

That default is safe because `VectorReranker` centers the corpus itself and
explicitly does not trust the adapter to have done it -- core programs against
the weakest backend. In the standard wiring the reranker is the corpus owner
and `fit()` is unnecessary; it exists for callers using this adapter directly.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = ["NeedleEmbedder", "NeedleUnavailable", "default_weights"]

log = logging.getLogger(__name__)

#: Probe text for discovering the model's dimension without a real query.
_PROBE = "list files"


class NeedleUnavailable(RuntimeError):
    """Needle could not be loaded. Raised at first use, never at import."""


def default_weights() -> Path | None:
    """Locate Needle weights, or None. Never raises."""
    override = os.environ.get("CL_AI_NEEDLE_WEIGHTS")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for candidate in _candidate_weight_dirs():
        try:
            if not candidate.is_dir():
                continue
            for entry in sorted(candidate.glob("*.cact")):
                if entry.is_file():
                    return entry
        except OSError:
            continue
    return None


def _candidate_weight_dirs() -> Iterable[Path]:
    yield Path.cwd() / "checkpoints"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            yield Path(base) / "cl-ai" / "models"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            yield Path(xdg) / "cl-ai" / "models"
    try:
        yield Path.home() / ".cache" / "cl-ai" / "models"
    except RuntimeError:  # no home directory on this machine
        return


class NeedleEmbedder:
    """`Embedder` over Needle 3. Thread-safe; the model is loaded once."""

    #: Short name recorded in the vector cache key, so vectors from a
    #: different backend can never be read back as these.
    backend = "needle3"

    def __init__(
        self,
        weights: str | os.PathLike[str] | None = None,
        *,
        mean: Sequence[float] | None = None,
    ) -> None:
        self._weights = Path(weights) if weights is not None else default_weights()
        self._mean: list[float] | None = list(mean) if mean is not None else None
        self._agent: object | None = None
        self._dim = 0
        # Construction is ~0.6s and the daemon may serve concurrent requests;
        # without this, two keystrokes racing at startup build two models.
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    @property
    def weights(self) -> Path | None:
        return self._weights

    def available(self) -> bool:
        """Whether a first `encode()` has a chance of working. Never raises."""
        if self._agent is not None:
            return True
        if self._weights is None or not self._weights.is_file():
            return False
        try:
            import needle  # noqa: F401
        except Exception:  # noqa: BLE001 - a missing or broken backend is a no
            return False
        return True

    def _load(self) -> object:
        with self._lock:
            if self._agent is not None:
                return self._agent
            if self._weights is None:
                raise NeedleUnavailable(
                    "no Needle weights found; set CL_AI_NEEDLE_WEIGHTS"
                )
            if not self._weights.is_file():
                raise NeedleUnavailable(f"weights not found: {self._weights}")
            try:
                import needle
            except Exception as exc:
                raise NeedleUnavailable(f"needle is not importable: {exc}") from exc
            try:
                self._agent = needle.Needle(weights=str(self._weights))
            except Exception as exc:
                raise NeedleUnavailable(f"could not load Needle: {exc}") from exc
            return self._agent

    def close(self) -> None:
        """Release the model. Safe to call repeatedly and after a failure."""
        with self._lock:
            agent, self._agent = self._agent, None
        if agent is None:
            return
        closer = getattr(agent, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                # Teardown must not raise: close() is called from shutdown
                # paths where there is nothing left to handle an exception.
                log.debug("needle close() failed", exc_info=True)

    # -- the port ---------------------------------------------------------

    def dim(self) -> int:
        """Vector width, discovered by embedding one probe string."""
        if self._dim:
            return self._dim
        self._dim = len(self._raw(_PROBE))
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text, center against the fitted mean, L2-normalise.

        Raises `NeedleUnavailable` if the model cannot be loaded. It does not
        return a partial batch: a caller cannot tell which row is missing, and
        `VectorReranker.build` already treats a raising backend as "no vector
        half", which is the honest outcome.
        """
        if not texts:
            return []
        return [self._finish(self._raw(text)) for text in texts]

    def fit(self, texts: Sequence[str]) -> NeedleEmbedder:
        """Fix the centering mean from a corpus. See CENTERING above.

        Refuses a single text, which can only ever produce the zero vector.
        """
        if len(texts) < 2:
            raise ValueError(
                "a centering mean needs at least two texts; "
                "one text centers to the zero vector"
            )
        raws = [self._raw(t) for t in texts]
        width = len(raws[0])
        total = [0.0] * width
        for raw in raws:
            if len(raw) != width:
                raise NeedleUnavailable("model returned inconsistent dimensions")
            for i, value in enumerate(raw):
                total[i] += value
        count = float(len(raws))
        self._mean = [t / count for t in total]
        self._dim = width
        return self

    @property
    def mean(self) -> list[float] | None:
        return None if self._mean is None else list(self._mean)

    # -- internals --------------------------------------------------------

    def _raw(self, text: str) -> list[float]:
        agent = self._load()
        embed = getattr(agent, "embed", None)
        if not callable(embed):
            raise NeedleUnavailable("this Needle build has no embed()")
        try:
            vector = embed(text)
        except Exception as exc:
            raise NeedleUnavailable(f"embed failed: {exc}") from exc
        if not vector:
            raise NeedleUnavailable("model returned an empty vector")
        return [float(v) for v in vector]

    def _finish(self, raw: list[float]) -> list[float]:
        mean = self._mean
        if mean is not None and len(mean) == len(raw):
            raw = [raw[i] - mean[i] for i in range(len(raw))]
        norm = math.sqrt(sum(v * v for v in raw))
        if norm == 0.0:
            # Exactly the corpus mean. Zero is the honest score against
            # everything; dividing here would raise inside a keystroke.
            return [0.0] * len(raw)
        return [v / norm for v in raw]
