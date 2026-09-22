"""The swappable seams.

Two ports, deliberately separate: you may want to keep Needle's embeddings
while moving the planner to a hosted model, or the reverse.

Governing rule: the core programs against the weakest backend. If core is
allowed to assume Needle's grammar guarantee, every swap becomes a rewrite --
so validation lives in core, and a backend that happens to guarantee validity
just makes it a no-op.

Adapters own everything provider-specific: schema dialect translation (Needle
takes flat {name, description, parameters}; OpenAI nests under
{"type":"function","function":{...}}), refusal semantics, statefulness,
agent caching, and embedding normalisation. None of that reaches core.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .ir import Capabilities, Plan, PlanRequest


@runtime_checkable
class Planner(Protocol):
    """Natural language + candidate tools -> a ranked Plan."""

    def capabilities(self) -> Capabilities:
        ...

    def plan(self, request: PlanRequest) -> Plan:
        """Fill arguments for the given tools.

        Must respect `request.deadline_ms` and return a refusal rather than a
        guess when nothing fits. Must never raise for ordinary model failure --
        return `Plan(refused=True)` so the UI can degrade to plain Tab.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Text -> vector, for the retrieval layer."""

    def dim(self) -> int:
        ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return vectors that are ALREADY centered and L2-normalised.

        Centering is the adapter's job because the correction is
        provider-specific. Needle 3's raw embeddings are anisotropic --
        measured, every pair sits near 0.93 cosine and ranking is meaningless
        until the corpus mean is removed.
        """
        ...
