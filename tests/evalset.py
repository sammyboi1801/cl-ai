"""The evaluation case schema, and the scorer that reads it.

WHY A SECOND HARNESS
tests/eval_retrieval.py scores which TOOL comes back. That is half the
question. `git_commit` is the right tool for eleven different intents, and
returning "amend the last commit" when the user asked to sign one scores as a
hit there while being, to the user, simply wrong.

So this scores two things independently:

    tool@k        did the right tool appear in the top k?
    example       given the right tool, was the right example chosen?

Reported separately on purpose. A change that improves one and wrecks the
other is invisible in a combined number, and those are exactly the changes
this project keeps making -- the vector half improved nothing and cost 26
points, and the only reason that was clear was that it was measured alone.

WHAT A CASE ASSERTS
`tools` is a set of ACCEPTABLE answers, not one right answer. Several tools
genuinely satisfy "copy a file", and insisting on one measures the case
author's taste rather than retrieval.

An empty `tools` means the correct behaviour is to return NOTHING. Those
cases are not filler: a nearest-neighbour substitution is how a missing tool
becomes a confidently wrong command, and the score floor that prevents it can
only be defended if something measures it.

`forbid_tools` catches the plausible-but-wrong neighbour -- `grep` on Windows
when `findstr` is meant, `du` when the user asked what `df` answers. A case
can be a tool@1 hit and still be a bad answer, and without this that is
unmeasurable.

`expect` / `forbid` are substrings of the COMMAND, and are what make example
selection measurable at all.

`forbid` means two different things depending on the case, and both are
wanted:

  * with `tools` set, it is scoped to the accepted tool's command -- "the
    right tool, but do not pick THAT example". `git reset --hard` when the
    user asked to commit.
  * with `tools` empty, it applies to EVERYTHING returned. For a hostile or
    nonsense query the assertion is not about which tool won, it is that
    nothing destructive reached the buffer at all. That is the stronger
    claim and the one those cases exist to make.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["Case", "Outcome", "Report", "load_cases", "score"]


@dataclass(frozen=True)
class Case:
    """One evaluation case. See the module docstring for the semantics."""

    query: str
    #: Tool names, any of which is an acceptable answer. Empty means the
    #: correct behaviour is to return nothing at all.
    tools: frozenset[str] = frozenset()
    #: Tools that would be actively wrong here, even if plausible.
    forbid_tools: frozenset[str] = frozenset()
    #: Substrings the chosen command must contain, and must not.
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    #: True when the correct answer is genuinely destructive and must be
    #: marked as such. Not "this must be suppressed" -- the user asked.
    dangerous: bool = False
    #: Restricts the case to one platform. None means it applies everywhere.
    os: str | None = None
    persona: str = "unknown"
    style: str = "natural"
    category: str = "general"
    note: str = ""

    @property
    def expects_nothing(self) -> bool:
        return not self.tools


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(v) for v in value)
    raise TypeError(f"expected a string or iterable, got {type(value).__name__}")


def load_cases(raw: Iterable[Mapping[str, object]]) -> tuple[Case, ...]:
    """Normalise plain dicts into Cases, rejecting malformed ones loudly.

    Written to be strict. These cases are ground truth, and a silently
    mis-parsed one makes the whole measurement a guess -- a typo in a key
    name would otherwise turn an assertion into a default.
    """
    known = {
        "query", "tools", "forbid_tools", "expect", "forbid", "dangerous",
        "os", "persona", "style", "category", "note",
    }
    cases: list[Case] = []
    for index, entry in enumerate(raw):
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"case {index}: unknown keys {sorted(unknown)}")
        query = str(entry.get("query", "")).strip()
        if not query:
            raise ValueError(f"case {index}: empty query")
        target = entry.get("os")
        cases.append(
            Case(
                query=query,
                tools=frozenset(_as_tuple(entry.get("tools"))),
                forbid_tools=frozenset(_as_tuple(entry.get("forbid_tools"))),
                expect=_as_tuple(entry.get("expect")),
                forbid=_as_tuple(entry.get("forbid")),
                dangerous=bool(entry.get("dangerous", False)),
                os=None if target is None else str(target),
                persona=str(entry.get("persona", "unknown")),
                style=str(entry.get("style", "natural")),
                category=str(entry.get("category", "general")),
                note=str(entry.get("note", "")),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class Outcome:
    """What happened for one case."""

    case: Case
    #: Tool names returned, best first.
    names: tuple[str, ...] = ()
    #: The command text for each, parallel to `names`.
    commands: tuple[str, ...] = ()
    dangerous_flags: tuple[bool, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def rank(self) -> int | None:
        """Position of the first acceptable tool, or None."""
        for position, name in enumerate(self.names):
            if name in self.case.tools:
                return position
        return None

    @property
    def tool_ok(self) -> bool:
        """Whether the TOP answer is acceptable. Empty is correct for a
        no-match case and wrong for every other kind."""
        if self.case.expects_nothing:
            return not self.names
        return self.rank == 0

    @property
    def forbidden_hit(self) -> str | None:
        """A wrong-but-plausible tool in the top slot."""
        if not self.names:
            return None
        return self.names[0] if self.names[0] in self.case.forbid_tools else None

    @property
    def example_ok(self) -> bool | None:
        """Whether the chosen COMMAND satisfies expect/forbid.

        None when the question does not arise -- no expectations were written,
        or the tool was wrong, in which case the example cannot be judged
        without conflating two different failures.
        """
        if self.case.expects_nothing or not self.names:
            return None
        if not (self.case.expect or self.case.forbid):
            return None
        if self.rank is None:
            return None
        command = self.commands[self.rank] if self.rank < len(self.commands) else ""
        if any(token not in command for token in self.case.expect):
            return False
        return all(token not in command for token in self.case.forbid)

    @property
    def safety_ok(self) -> bool | None:
        """For a no-match case: did anything destructive reach the buffer?

        Separate from `example_ok` because the question is different. There
        is no acceptable tool to scope to, so the claim is about everything
        returned -- a hostile or nonsense query must not put `rm` or `dd` in
        someone's prompt by way of a stray lexical match. Returning nothing
        satisfies this trivially, which is correct: that is the ideal.
        """
        if not (self.case.expects_nothing and self.case.forbid):
            return None
        return not any(
            token in command
            for command in self.commands
            for token in self.case.forbid
        )

    @property
    def danger_ok(self) -> bool | None:
        """Whether a destructive answer was marked. None if not applicable."""
        if not self.case.dangerous or self.rank is None:
            return None
        if self.rank >= len(self.dangerous_flags):
            return None
        return self.dangerous_flags[self.rank]


@dataclass
class Report:
    """Aggregated scores. Every rate is over the cases it actually applies to."""

    outcomes: list[Outcome] = field(default_factory=list)

    def _applicable(self) -> list[Outcome]:
        return [o for o in self.outcomes if not o.case.expects_nothing]

    def tool_at(self, k: int) -> float:
        rows = self._applicable()
        if not rows:
            return 0.0
        hits = sum(1 for o in rows if o.rank is not None and o.rank < k)
        return hits / len(rows)

    def mrr(self) -> float:
        rows = self._applicable()
        if not rows:
            return 0.0
        return sum(1.0 / (o.rank + 1) for o in rows if o.rank is not None) / len(rows)

    def example_accuracy(self) -> tuple[float, int]:
        judged = [o for o in self.outcomes if o.example_ok is not None]
        if not judged:
            return 0.0, 0
        return sum(1 for o in judged if o.example_ok) / len(judged), len(judged)

    def no_match_accuracy(self) -> tuple[float, int]:
        """Of the cases that should return nothing, how many did?"""
        rows = [o for o in self.outcomes if o.case.expects_nothing]
        if not rows:
            return 0.0, 0
        return sum(1 for o in rows if not o.names) / len(rows), len(rows)

    def safety(self) -> tuple[float, int]:
        """Of the no-match cases that named destructive tokens, how many
        stayed clear of them?"""
        judged = [o for o in self.outcomes if o.safety_ok is not None]
        if not judged:
            return 0.0, 0
        return sum(1 for o in judged if o.safety_ok) / len(judged), len(judged)

    def danger_recall(self) -> tuple[float, int]:
        judged = [o for o in self.outcomes if o.danger_ok is not None]
        if not judged:
            return 0.0, 0
        return sum(1 for o in judged if o.danger_ok) / len(judged), len(judged)

    def forbidden_rate(self) -> tuple[float, int]:
        """How often a known-wrong neighbour took the top slot."""
        rows = [o for o in self.outcomes if o.case.forbid_tools]
        if not rows:
            return 0.0, 0
        return sum(1 for o in rows if o.forbidden_hit) / len(rows), len(rows)

    def latency(self) -> tuple[float, float]:
        times = sorted(o.elapsed_ms for o in self.outcomes)
        if not times:
            return 0.0, 0.0
        return times[len(times) // 2], times[-1]

    def by(self, attribute: str) -> dict[str, tuple[float, int]]:
        """tool@1 grouped by persona, style or category.

        The grouping is the point. A single aggregate hides that completion
        is excellent and the vocabulary gap is unsolved, which are different
        problems needing different work.
        """
        buckets: dict[str, list[Outcome]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(getattr(outcome.case, attribute), []).append(outcome)
        return {
            key: (sum(1 for o in rows if o.tool_ok) / len(rows), len(rows))
            for key, rows in sorted(buckets.items())
        }

    def failures(self) -> list[Outcome]:
        return [
            o
            for o in self.outcomes
            if not o.tool_ok
            or o.example_ok is False
            or o.safety_ok is False
            or o.forbidden_hit
        ]


def score(outcomes: Sequence[Outcome]) -> Report:
    return Report(list(outcomes))
