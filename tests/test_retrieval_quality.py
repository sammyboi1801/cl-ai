"""Retrieval QUALITY, measured against realistic queries.

Every other test here asserts a mechanism. This one asserts an outcome, and it
is the only test that can catch the failure that matters most: retrieval that
runs perfectly and ranks badly. Nothing about a wrong ranking raises, so
without a measured floor it rots silently.

The thresholds are floors, not targets, and are set below what the
implementation currently achieves. That gap is deliberate: a test that pins the
exact current number fails on every harmless reordering and gets deleted, while
a floor only fires on a real regression.

Scored against the committed fixture corpus, which is ten tools -- enough to
catch a regression, far too small to say anything about ranking at scale. The
figures below come from `tests/eval_retrieval.py` against the full 7,367-page
tldr corpus, and are reproducible with:

    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_retrieval

    unrestricted (4,822 tools)   hit@1 0.519  hit@3 0.741  mrr 0.630  p50 11ms
    gated to installed (1,240)   hit@1 0.704  hit@5 0.815  mrr 0.741  p50 10ms

The gated figure is the one a user actually experiences, and the gap between
them is itself a finding: restricting to what is installed is a QUALITY signal,
not merely a correctness one. Obscure tools are often excellent textual
matches -- `gdown` really is for downloading a file -- so removing what cannot
be run promotes the canonical answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.normalize import normalize
from cl_ai.retrieval.index import ToolIndex

FIXTURES = Path(__file__).parent / "fixtures" / "tldr"

#: (query, tools that would be a correct answer).
#:
#: Several answers are accepted because several tools genuinely satisfy some of
#: these. Insisting on one would measure my preferences, not retrieval.
CASES: tuple[tuple[str, frozenset[str]], ...] = (
    # Command fragments being completed -- Tab semantics.
    ("git com", frozenset({"git_commit"})),
    ("git commit", frozenset({"git_commit"})),
    ("pacman", frozenset({"pacman"})),
    ("apt-g", frozenset({"apt_get"})),
    # Natural language against fixture descriptions and intents.
    ("commit staged files with a message", frozenset({"git_commit"})),
    ("commit files to the repository", frozenset({"git_commit"})),
    ("remove a directory recursively", frozenset({"rm"})),
    ("list directory contents", frozenset({"ls", "MOUNT", "dir"})),
    ("install a package", frozenset({"pacman_sync", "apt_get", "pm_install_commit"})),
    ("synchronize packages", frozenset({"pacman_sync"})),
    ("query the package database", frozenset({"pacman_query"})),
)


@pytest.fixture(scope="module")
def index() -> ToolIndex:
    catalog = normalize(TldrSource(FIXTURES).harvest())
    built = ToolIndex.from_catalog(catalog, shell="bash", os_name="linux")
    assert len(built) > 5, "fixture corpus looks empty"
    return built


def _measure(index: ToolIndex, limit: int = 5) -> dict[str, float]:
    hit1 = hit3 = hit5 = 0
    reciprocal = 0.0
    for query, wanted in CASES:
        names = [c.tool.name for c in index.search(query, limit=limit)]
        rank = next((i for i, n in enumerate(names) if n in wanted), None)
        if rank is None:
            continue
        reciprocal += 1.0 / (rank + 1)
        hit1 += rank < 1
        hit3 += rank < 3
        hit5 += rank < 5
    total = len(CASES)
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total,
        "mrr": reciprocal / total,
    }


def test_quality_floor(index: ToolIndex) -> None:
    """Floors, not targets. See the module docstring."""
    scores = _measure(index)
    assert scores["hit@1"] >= 0.60, scores
    assert scores["hit@3"] >= 0.80, scores
    assert scores["hit@5"] >= 0.80, scores
    assert scores["mrr"] >= 0.70, scores


@pytest.mark.parametrize(("query", "wanted"), CASES, ids=[c[0] for c in CASES])
def test_each_query_finds_something_acceptable(
    index: ToolIndex, query: str, wanted: frozenset[str]
) -> None:
    """Per-case, so a failure names the query rather than a summary statistic."""
    names = [c.tool.name for c in index.search(query, limit=5)]
    assert set(names) & wanted, f"{query!r} -> {names}"


def test_completion_queries_rank_the_exact_tool_first(index: ToolIndex) -> None:
    """Tab semantics: a typed prefix of a real command IS the answer."""
    for query, wanted in (
        ("git com", "git_commit"),
        ("git commit", "git_commit"),
        ("apt-g", "apt_get"),
    ):
        names = [c.tool.name for c in index.search(query, limit=5)]
        assert names and names[0] == wanted, f"{query!r} -> {names}"


def test_a_nonsense_query_returns_nothing(index: ToolIndex) -> None:
    """The honest "no match" the score floor exists to produce.

    A nearest-neighbour substitution here is how a missing tool turns into a
    confidently wrong command.
    """
    for query in ("zzqqxxyy", "asdfghjkl", "qwertyuiop zxcvbnm"):
        assert index.search(query) == [], query


def test_intent_phrasing_beats_name_matching(index: ToolIndex) -> None:
    """The regression that cost hit@1 0.20 -> 0.40.

    Example descriptions are the only text phrased as intent. With only names
    and tool descriptions indexed, "copy a file" returned the tool literally
    named `file`.
    """
    names = [c.tool.name for c in index.search("commit staged files with a message")]
    assert names[0] == "git_commit"


def test_search_latency_is_within_the_keystroke_budget(index: ToolIndex) -> None:
    """Retrieval sits behind Tab, so a slow path is a hung prompt.

    Generous on purpose -- CI runners are slow and shared, and this is meant to
    catch an algorithmic regression, not to benchmark the machine.
    """
    import time

    worst = 0.0
    for query, _ in CASES:
        start = time.perf_counter()
        index.search(query, limit=5)
        worst = max(worst, (time.perf_counter() - start) * 1000)
    assert worst < 500.0, f"slowest query took {worst:.1f}ms"


def test_ranking_is_stable_across_runs(index: ToolIndex) -> None:
    """A user cycling candidates must not see them reshuffle."""
    for query, _ in CASES:
        first = [c.tool.name for c in index.search(query, limit=5)]
        assert first == [c.tool.name for c in index.search(query, limit=5)]
