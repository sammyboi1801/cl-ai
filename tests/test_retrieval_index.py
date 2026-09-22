"""Tests for the tool index: gates, the score floor, diversity, and ordering.

The gates are the part that must not be softened. A tool that is not installed
cannot be run, so ranking it lower is the wrong answer -- it must not appear.
And the coverage floor exists so "no match" is a real outcome rather than a
nearest-neighbour substitution, which is how a missing tool becomes a wrong
command.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.normalize import normalize
from cl_ai.ir import Capability, ContextFacts, Example, Tool
from cl_ai.retrieval.index import (
    DEFAULT_FIELDS,
    MAX_PER_BINARY,
    MIN_COVERAGE,
    Candidate,
    ToolIndex,
)
from cl_ai.retrieval.vectors import VectorReranker

from .test_retrieval_vectors import AXES, FakeEmbedder

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures" / "tldr"


def tool(
    name: str,
    description: str = "",
    *,
    binary: str | None = None,
    path: tuple[str, ...] = (),
    intents: Sequence[str] = (),
    platforms: frozenset[str] = frozenset({"bash"}),
    capabilities: frozenset[Capability] = frozenset(),
) -> Tool:
    return Tool(
        name=name,
        description=description,
        binary=binary or name.split("_")[0],
        path=path,
        platforms=platforms,
        capabilities=capabilities,
        examples=tuple(
            Example(description=i, command=f"{name} --x") for i in intents
        ),
    )


def index_of(*tools: Tool) -> ToolIndex:
    return ToolIndex(tools=tools)


# --------------------------------------------------------------------------
# Basic retrieval
# --------------------------------------------------------------------------

def test_finds_a_tool_by_description() -> None:
    idx = index_of(
        tool("ls", "list directory contents"),
        tool("cat", "print file contents"),
    )
    assert idx.search("list directory")[0].tool.name == "ls"


def test_finds_a_tool_by_intent() -> None:
    """Example descriptions are the highest-weighted field for a reason."""
    idx = index_of(
        tool("git_commit", "record changes", intents=["commit staged files"]),
        tool("git_log", "show history", intents=["view the commit log"]),
    )
    assert idx.search("commit staged files")[0].tool.name == "git_commit"


def test_empty_query_returns_nothing() -> None:
    idx = index_of(tool("ls", "list files"))
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_empty_index_returns_nothing() -> None:
    assert ToolIndex(tools=()).search("anything") == []
    assert len(ToolIndex(tools=())) == 0


def test_limit_is_respected() -> None:
    tools = tuple(tool(f"t{i}", "shared description here") for i in range(10))
    assert len(ToolIndex(tools=tools).search("shared", limit=3)) == 3


def test_results_are_candidates_with_explanations() -> None:
    idx = index_of(tool("ls", "list directory contents"))
    candidate = idx.search("list")[0]
    assert isinstance(candidate, Candidate)
    assert candidate.score > 0
    assert candidate.signals
    assert "list" in candidate.matched


def test_dangerous_is_surfaced() -> None:
    """The widget shows a destructive marker; it reads this."""
    idx = index_of(
        tool("rm", "remove files", capabilities=frozenset({Capability.DESTRUCTIVE}))
    )
    assert idx.search("remove files")[0].dangerous


# --------------------------------------------------------------------------
# Completion semantics -- what pressing Tab means
# --------------------------------------------------------------------------

def test_a_typed_prefix_wins() -> None:
    """`pyth` must find `python`. BM25 cannot: `pyth` is not a corpus term."""
    idx = index_of(
        tool("python", "an interpreted language"),
        tool("perl", "another language"),
    )
    assert idx.search("pyth")[0].tool.name == "python"


def test_the_closest_completion_comes_first() -> None:
    """`git com` means `git commit`, not `git commit-graph`.

    This regressed once: relevance and completion competed on one scale and
    completion lost, so `git commit` fell off the list entirely.
    """
    idx = index_of(
        tool("git_commit", "record changes", path=("commit",), binary="git",
             intents=["commit staged files", "amend a commit", "sign a commit"]),
        tool("git_commit_graph", "write a graph", path=("commit-graph",), binary="git"),
        tool("git_commit_tree", "low level tree", path=("commit-tree",), binary="git"),
    )
    assert idx.search("git com")[0].tool.name == "git_commit"


def test_exact_name_beats_a_longer_prefix_match() -> None:
    idx = index_of(tool("ls", "list"), tool("lsof", "list open files"))
    assert idx.search("ls")[0].tool.name == "ls"


def test_a_prefix_match_bypasses_the_coverage_floor() -> None:
    """`git com` matches almost none of the query's IDF mass, yet is correct."""
    idx = index_of(
        tool("git_commit", "", path=("commit",), binary="git"),
        tool("other", "unrelated"),
    )
    assert idx.search("git com")[0].tool.name == "git_commit"


def test_a_tail_only_prefix_match_does_not_bypass_the_floor() -> None:
    """`comma` starts with `com` but ignores the `git` the user typed.

    Admitting tail-only matches filled the list with tools that matched half
    the query.
    """
    idx = index_of(
        tool("git_commit", "record changes", path=("commit",), binary="git"),
        tool("comma", "run a command"),
        tool("comm", "compare files"),
    )
    names = [c.tool.name for c in idx.search("git com")]
    assert names[0] == "git_commit"
    assert "comma" not in names


# --------------------------------------------------------------------------
# The score floor
# --------------------------------------------------------------------------

def test_nonsense_query_returns_no_match() -> None:
    """An honest "nothing" beats a nearest-neighbour substitution."""
    idx = index_of(tool("ls", "list directory contents"), tool("cat", "print files"))
    assert idx.search("zzqqxx") == []


def test_a_tool_matching_only_a_common_word_is_floored_out() -> None:
    """Coverage is IDF-weighted, so a common word alone is not enough."""
    tools = tuple(
        tool(f"t{i}", "operates on a file somewhere") for i in range(20)
    ) + (tool("special", "compress a directory with zstd"),)
    idx = ToolIndex(tools=tools)
    names = [c.tool.name for c in idx.search("zstd file")]
    assert names[0] == "special"


def test_floor_can_be_relaxed() -> None:
    idx = index_of(tool("ls", "list directory contents"), tool("cat", "print files"))
    strict = idx.search("list unrelatedword", min_coverage=0.99)
    loose = idx.search("list unrelatedword", min_coverage=0.0)
    assert len(loose) >= len(strict)


def test_unknown_terms_do_not_drag_coverage_down() -> None:
    """A filename the user typed is not evidence against every tool.

    Counting unseen terms in the denominator made every query with an argument
    fall under the floor.
    """
    idx = index_of(tool("ls", "list directory contents"))
    assert idx.search("list zzqqxxfilename") != []


def test_coverage_is_reported() -> None:
    idx = index_of(tool("ls", "list directory contents"))
    assert idx.search("list directory")[0].coverage == pytest.approx(1.0)


def test_min_coverage_default_is_permissive() -> None:
    """Returning nothing is right only when there is genuinely nothing."""
    assert 0.0 < MIN_COVERAGE < 0.5


# --------------------------------------------------------------------------
# Hard gates
# --------------------------------------------------------------------------

def test_uninstalled_tools_are_excluded_not_demoted() -> None:
    """A tool that is not installed cannot be run.

    Ranking it lower would still show it; the gate has to remove it.
    """
    # Identical descriptions, so the gate is the ONLY thing separating them.
    # Giving them different text would let the coverage floor decide instead,
    # and the test would pass for the wrong reason.
    idx = index_of(
        tool("ls", "list files in a directory"),
        tool("exa", "list files in a directory"),
    )
    assert len(idx.search("list files in a directory")) == 2
    context = ContextFacts(installed=frozenset({"ls"}))
    names = [c.tool.name for c in idx.search("list files in a directory", context=context)]
    assert names == ["ls"]


def test_an_empty_installed_set_means_unknown_not_nothing() -> None:
    """ContextFacts defaults to an empty set; that must not gate everything out.

    An unpopulated context is "we did not look", and treating it as "nothing is
    installed" would make the whole catalog vanish.
    """
    idx = index_of(tool("ls", "list directory contents"))
    assert idx.search("list", context=ContextFacts()) != []


def test_gate_applies_to_the_binary_not_the_tool_name() -> None:
    """`git commit` is installed if `git` is."""
    idx = index_of(tool("git_commit", "record changes", path=("commit",), binary="git"))
    context = ContextFacts(installed=frozenset({"git"}))
    assert idx.search("record changes", context=context) != []


def test_platform_filtering_happens_at_index_time() -> None:
    """A filter after ranking silently shortens the list instead of refilling."""
    catalog = normalize(TldrSource(FIXTURES).harvest())
    posix = ToolIndex.from_catalog(catalog, shell="bash", os_name="linux")
    windows = ToolIndex.from_catalog(catalog, shell="cmd", os_name="win32")
    assert len(posix) > 0
    assert len(windows) > 0
    for tool_ in posix.tools:
        assert "bash" in tool_.platforms


def test_from_catalog_without_a_shell_indexes_everything() -> None:
    catalog = normalize(TldrSource(FIXTURES).harvest())
    assert len(ToolIndex.from_catalog(catalog)) == len(catalog.tools)


# --------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------

def test_one_binary_cannot_fill_the_list() -> None:
    """Without the cap, `git` returns git_commit, git_push, git_pull...

    The user cycling five candidates would see one binary.
    """
    tools = tuple(
        tool(f"git_{verb}", f"git {verb} does something", path=(verb,), binary="git")
        for verb in ("commit", "push", "pull", "fetch", "merge", "rebase")
    )
    idx = ToolIndex(tools=tools)
    results = idx.search("git does something", limit=5)
    assert len(results) <= MAX_PER_BINARY


def test_the_cap_is_configurable() -> None:
    tools = tuple(
        tool(f"git_{verb}", f"git {verb} thing", path=(verb,), binary="git")
        for verb in ("a", "b", "c", "d")
    )
    idx = ToolIndex(tools=tools)
    assert len(idx.search("git thing", max_per_binary=4)) == 4


def test_diversity_does_not_hide_a_different_binary() -> None:
    tools = (
        tool("git_a", "shared words here", path=("a",), binary="git"),
        tool("git_b", "shared words here", path=("b",), binary="git"),
        tool("git_c", "shared words here", path=("c",), binary="git"),
        tool("hg_a", "shared words here", path=("a",), binary="hg"),
    )
    names = [c.tool.name for c in ToolIndex(tools=tools).search("shared words")]
    assert "hg_a" in names


# --------------------------------------------------------------------------
# Prominence
# --------------------------------------------------------------------------

def test_a_bare_binary_is_preferred_over_a_deep_subcommand() -> None:
    """`ls` is the ordinary reading of "list files"; `git ls-files` is not."""
    idx = index_of(
        tool("ls", "list files in a directory"),
        tool("git_ls_files", "list files in a directory", path=("ls-files",),
             binary="git"),
    )
    assert idx.search("list files in a directory")[0].tool.name == "ls"


def test_prominence_does_not_bury_a_clearly_better_subcommand() -> None:
    """`docker container ls` must still win "list running containers"."""
    idx = index_of(
        tool("docker", "a container runtime"),
        tool("docker_container_ls", "list running containers",
             path=("container", "ls"), binary="docker",
             intents=["list all running containers"]),
    )
    top = idx.search("list all running containers")[0].tool.name
    assert top == "docker_container_ls"


# --------------------------------------------------------------------------
# Determinism and robustness
# --------------------------------------------------------------------------

def test_search_is_deterministic() -> None:
    tools = tuple(tool(f"t{i}", "identical description text") for i in range(10))
    idx = ToolIndex(tools=tools)
    first = [c.tool.name for c in idx.search("identical description")]
    assert first == [c.tool.name for c in idx.search("identical description")]


@pytest.mark.parametrize(
    "query",
    ["", "   ", "!!!", "---", "{{}}", "a" * 500, "🎉", "\x00", "SELECT * FROM x"],
)
def test_hostile_queries_never_raise(query: str) -> None:
    idx = index_of(tool("ls", "list directory contents"), tool("git", "vcs"))
    assert isinstance(idx.search(query), list)


def test_tools_with_no_text_at_all() -> None:
    idx = index_of(Tool(name="x", description="", binary="x"))
    assert idx.search("anything") == []
    assert idx.search("x") != []          # still findable by name


def test_default_fields_cover_the_document() -> None:
    names = {f.name for f in DEFAULT_FIELDS}
    assert names == {"name", "description", "intents", "commands"}


def test_intents_are_weighted_above_names() -> None:
    """The measured finding: name-heavy weighting scored hit@1 0.20."""
    by_name = {f.name: f for f in DEFAULT_FIELDS}
    assert by_name["intents"].weight > by_name["name"].weight


def test_length_normalisation_stays_low() -> None:
    """More documentation means more useful here, not more diluted."""
    for field in DEFAULT_FIELDS:
        assert field.b <= 0.5


def test_explain_produces_a_trace() -> None:
    idx = index_of(tool("ls", "list directory contents"))
    lines = idx.explain("list directory")
    assert lines and "ls" in lines[0]
    assert "coverage" in lines[0]


# --------------------------------------------------------------------------
# The vector half plugs in
# --------------------------------------------------------------------------

def test_search_works_without_a_vector_half() -> None:
    """The designed default: lexical alone."""
    idx = index_of(tool("ls", "list directory contents"))
    assert idx.search("list", vectors=None) != []


def test_an_unavailable_reranker_changes_nothing() -> None:
    idx = index_of(tool("ls", "list directory contents"), tool("cat", "print"))
    without = [c.tool.name for c in idx.search("list")]
    with_empty = [
        c.tool.name for c in idx.search("list", vectors=VectorReranker())
    ]
    assert without == with_empty


def test_the_vector_half_participates_when_available() -> None:
    tools = (
        tool("git", "version control", intents=["commit changes"]),
        tool("docker", "container runtime", intents=["list containers"]),
        tool("ls", "list directory contents", intents=["list files"]),
    )
    idx = ToolIndex(tools=tools)
    reranker = VectorReranker(embedder=FakeEmbedder(AXES)).build(tools)
    assert reranker.available()
    results = idx.search("list containers", vectors=reranker)
    assert [c.tool.name for c in results]


def test_a_broken_reranker_cannot_break_search() -> None:
    """The lexical half is the one that must always work."""

    class Hostile:
        def available(self) -> bool:
            return True

        def rerank(self, query: str, candidates: Sequence[int]) -> list[int]:
            return [9999, -1]

    idx = index_of(tool("ls", "list directory contents"))
    assert idx.search("list", vectors=Hostile()) != []


def test_the_semantic_weight_is_configurable_and_actually_applied() -> None:
    """The weight has to be re-measured per backend, and after any
    fine-tuning, so it cannot be a constant buried in the fusion.

    Swept over the 150-case set, no off-the-shelf embedder earned a positive
    weight: tool@1 fell monotonically from 0.322 at weight 0 to 0.271 at 1.0.
    A weight that silently did nothing would have hidden that.
    """
    from cl_ai.retrieval.index import SEMANTIC_WEIGHT

    tools = tuple(
        Tool(name=f"t{i}", description=f"tool number {i} for listing files",
             binary=f"t{i}")
        for i in range(6)
    )
    index = ToolIndex(tools=tools)

    class Reversing:
        """A reranker that inverts the lexical order, so any influence at all
        is visible in the result."""

        @staticmethod
        def available() -> bool:
            return True

        @staticmethod
        def rerank(query: str, candidates, depth=None):
            return list(reversed(list(candidates)))

    ignored = index.search("listing files", limit=6, vectors=Reversing(),
                           semantic_weight=0.0)
    heeded = index.search("listing files", limit=6, vectors=Reversing(),
                          semantic_weight=50.0)
    assert [c.tool.name for c in ignored] != [c.tool.name for c in heeded]
    assert 0.0 <= SEMANTIC_WEIGHT <= 2.0
