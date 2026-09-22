"""Tests for the catalog build pipeline.

The property under test is resilience. A build runs on a stranger's machine
against whatever documentation happens to be installed, so every stage has to
degrade rather than fail: the user asked for command completion, not for a
stack trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from cl_ai.catalog.build import BuildResult, build, default_sources
from cl_ai.catalog.discovery import Discovered, EntryKind
from cl_ai.catalog.extract.base import RawTool
from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.store import load
from cl_ai.ir import SourceTier

FIXTURES = Path(__file__).parent / "fixtures" / "tldr"


class FakeSource:
    """A stand-in tier, so the pipeline can be tested without a corpus."""

    def __init__(
        self,
        name: str,
        findings: Iterable[RawTool] = (),
        *,
        available: bool = True,
        raises: bool = False,
        tier: SourceTier = SourceTier.TLDR,
    ) -> None:
        self.name = name
        self.tier = tier
        self._findings = list(findings)
        self._available = available
        self._raises = raises
        self.asked_for: list[Iterable[str] | None] = []

    def available(self) -> bool:
        if self._raises:
            raise RuntimeError("probe exploded")
        return self._available

    def harvest(self, binaries: Iterable[str] | None = None) -> Iterator[RawTool]:
        self.asked_for.append(binaries)
        if self._raises:
            raise RuntimeError("harvest exploded")
        yield from self._findings


def raw(binary: str, source: str = "s") -> RawTool:
    return RawTool(
        binary=binary,
        path=(),
        description=f"{binary} does a thing",
        tier=SourceTier.TLDR,
        source=source,
        os_targets=frozenset({"common"}),
    )


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

def test_build_produces_a_catalog_and_a_report(tmp_path: Path) -> None:
    result = build(
        [FakeSource("fake", [raw("git"), raw("ls")])],
        path=tmp_path / "c.json",
    )
    assert isinstance(result, BuildResult)
    assert len(result.catalog) == 2
    assert result.findings == 2
    assert result.used == ("fake",)
    assert result.ok


def test_build_writes_a_loadable_artifact(tmp_path: Path) -> None:
    target = tmp_path / "c.json"
    result = build([FakeSource("fake", [raw("git")])], path=target)
    assert result.path == target
    loaded = load(target)
    assert loaded is not None
    assert loaded.catalog == result.catalog
    assert loaded.built_with == ("fake",)


def test_write_can_be_disabled(tmp_path: Path) -> None:
    result = build([FakeSource("fake", [raw("git")])], path=tmp_path / "c.json",
                   write=False)
    assert result.path is None
    assert not (tmp_path / "c.json").exists()


def test_binaries_filter_is_passed_through(tmp_path: Path) -> None:
    """Schematising 7,000 commands when 1,200 are on PATH is wasted work."""
    source = FakeSource("fake", [raw("git")])
    build([source], binaries=["git"], path=tmp_path / "c.json")
    assert source.asked_for == [["git"]]


def test_cache_key_is_set_only_when_discovery_is_supplied(tmp_path: Path) -> None:
    entries = [
        Discovered(name="git", kind=EntryKind.EXECUTABLE, path="/usr/bin/git",
                   size=1, mtime_ns=2)
    ]
    keyed = build([FakeSource("fake", [raw("git")])], discovered=entries,
                  path=tmp_path / "a.json")
    unkeyed = build([FakeSource("fake", [raw("git")])], path=tmp_path / "b.json")
    assert keyed.key
    assert unkeyed.key == ""

    loaded = load(tmp_path / "a.json")
    assert loaded is not None
    assert loaded.is_valid_for(keyed.key)


def test_cache_key_covers_the_tier_set(tmp_path: Path) -> None:
    """A catalog goes stale when the build changes, not only the machine."""
    entries = [
        Discovered(name="git", kind=EntryKind.EXECUTABLE, path="/usr/bin/git",
                   size=1, mtime_ns=2)
    ]
    one = build([FakeSource("a", [raw("git")])], discovered=entries,
                path=tmp_path / "1.json")
    two = build(
        [FakeSource("a", [raw("git")]), FakeSource("b", [raw("ls")])],
        discovered=entries,
        path=tmp_path / "2.json",
    )
    assert one.key != two.key


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------

def test_unavailable_tier_is_skipped_not_fatal(tmp_path: Path) -> None:
    result = build(
        [
            FakeSource("present", [raw("git")]),
            FakeSource("absent", [raw("ls")], available=False),
        ],
        path=tmp_path / "c.json",
    )
    assert result.used == ("present",)
    assert result.skipped == ("absent",)
    assert len(result.catalog) == 1


def test_a_broken_tier_costs_only_that_tier(tmp_path: Path) -> None:
    """A future or third-party tier that raises must not sink the build."""
    result = build(
        [FakeSource("good", [raw("git")]), FakeSource("bad", raises=True)],
        path=tmp_path / "c.json",
    )
    assert result.used == ("good",)
    assert result.skipped == ("bad",)
    assert len(result.catalog) == 1


def test_no_tiers_at_all_still_produces_an_artifact(tmp_path: Path) -> None:
    """An empty catalog is a valid answer; a missing file is not."""
    result = build([], path=tmp_path / "c.json")
    assert len(result.catalog) == 0
    assert result.ok
    assert result.path is not None
    assert load(result.path) is not None


def test_every_tier_unavailable(tmp_path: Path) -> None:
    result = build(
        [FakeSource("a", available=False), FakeSource("b", available=False)],
        path=tmp_path / "c.json",
    )
    assert result.used == ()
    assert result.skipped == ("a", "b")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_lint_runs_over_the_raw_findings(tmp_path: Path) -> None:
    """Placeholder leaks are only detectable before the merge.

    If the pipeline linted the catalog alone, this defect would be invisible.
    """
    from cl_ai.catalog.extract.base import RawExample

    leaky = RawTool(
        binary="t",
        path=(),
        description="d",
        tier=SourceTier.TLDR,
        source="page.md",
        os_targets=frozenset({"common"}),
        examples=(
            RawExample(description="d", template="t {{f}}", literal="t {{f}}"),
        ),
    )
    result = build([FakeSource("fake", [leaky])], path=tmp_path / "c.json")
    assert not result.ok
    assert "placeholder-leak" in {f.code for f in result.report.errors}


def test_summary_is_human_readable(tmp_path: Path) -> None:
    result = build([FakeSource("fake", [raw("git")])], path=tmp_path / "c.json")
    text = result.summary()
    assert "1 tools" in text
    assert "fake" in text
    assert "errors=0" in text


def test_summary_mentions_skipped_tiers(tmp_path: Path) -> None:
    result = build(
        [FakeSource("fake", [raw("git")]), FakeSource("gone", available=False)],
        path=tmp_path / "c.json",
    )
    assert "skipped gone" in result.summary()


def test_summary_with_no_tiers(tmp_path: Path) -> None:
    assert "no tiers" in build([], path=tmp_path / "c.json").summary()


# --------------------------------------------------------------------------
# With the real fixture corpus
# --------------------------------------------------------------------------

def test_end_to_end_over_the_fixtures(tmp_path: Path) -> None:
    target = tmp_path / "c.json"
    result = build([TldrSource(FIXTURES)], path=target)
    assert result.used == ("tldr",)
    assert len(result.catalog) > 5
    assert result.ok, [str(f) for f in result.report.errors]

    loaded = load(target)
    assert loaded is not None
    tool = loaded.catalog.select("git_commit", "bash", "linux")
    assert tool is not None
    assert tool.binary == "git"
    assert any(p.flag == "--message" for p in tool.params)


def test_build_is_reproducible(tmp_path: Path) -> None:
    """Two builds of the same inputs must give the same bytes."""
    a = build([TldrSource(FIXTURES)], path=tmp_path / "a.json")
    b = build([TldrSource(FIXTURES)], path=tmp_path / "b.json")
    assert a.catalog == b.catalog
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_default_sources_are_constructible() -> None:
    """Whether a corpus is installed or not, this must not raise."""
    sources = default_sources()
    assert sources
    for source in sources:
        assert isinstance(source.available(), bool)
