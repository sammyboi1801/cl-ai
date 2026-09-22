"""Tests for stage 3 -- merging raw findings into a catalog.

The cases that matter here are the ones where a careless merge invents a tool
no source described. Those are not hypothetical: `dir` really is documented
three times, for a router, for cmd, and for GNU coreutils.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cl_ai.catalog.extract.base import RawExample, RawParam, RawTool
from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.normalize import (
    Catalog,
    Family,
    family_for_os_name,
    family_of,
    normalize,
)
from cl_ai.ir import Capability, ParamKind, SourceTier

FIXTURES = Path(__file__).parent / "fixtures" / "tldr"


def raw(
    binary: str = "tool",
    *,
    path: tuple[str, ...] = (),
    description: str = "does a thing",
    tier: SourceTier = SourceTier.TLDR,
    source: str = "s",
    os_targets: frozenset[str] = frozenset({"common"}),
    params: tuple[RawParam, ...] = (),
    examples: tuple[RawExample, ...] = (),
    homepage: str | None = None,
) -> RawTool:
    return RawTool(
        binary=binary,
        path=path,
        description=description,
        tier=tier,
        source=source,
        os_targets=os_targets,
        params=params,
        examples=examples,
        homepage=homepage,
    )


def example(literal: str = "tool --x", description: str = "d") -> RawExample:
    return RawExample(description=description, template=literal, literal=literal)


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["common"], Family.COMMON),
        (["linux"], Family.POSIX),
        (["osx"], Family.POSIX),
        (["android"], Family.POSIX),
        (["windows"], Family.WINDOWS),
        (["dos"], Family.WINDOWS),
        (["cisco-ios"], Family.APPLIANCE),
        (["plan9"], Family.UNKNOWN),
        ([], Family.UNKNOWN),
    ],
)
def test_family_of(tags: list[str], expected: Family) -> None:
    assert family_of(tags) is expected


def test_mixed_concrete_families_are_unknown() -> None:
    """Refusing to pick is the point: a finding we cannot place stays visible."""
    assert family_of(["linux", "windows"]) is Family.UNKNOWN


def test_common_plus_one_concrete_family_resolves_to_the_concrete_one() -> None:
    assert family_of(["common", "linux"]) is Family.POSIX


@pytest.mark.parametrize(
    ("os_name", "expected"),
    [
        ("win32", Family.WINDOWS),
        ("windows", Family.WINDOWS),
        ("darwin", Family.POSIX),
        ("linux", Family.POSIX),
        ("freebsd13", Family.POSIX),   # sys.platform carries a version suffix
        ("plan9", Family.UNKNOWN),
        ("", Family.UNKNOWN),
    ],
)
def test_family_for_os_name(os_name: str, expected: Family) -> None:
    assert family_for_os_name(os_name) is expected


def test_only_concrete_families_are_concrete() -> None:
    assert Family.POSIX.is_concrete
    assert Family.WINDOWS.is_concrete
    assert not Family.COMMON.is_concrete
    assert not Family.UNKNOWN.is_concrete


# --------------------------------------------------------------------------
# Variant separation -- the correctness case
# --------------------------------------------------------------------------

def test_same_name_different_families_do_not_merge() -> None:
    """Windows `dir`, GNU `dir` and Cisco `dir` share only a name.

    A single merged entry would draw its examples from a router, a cmd prompt
    and a Linux box at once.
    """
    catalog = normalize([
        raw("dir", description="list directory contents", os_targets=frozenset({"windows"}),
            source="w", examples=(example("dir /w"),)),
        raw("dir", description="alias of ls", os_targets=frozenset({"linux"}),
            source="l", examples=(example("dir -C"),)),
        raw("dir", description="show flash contents", os_targets=frozenset({"cisco-ios"}),
            source="c", examples=(example("dir flash:"),)),
    ])
    variants = catalog.by_name["dir"]
    assert len(variants) == 3
    assert {v.family for v in variants} == {
        Family.WINDOWS,
        Family.POSIX,
        Family.APPLIANCE,
    }
    for variant in variants:
        assert len(variant.tool.examples) == 1


def test_same_family_findings_do_merge() -> None:
    catalog = normalize([
        raw("ls", os_targets=frozenset({"linux"}), source="a", examples=(example("ls -l"),)),
        raw("ls", os_targets=frozenset({"osx"}), source="b", examples=(example("ls -G"),)),
    ])
    variants = catalog.by_name["ls"]
    assert len(variants) == 1
    assert {e.command for e in variants[0].tool.examples} == {"ls -l", "ls -G"}


# --------------------------------------------------------------------------
# Selection needs BOTH axes
# --------------------------------------------------------------------------

@pytest.fixture()
def dir_catalog() -> Catalog:
    return normalize([
        raw("dir", description="windows dir", os_targets=frozenset({"windows"}), source="w"),
        raw("dir", description="gnu dir", os_targets=frozenset({"linux"}), source="l"),
    ])


def test_selection_uses_os_to_disambiguate_a_shared_shell(dir_catalog: Catalog) -> None:
    """pwsh runs on Windows AND Linux, so the shell alone cannot decide.

    This was a real bug: both variants list pwsh, so the tie broke
    alphabetically and handed a Windows user the GNU page.
    """
    assert dir_catalog.select("dir", "pwsh", "win32").description == "windows dir"
    assert dir_catalog.select("dir", "pwsh", "linux").description == "gnu dir"


def test_selection_by_shell_alone_still_works_when_unambiguous(
    dir_catalog: Catalog,
) -> None:
    assert dir_catalog.select("dir", "cmd").description == "windows dir"
    assert dir_catalog.select("dir", "bash").description == "gnu dir"


def test_wrong_family_is_dropped_not_ranked() -> None:
    """GNU `dir` does not exist on Windows; it is wrong, not merely worse."""
    catalog = normalize([
        raw("dir", description="gnu dir", os_targets=frozenset({"linux"}), source="l"),
    ])
    assert catalog.select("dir", "pwsh", "win32") is None


def test_concrete_family_beats_common() -> None:
    """A platform-specific page exists precisely because behaviour differs."""
    catalog = normalize([
        raw("ls", description="generic", os_targets=frozenset({"common"}), source="c"),
        raw("ls", description="linux specific", os_targets=frozenset({"linux"}), source="l"),
    ])
    assert catalog.select("ls", "bash", "linux").description == "linux specific"


def test_common_wins_when_nothing_is_known() -> None:
    catalog = normalize([
        raw("ls", description="generic", os_targets=frozenset({"common"}), source="c"),
        raw("ls", description="linux specific", os_targets=frozenset({"linux"}), source="l"),
    ])
    assert catalog.select("ls").description == "generic"


def test_unknown_os_widens_rather_than_empties(dir_catalog: Catalog) -> None:
    """An unrecognised OS must not look identical to the tool not existing."""
    assert dir_catalog.select("dir", "cmd", "plan9") is not None


def test_select_missing_name() -> None:
    assert normalize([]).select("nope", "bash") is None


def test_select_shell_that_cannot_run_it() -> None:
    catalog = normalize([
        raw("dir", os_targets=frozenset({"windows"}), source="w"),
    ])
    assert catalog.select("dir", "bash") is None


def test_for_target_returns_one_variant_per_name() -> None:
    catalog = normalize([
        raw("dir", description="windows", os_targets=frozenset({"windows"}), source="w"),
        raw("dir", description="gnu", os_targets=frozenset({"linux"}), source="l"),
        raw("git", os_targets=frozenset({"common"}), source="g"),
    ])
    tools = catalog.for_target("pwsh", "win32")
    names = [t.name for t in tools]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert {t.description for t in tools if t.name == "dir"} == {"windows"}


# --------------------------------------------------------------------------
# Tier precedence and conflicts
# --------------------------------------------------------------------------

def test_highest_tier_wins() -> None:
    catalog = normalize([
        raw("tool", description="from tldr", tier=SourceTier.TLDR, source="tldr"),
        raw("tool", description="from native", tier=SourceTier.NATIVE, source="native"),
    ])
    assert catalog.select("tool").description == "from native"


def test_losing_value_is_recorded_not_discarded() -> None:
    catalog = normalize([
        raw("tool", description="from tldr", tier=SourceTier.TLDR, source="tldr"),
        raw("tool", description="from native", tier=SourceTier.NATIVE, source="native"),
    ])
    conflicts = [c for c in catalog.conflicts if c.field == "description"]
    assert len(conflicts) == 1
    assert conflicts[0].kept == "from native"
    assert conflicts[0].dropped == "from tldr"
    assert not conflicts[0].same_tier
    tool = catalog.select("tool")
    assert tool.provenance is not None
    assert "from tldr" in tool.provenance.conflicts


def test_same_tier_disagreement_is_flagged() -> None:
    """Two equal-standing sources contradicting each other wants a human."""
    catalog = normalize([
        raw("tool", description="alpha", source="a"),
        raw("tool", description="beta", source="b"),
    ])
    conflicts = [c for c in catalog.conflicts if c.field == "description"]
    assert conflicts and all(c.same_tier for c in conflicts)


def test_identical_values_are_not_conflicts() -> None:
    catalog = normalize([
        raw("tool", description="same", source="a"),
        raw("tool", description="same", source="b"),
    ])
    assert [c for c in catalog.conflicts if c.field == "description"] == []


def test_empty_value_does_not_beat_a_real_one() -> None:
    """An absent description is not a claim, even from a better tier."""
    catalog = normalize([
        raw("tool", description="", tier=SourceTier.NATIVE, source="native"),
        raw("tool", description="real", tier=SourceTier.TLDR, source="tldr"),
    ])
    assert catalog.select("tool").description == "real"


def test_merge_is_deterministic() -> None:
    """Same inputs in any order must give a byte-identical catalog."""
    findings = [
        raw("tool", description="alpha", source="a"),
        raw("tool", description="beta", source="b"),
        raw("other", description="gamma", source="c"),
    ]
    first = normalize(findings)
    second = normalize(list(reversed(findings)))
    assert first == second
    assert [v.tool.name for v in first.variants] == [
        v.tool.name for v in second.variants
    ]


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

def test_option_spellings_do_not_become_enum_values() -> None:
    """`-m` is a way to write --message, not a value it accepts.

    Copying spellings into enum would let a planner pass "-m" as the message.
    """
    catalog = normalize([
        raw(
            "tool",
            params=(
                RawParam(
                    name="message",
                    kind=ParamKind.OPTION,
                    short="-m",
                    long="--message",
                    choices=("-m", "--message"),
                    type="string",
                ),
            ),
        )
    ])
    param = catalog.select("tool").params[0]
    assert param.flag == "--message"
    assert param.short == "-m"
    assert param.enum == ()


def test_subcommand_choices_do_become_enum_values() -> None:
    """`{{[add|install]}}` genuinely is a value set."""
    catalog = normalize([
        raw(
            "tool",
            params=(
                RawParam(
                    name="add",
                    kind=ParamKind.SUBCOMMAND,
                    choices=("add", "install"),
                ),
            ),
        )
    ])
    param = catalog.select("tool").params[0]
    assert param.kind is ParamKind.SUBCOMMAND
    assert param.enum == ("add", "install")


def test_params_merge_by_long_spelling() -> None:
    catalog = normalize([
        raw("tool", source="a", params=(
            RawParam(name="v", kind=ParamKind.OPTION, short="-v", long="--verbose"),
        )),
        raw("tool", source="b", params=(
            RawParam(name="verbose", kind=ParamKind.OPTION, long="--verbose"),
        )),
    ])
    options = [p for p in catalog.select("tool").params if p.flag == "--verbose"]
    assert len(options) == 1


def test_param_type_survives_the_merge() -> None:
    catalog = normalize([
        raw("tool", params=(
            RawParam(name="count", kind=ParamKind.POSITIONAL, type="integer"),
        ))
    ])
    assert catalog.select("tool").params[0].type == "integer"


def test_unrenderable_option_is_dropped_with_a_record() -> None:
    """Param would reject it; one bad parameter must not sink a catalog."""
    catalog = normalize([
        raw("tool", params=(RawParam(name="mystery", kind=ParamKind.OPTION),))
    ])
    assert catalog.select("tool").params == ()
    assert any("mystery" in c.dropped for c in catalog.conflicts)


def test_param_provenance_is_kept() -> None:
    catalog = normalize([
        raw("tool", source="page.md", params=(
            RawParam(name="f", kind=ParamKind.OPTION, long="--flag"),
        ))
    ])
    param = catalog.select("tool").params[0]
    assert param.provenance is not None
    assert param.provenance.source == "page.md"
    assert param.provenance.tier is SourceTier.TLDR


# --------------------------------------------------------------------------
# Other merged fields
# --------------------------------------------------------------------------

def test_capabilities_are_unioned_not_picked() -> None:
    """Destructive evidence from any source has to survive the merge.

    A needless warning is cheap; an unflagged `rm -rf` is not.
    """
    catalog = normalize([
        raw("rm", tier=SourceTier.NATIVE, source="native", description="remove"),
        raw("rm", tier=SourceTier.TLDR, source="tldr", examples=(example("rm -rf x"),)),
    ])
    assert Capability.DESTRUCTIVE in catalog.select("rm").capabilities


def test_examples_are_additive_and_deduplicated() -> None:
    catalog = normalize([
        raw("tool", source="a", examples=(example("tool --x"), example("tool --y"))),
        raw("tool", source="b", examples=(example("tool --x"), example("tool --z"))),
    ])
    examples = catalog.select("tool").examples
    assert sorted(e.command for e in examples) == ["tool --x", "tool --y", "tool --z"]


def test_platforms_union_across_os_targets() -> None:
    catalog = normalize([
        raw("tool", os_targets=frozenset({"linux"}), source="a"),
        raw("tool", os_targets=frozenset({"osx"}), source="b"),
    ])
    assert "bash" in catalog.select("tool").platforms


def test_homepage_is_kept() -> None:
    catalog = normalize([raw("tool", homepage="https://example.com")])
    assert catalog.select("tool").homepage == "https://example.com"


def test_subcommand_path_is_preserved() -> None:
    catalog = normalize([raw("git", path=("commit",))])
    tool = catalog.select("git_commit")
    assert tool is not None
    assert tool.binary == "git"
    assert tool.path == ("commit",)


# --------------------------------------------------------------------------
# Catalog mechanics
# --------------------------------------------------------------------------

def test_empty_input() -> None:
    catalog = normalize([])
    assert len(catalog) == 0
    assert catalog.tools == ()
    assert catalog.select("anything") is None
    assert catalog.for_target("bash") == ()


def test_len_and_tools_agree() -> None:
    catalog = normalize([raw("a"), raw("b")])
    assert len(catalog) == len(catalog.tools) == 2


def test_normalize_accepts_a_generator() -> None:
    """The extractor yields lazily; normalize must not require a list."""
    catalog = normalize(f for f in [raw("a"), raw("b")])
    assert len(catalog) == 2


# --------------------------------------------------------------------------
# Against the real fixture corpus
# --------------------------------------------------------------------------

def test_fixture_corpus_merges_cleanly() -> None:
    findings = list(TldrSource(FIXTURES).harvest())
    catalog = normalize(findings)
    assert len(catalog) > 0
    assert catalog.rejected == ()
    # Every name resolves to at most one variant per family.
    for name, variants in catalog.by_name.items():
        families = [v.family for v in variants]
        assert len(families) == len(set(families)), name


def test_pacman_flag_subcommands_stay_separate_tools() -> None:
    catalog = normalize(TldrSource(FIXTURES).harvest())
    assert catalog.select("pacman_sync", "bash", "linux") is not None
    assert catalog.select("pacman_query", "bash", "linux") is not None
    assert catalog.select("pacman_sync").path == ("--sync",)


def test_git_commit_from_fixtures_has_renderable_options() -> None:
    catalog = normalize(TldrSource(FIXTURES).harvest())
    tool = catalog.select("git_commit", "bash", "linux")
    assert tool is not None
    options = [p for p in tool.params if p.kind is ParamKind.OPTION]
    assert options
    for option in options:
        assert option.flag or option.short
