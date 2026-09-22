"""Tests for the catalog linter.

The linter's job is to catch entries that could produce a CONFIDENTLY WRONG
command, so the tests are mostly about keeping ERROR meaningful: every error
must be a real defect, and the legitimate-looking cases must not fire. A
report nobody trusts is a report nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cl_ai.catalog.extract.base import RawExample, RawTool
from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.lint import Finding, Severity, check_findings, lint
from cl_ai.catalog.normalize import Family, Variant, normalize
from cl_ai.ir import Capability, Param, ParamKind, Provenance, SourceTier, Tool

FIXTURES = Path(__file__).parent / "fixtures" / "tldr"


def catalog_of(*tools: Tool, family: Family = Family.COMMON):
    from cl_ai.catalog.normalize import Catalog

    return Catalog(variants=tuple(Variant(family=family, tool=t) for t in tools))


def tool(**kwargs) -> Tool:
    base = {
        "name": "sample",
        "description": "does a thing",
        "binary": "sample",
        "platforms": frozenset({"bash"}),
        "examples": ("sample --x",),
        "provenance": Provenance(tier=SourceTier.TLDR, source="s"),
    }
    base.update(kwargs)
    return Tool(**base)  # type: ignore[arg-type]


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


# --------------------------------------------------------------------------
# Clean input stays clean
# --------------------------------------------------------------------------

def test_a_good_tool_produces_no_errors() -> None:
    report = lint(catalog_of(tool()))
    assert report.errors == ()


def test_report_is_falsey_when_there_is_nothing_to_say() -> None:
    report = lint(catalog_of(tool(provenance=Provenance(
        tier=SourceTier.NATIVE, source="s"
    ))))
    assert not report
    assert report.worst_severity is None


def test_checked_counts_variants() -> None:
    assert lint(catalog_of(tool(name="a"), tool(name="b"))).checked == 2


# --------------------------------------------------------------------------
# Errors -- things that could render a wrong command
# --------------------------------------------------------------------------

def test_bad_identifier_is_an_error() -> None:
    report = lint(catalog_of(tool(name="not a key!")))
    assert "bad-identifier" in codes(report)
    assert report.worst_severity is Severity.ERROR


def test_missing_binary_is_an_error() -> None:
    assert "no-binary" in codes(lint(catalog_of(tool(binary=""))))


def test_duplicate_variant_in_one_family_is_an_error() -> None:
    """It would make select() nondeterministic."""
    report = lint(catalog_of(tool(name="dup"), tool(name="dup")))
    assert "duplicate-variant" in codes(report)


def test_same_name_in_different_families_is_not_a_duplicate() -> None:
    from cl_ai.catalog.normalize import Catalog

    catalog = Catalog(variants=(
        Variant(family=Family.WINDOWS, tool=tool(name="dir")),
        Variant(family=Family.POSIX, tool=tool(name="dir")),
    ))
    assert "duplicate-variant" not in codes(lint(catalog))


# --------------------------------------------------------------------------
# Warnings
# --------------------------------------------------------------------------

def test_option_carrying_enum_values_is_flagged() -> None:
    """Spellings in `enum` would let a planner pass "-m" as a value."""
    bad = tool(params=(
        Param(
            name="message",
            type="string",
            kind=ParamKind.OPTION,
            flag="--message",
            enum=("-m", "--message"),
        ),
    ))
    assert "spellings-as-values" in codes(lint(catalog_of(bad)))


def test_unreachable_tool_is_flagged() -> None:
    assert "unreachable" in codes(lint(catalog_of(tool(platforms=frozenset()))))


def test_appliance_tools_are_expected_to_be_unreachable() -> None:
    """A Cisco command runs on a router; no host shell is the correct answer."""
    report = lint(catalog_of(tool(platforms=frozenset()), family=Family.APPLIANCE))
    assert "unreachable" not in codes(report)


def test_shell_syntax_masquerading_as_a_tool_is_flagged() -> None:
    """`|` is the renderer's job, not an invocable command."""
    assert "shell-syntax-as-tool" in codes(lint(catalog_of(tool(name="pipe"))))


def test_destructive_without_a_description_is_flagged() -> None:
    """The widget shows a warning; the user needs something to judge."""
    bad = tool(description="", capabilities=frozenset({Capability.DESTRUCTIVE}))
    assert "unexplained-destructive" in codes(lint(catalog_of(bad)))


def test_bad_param_name_is_flagged() -> None:
    bad = tool(params=(Param(name="_weird", type="string"),))
    assert "bad-param-name" in codes(lint(catalog_of(bad)))


def test_rejected_findings_are_surfaced() -> None:
    from cl_ai.catalog.normalize import Catalog

    catalog = Catalog(rejected=(("page.md", "no usable identifier"),))
    report = lint(catalog)
    assert "rejected-finding" in codes(report)


# --------------------------------------------------------------------------
# Info
# --------------------------------------------------------------------------

def test_unschematized_tool_is_info_not_error() -> None:
    """"Exists but I have no schema" is a legitimate, useful state."""
    report = lint(catalog_of(tool(examples=(), params=())))
    assert "unschematized" in codes(report)
    assert report.errors == ()


def test_missing_description_is_info() -> None:
    assert "no-description" in codes(lint(catalog_of(tool(description=""))))


def test_low_tier_schema_is_flagged() -> None:
    bad = tool(provenance=Provenance(tier=SourceTier.HELPTEXT, source="s"))
    assert "low-tier-schema" in codes(lint(catalog_of(bad)))


def test_low_tier_enum_is_flagged() -> None:
    """Prose sources only ever show some values; treating them as the full
    set makes a planner reject valid arguments."""
    bad = tool(params=(
        Param(
            name="mode",
            type="string",
            kind=ParamKind.SUBCOMMAND,
            enum=("a", "b"),
            provenance=Provenance(tier=SourceTier.TLDR, source="s"),
        ),
    ))
    assert "low-tier-enum" in codes(lint(catalog_of(bad)))


def test_enum_from_a_trustworthy_tier_is_not_flagged() -> None:
    fine = tool(params=(
        Param(
            name="mode",
            type="string",
            kind=ParamKind.SUBCOMMAND,
            enum=("a", "b"),
            provenance=Provenance(tier=SourceTier.COMPLETIONS, source="s"),
        ),
    ))
    assert "low-tier-enum" not in codes(lint(catalog_of(fine)))


# --------------------------------------------------------------------------
# Placeholder leaks -- checked on RAW findings, where the evidence exists
# --------------------------------------------------------------------------

def raw_with(template: str, literal: str) -> RawTool:
    return RawTool(
        binary="t",
        path=(),
        description="d",
        tier=SourceTier.TLDR,
        source="page.md",
        examples=(RawExample(description="d", template=template, literal=literal),),
    )


def test_unsubstituted_slot_is_an_error() -> None:
    findings = [raw_with("t {{file}}", "t {{file}}")]
    report = lint(normalize(findings), extra=check_findings(findings))
    assert "placeholder-leak" in codes(report)
    assert report.worst_severity is Severity.ERROR


def test_escaped_go_template_is_not_a_leak() -> None:
    """`docker --format "{{.ID}}"` is a working command, not a defect.

    This is why the check lives on raw findings: the merged Tool has lost the
    template, and judging the literal alone gave nine false positives on the
    real corpus.
    """
    findings = [raw_with(r'docker ps --format "\{\{.ID\}\}"',
                         'docker ps --format "{{.ID}}"')]
    report = lint(normalize(findings), extra=check_findings(findings))
    assert "placeholder-leak" not in codes(report)


def test_escaped_json_braces_are_not_a_leak() -> None:
    findings = [raw_with(r"aws x --item '{a\}\}'", "aws x --item '{a}}'")]
    assert check_findings(findings) == []


def test_substituted_example_is_not_a_leak() -> None:
    findings = [raw_with("t {{file}}", "t path/to/file")]
    assert [f for f in check_findings(findings) if f.code == "placeholder-leak"] == []


def test_parse_warnings_are_surfaced_as_info() -> None:
    finding = RawTool(
        binary="t",
        path=(),
        description="d",
        tier=SourceTier.TLDR,
        source="page.md",
        warnings=("unbalanced placeholder braces",),
    )
    found = check_findings([finding])
    assert [f.code for f in found] == ["parse-warning"]
    assert found[0].severity is Severity.INFO


# --------------------------------------------------------------------------
# Report mechanics
# --------------------------------------------------------------------------

def test_findings_are_sorted_most_serious_first() -> None:
    report = lint(catalog_of(
        tool(name="zzz", description=""),
        tool(name="not a key!"),
    ))
    ranks = [f.severity.rank for f in report.findings]
    assert ranks == sorted(ranks)


def test_report_is_deterministic() -> None:
    catalog = catalog_of(tool(name="a", description=""), tool(name="b", platforms=frozenset()))
    assert lint(catalog).findings == lint(catalog).findings


def test_counts_and_by_code() -> None:
    report = lint(catalog_of(tool(name="not a key!", description="")))
    assert report.counts()["error"] >= 1
    assert report.by_code()["bad-identifier"] == 1


def test_finding_str_is_readable() -> None:
    text = str(
        Finding(
            code="c", severity=Severity.ERROR, tool="t", message="m", source="src"
        )
    )
    assert "error" in text and "t" in text and "src" in text


def test_extra_findings_are_included() -> None:
    extra = [Finding(code="x", severity=Severity.WARNING, tool="t", message="m")]
    assert "x" in codes(lint(catalog_of(tool()), extra=extra))


def test_empty_catalog() -> None:
    report = lint(normalize([]))
    assert report.checked == 0
    assert not report


# --------------------------------------------------------------------------
# Against the real fixtures
# --------------------------------------------------------------------------

def test_fixture_corpus_has_no_lint_errors() -> None:
    """The committed fixtures must be a clean baseline.

    If this ever fails, either a real defect has been introduced or the linter
    has grown a false positive -- both worth stopping for.
    """
    findings = list(TldrSource(FIXTURES).harvest())
    report = lint(normalize(findings), extra=check_findings(findings))
    assert report.errors == (), [str(f) for f in report.errors]


@pytest.mark.parametrize("severity", list(Severity))
def test_severity_ranks_are_total(severity: Severity) -> None:
    assert isinstance(severity.rank, int)
