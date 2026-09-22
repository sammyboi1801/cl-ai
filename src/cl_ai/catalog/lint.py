"""Stage 4 -- quality gates over a built catalog.

The linter exists because of what a bad catalog entry does. A missing tool is
invisible and harmless: Tab behaves like ordinary Tab. A *wrong* tool is
actively dangerous -- it puts a plausible, incorrect command in the prompt
buffer with Enter one keystroke away. So the failure this checks for is not
"incomplete", it is "confidently wrong".

Everything here is a report, never an exception. A catalog with problems is
still far more useful than no catalog, so the build proceeds and the findings
are surfaced. The caller decides what is fatal; `worst_severity` and
`counts()` exist so that decision can be made without re-deriving it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from cl_ai.ir import Capability, ParamKind, SourceTier, Tool

from .extract.base import RawTool
from .normalize import Catalog, Family

__all__ = [
    "Finding",
    "LintReport",
    "Severity",
    "check_findings",
    "lint",
]


class Severity(str, Enum):
    """How much a finding should worry the reader.

    ERROR is reserved for entries that could produce a wrong command, not
    merely a poor one. Keeping that bar high is what stops the report becoming
    noise that nobody reads -- the same reason the destructive marker is not
    attached to everything.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @property
    def rank(self) -> int:
        return {"error": 0, "warning": 1, "info": 2}[self.value]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    tool: str
    message: str
    source: str = ""

    def __str__(self) -> str:
        where = f" ({self.source})" if self.source else ""
        return f"[{self.severity.value}] {self.code} {self.tool}: {self.message}{where}"


@dataclass(frozen=True)
class LintReport:
    findings: tuple[Finding, ...] = ()
    checked: int = 0

    def counts(self) -> dict[str, int]:
        return dict(Counter(f.severity.value for f in self.findings))

    def by_code(self) -> dict[str, int]:
        return dict(Counter(f.code for f in self.findings))

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def worst_severity(self) -> Severity | None:
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key=lambda s: s.rank)

    def __bool__(self) -> bool:
        """True when there is anything to report."""
        return bool(self.findings)


# A shell metacharacter inside a tool's own name means the "tool" is really
# shell syntax. `|`, `((` and friends are the renderer's job -- it owns
# pipelines and grouping -- and offering them as invocable tools invites a
# planner to emit a pipe as though it were a command.
_SYNTAX_NAMES = frozenset(
    {
        "pipe", "lparen_lparen", "rparen_rparen", "lbracket_lbracket",
        "rbracket_rbracket", "lbrace", "rbrace", "gt", "lt", "amp",
        "semi", "caret", "dollar", "eq",
    }
)

_PLACEHOLDER_LEAK = re.compile(r"\{\{|\}\}")
_ESCAPED_BRACE = re.compile(r"\\[{}]")
_IDENT_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")


def check_findings(findings: Iterable[RawTool]) -> list[Finding]:
    """Checks that require the RAW findings, before the merge discards evidence.

    Placeholder leaks are the reason this exists. An unsubstituted `{{slot}}`
    reaching a rendered command is the single worst catalog defect -- it puts
    visibly broken text in the user's prompt buffer -- so it deserves an ERROR.
    But it cannot be detected from a finished Tool.

    The real corpus makes that concrete. `docker image ls --format "{{.ID}}"`,
    `fakedata "{{Generator}}"` and Clojure's `-Sdeps '{...}}'` all contain
    brace pairs in the *correct* output, because the braces belong to the
    command rather than to tldr. `{{Generator}}` is shape-identical to a leaked
    slot, so no amount of pattern matching separates them. The only thing that
    does is the template: if the source escaped its braces, the output is meant
    to have them.

    Checking the template against its own literal is therefore the only sound
    test, and the template exists only here. Running this check on a merged
    catalog produced nine false positives, every one of them a working command.
    """
    out: list[Finding] = []
    for finding in findings:
        for example in finding.examples:
            if _ESCAPED_BRACE.search(example.template):
                continue    # braces the source deliberately kept
            if _PLACEHOLDER_LEAK.search(example.literal):
                out.append(
                    Finding(
                        code="placeholder-leak",
                        severity=Severity.ERROR,
                        tool=finding.qualified_name,
                        message=(
                            "example kept an unsubstituted slot: "
                            f"{example.literal[:60]!r}"
                        ),
                        source=finding.source,
                    )
                )
                break
        for warning in finding.warnings:
            out.append(
                Finding(
                    code="parse-warning",
                    severity=Severity.INFO,
                    tool=finding.qualified_name,
                    message=warning,
                    source=finding.source,
                )
            )
    return out


def _check_tool(tool: Tool, family: Family) -> list[Finding]:
    source = tool.provenance.source if tool.provenance else ""
    out: list[Finding] = []

    def add(code: str, severity: Severity, message: str) -> None:
        out.append(
            Finding(
                code=code,
                severity=severity,
                tool=tool.name,
                message=message,
                source=source,
            )
        )

    # -- identity ---------------------------------------------------------
    if not _IDENT_OK.match(tool.name):
        add(
            "bad-identifier",
            Severity.ERROR,
            f"name {tool.name!r} is not a usable plan-step key",
        )
    if not tool.binary:
        add("no-binary", Severity.ERROR, "no binary to invoke")

    # Placeholder leaks are NOT checked here. See check_findings(): a merged
    # Tool has lost the evidence needed to judge them, and guessing produced
    # nine false positives on the real corpus.

    for param in tool.params:
        if param.kind is ParamKind.OPTION and not (param.flag or param.short):
            add(
                "unrenderable-option",
                Severity.ERROR,
                f"option {param.name!r} has no spelling to emit",
            )
        if param.kind is ParamKind.OPTION and param.flag and param.enum:
            # enum on an option means someone put flag spellings where values
            # belong; a planner would pass "-m" as the value of --message.
            add(
                "spellings-as-values",
                Severity.WARNING,
                f"option {param.name!r} carries enum values {param.enum}",
            )
        if not _IDENT_OK.match(param.name):
            add(
                "bad-param-name",
                Severity.WARNING,
                f"parameter {param.name!r} is not a usable argument key",
            )

    # -- reachability -----------------------------------------------------
    if not tool.platforms and family is not Family.APPLIANCE:
        add(
            "unreachable",
            Severity.WARNING,
            "no shell can invoke this tool",
        )
    if tool.name in _SYNTAX_NAMES:
        add(
            "shell-syntax-as-tool",
            Severity.WARNING,
            "this is shell syntax the renderer owns, not an invocable tool",
        )

    # -- completeness -----------------------------------------------------
    if not tool.schematized:
        add(
            "unschematized",
            Severity.INFO,
            "known to exist but has neither parameters nor examples",
        )
    if not tool.description:
        add("no-description", Severity.INFO, "no description to retrieve on")

    # -- trust ------------------------------------------------------------
    if tool.provenance and tool.provenance.tier >= SourceTier.HELPTEXT:
        add(
            "low-tier-schema",
            Severity.INFO,
            f"schema came from tier {int(tool.provenance.tier)}; treat as unverified",
        )
    for param in tool.params:
        prov = param.provenance
        if prov and prov.enum_untrustworthy(param.enum):
            add(
                "low-tier-enum",
                Severity.INFO,
                f"value set for {param.name!r} came from tier {int(prov.tier)}",
            )

    # -- safety -----------------------------------------------------------
    if Capability.DESTRUCTIVE in tool.capabilities and not tool.description:
        # The widget shows a destructive marker; with no description the user
        # has nothing to judge the command by before pressing Enter.
        add(
            "unexplained-destructive",
            Severity.WARNING,
            "flagged destructive but has no description to show the user",
        )

    return out


def lint(catalog: Catalog, *, extra: Iterable[Finding] = ()) -> LintReport:
    """Check every variant in a catalog.

    Findings are sorted by severity then tool, so the most serious problems
    read first and two runs over the same catalog produce identical output.
    """
    findings: list[Finding] = list(extra)
    for variant in catalog.variants:
        findings.extend(_check_tool(variant.tool, variant.family))

    # Same name in the same family twice would make select() nondeterministic.
    seen: Counter[tuple[str, Family]] = Counter(
        (v.tool.name, v.family) for v in catalog.variants
    )
    for (name, family), count in sorted(seen.items(), key=lambda kv: kv[0]):
        if count > 1:
            findings.append(
                Finding(
                    code="duplicate-variant",
                    severity=Severity.ERROR,
                    tool=name,
                    message=f"{count} variants share family {family.value}",
                )
            )

    for source, reason in catalog.rejected:
        findings.append(
            Finding(
                code="rejected-finding",
                severity=Severity.WARNING,
                tool="-",
                message=reason,
                source=source,
            )
        )

    findings.sort(key=lambda f: (f.severity.rank, f.tool, f.code))
    return LintReport(findings=tuple(findings), checked=len(catalog.variants))
