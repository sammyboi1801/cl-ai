"""Stage 3 -- raw findings -> canonical schemas.

Merge rule: HIGHEST TIER WINS. Losing values are not discarded; they are kept
in provenance with a conflict flag. Silent merging is how catalogs rot.

VARIANTS, AND WHY THEY ARE NOT OPTIONAL
---------------------------------------
The obvious design is one Tool per name. It is wrong, and the corpus says so
plainly: `dir` is documented three times -- as a Cisco IOS command, as the
Windows shell builtin, and as a GNU coreutil. They share nothing but a name.
Merging them produces a single entry whose examples are drawn from a router, a
cmd prompt and a Linux box at once, and a user asking to list files gets a
confidently wrong command. Same story for BSD versus GNU `ls`.

So identity here is (name, platform family), and a name maps to a TUPLE of
variants. Selection happens at retrieval, where the shell is actually known --
not at merge time, where it is not.

The compatibility test is the OS FAMILY, deliberately not the shell set.
Shell sets cannot be used: `OS_TO_SHELLS` intentionally over-approximates, so
pwsh appears under both linux and windows, and any intersection test would
happily merge Windows `dir` with GNU `dir` through that single shared shell.
The family axis is the one that actually distinguishes them.

`common` stays its own variant rather than being folded into each concrete
family. Folding would mean inventing a merged tool no source ever described;
keeping it separate means a Linux user may see both a `common` and a `posix`
`ls`, which is honest -- two sources really did describe it differently -- and
`select()` resolves it by preferring the more specific family.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Generic, TypeVar

from cl_ai.ir import (
    Capability,
    Param,
    ParamKind,
    Provenance,
    SourceTier,
    Tool,
)

from .extract.base import RawParam, RawTool, infer_capabilities, shells_for_os

__all__ = [
    "Catalog",
    "Family",
    "MergeConflict",
    "Variant",
    "family_for_os_name",
    "family_of",
    "normalize",
]

T = TypeVar("T")


# --------------------------------------------------------------------------
# Platform families
# --------------------------------------------------------------------------

class Family(str, Enum):
    """A group of operating systems whose commands are genuinely the same.

    Coarse on purpose. The question this answers is not "which OS is this?"
    but "would merging two findings across this boundary invent a tool that
    no source described?".
    """

    POSIX = "posix"
    WINDOWS = "windows"
    APPLIANCE = "appliance"
    COMMON = "common"
    UNKNOWN = "unknown"

    @property
    def is_concrete(self) -> bool:
        """Whether this family names real platforms rather than "everywhere"."""
        return self in (Family.POSIX, Family.WINDOWS, Family.APPLIANCE)


_FAMILY_OF_OS: Mapping[str, Family] = {
    "common": Family.COMMON,
    "linux": Family.POSIX,
    "osx": Family.POSIX,
    "freebsd": Family.POSIX,
    "netbsd": Family.POSIX,
    "openbsd": Family.POSIX,
    "sunos": Family.POSIX,
    "android": Family.POSIX,
    "windows": Family.WINDOWS,
    "dos": Family.WINDOWS,
    "cisco-ios": Family.APPLIANCE,
}


#: Runtime OS names -> family. Covers both what sys.platform reports and what
#: documentation sources call the same platform, because callers legitimately
#: have either: ContextFacts.os comes from the running process, while a tldr
#: page is filed under the project's own naming.
_FAMILY_OF_RUNTIME_OS: Mapping[str, Family] = {
    "win32": Family.WINDOWS,
    "windows": Family.WINDOWS,
    "nt": Family.WINDOWS,
    "cygwin": Family.WINDOWS,
    "dos": Family.WINDOWS,
    "darwin": Family.POSIX,
    "macos": Family.POSIX,
    "osx": Family.POSIX,
    "linux": Family.POSIX,
    "linux2": Family.POSIX,
    "freebsd": Family.POSIX,
    "openbsd": Family.POSIX,
    "netbsd": Family.POSIX,
    "sunos": Family.POSIX,
    "solaris": Family.POSIX,
    "android": Family.POSIX,
    "posix": Family.POSIX,
}


def family_for_os_name(os_name: str) -> Family:
    """Map a runtime OS name to a family, UNKNOWN when unrecognised.

    UNKNOWN is treated as "do not filter" by select(): an OS we cannot place
    should widen the candidate set, never silently empty it. Returning nothing
    because we failed to recognise the platform would look identical to the
    tool not existing.
    """
    name = os_name.strip().lower()
    if name in _FAMILY_OF_RUNTIME_OS:
        return _FAMILY_OF_RUNTIME_OS[name]
    # sys.platform values carry version suffixes, e.g. "freebsd13".
    for known, family in _FAMILY_OF_RUNTIME_OS.items():
        if name.startswith(known):
            return family
    return Family.UNKNOWN


def family_of(os_targets: Iterable[str]) -> Family:
    """The family a finding belongs to.

    A finding tagged with several families is UNKNOWN rather than being
    assigned to one of them: that combination means the extractor produced
    something we cannot place, and quietly picking a family would hide it.
    """
    families = {
        _FAMILY_OF_OS.get(tag.strip().lower(), Family.UNKNOWN) for tag in os_targets
    }
    if not families:
        return Family.UNKNOWN
    if len(families) == 1:
        return next(iter(families))
    concrete = {f for f in families if f.is_concrete}
    if len(concrete) == 1:
        return next(iter(concrete))
    return Family.UNKNOWN


# --------------------------------------------------------------------------
# Conflict records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MergeConflict:
    """One value that lost a merge. Kept so a bad schema stays debuggable.

    Recorded even when the loser came from the same tier as the winner,
    because a same-tier disagreement is the more interesting case: it means
    two sources of equal standing contradict each other and a human should
    look.
    """

    tool: str
    field: str
    kept: str
    kept_source: str
    dropped: str
    dropped_source: str
    same_tier: bool = False


@dataclass(frozen=True)
class _Claim(Generic[T]):
    """A value one source asserts, with the standing to judge it by."""

    value: T
    tier: SourceTier
    source: str

    @property
    def rank(self) -> tuple[int, str]:
        # Tier first (lower is better), then source name. The source name
        # tie-break exists solely to make the result deterministic: without it
        # two equal-tier claims would resolve by dict ordering, and the catalog
        # would differ between runs on the same inputs.
        return (int(self.tier), self.source)


def _pick(
    claims: Sequence[_Claim[T]],
    *,
    tool: str,
    field_name: str,
    conflicts: list[MergeConflict],
) -> tuple[T, Provenance] | None:
    """Choose one claim, recording every claim it beat."""
    usable = [c for c in claims if c.value not in (None, "", (), frozenset())]
    if not usable:
        return None
    ordered = sorted(usable, key=lambda c: c.rank)
    winner = ordered[0]
    dropped: list[str] = []
    for loser in ordered[1:]:
        if loser.value == winner.value:
            continue
        dropped.append(str(loser.value))
        conflicts.append(
            MergeConflict(
                tool=tool,
                field=field_name,
                kept=str(winner.value),
                kept_source=winner.source,
                dropped=str(loser.value),
                dropped_source=loser.source,
                same_tier=loser.tier == winner.tier,
            )
        )
    provenance = Provenance(
        tier=winner.tier,
        source=winner.source,
        conflicts=tuple(dropped),
    )
    return winner.value, provenance


# --------------------------------------------------------------------------
# Parameter merging
# --------------------------------------------------------------------------

def _param_key(param: RawParam) -> tuple[str, str]:
    """Identity of a parameter across sources.

    Options are keyed by their long spelling when there is one, because that is
    the spelling sources agree on; a short form alone is ambiguous between
    tools. Positionals have only their name.
    """
    if param.kind is ParamKind.OPTION:
        return (param.kind.value, param.long or param.short or param.name)
    return (param.kind.value, param.name)


def _merge_params(
    findings: Sequence[RawTool], *, tool: str, conflicts: list[MergeConflict]
) -> tuple[Param, ...]:
    grouped: dict[tuple[str, str], list[_Claim[RawParam]]] = defaultdict(list)
    for finding in findings:
        for param in finding.params:
            grouped[_param_key(param)].append(
                _Claim(value=param, tier=finding.tier, source=finding.source)
            )

    out: list[Param] = []
    for key, claims in sorted(grouped.items()):
        chosen = _pick(
            claims, tool=tool, field_name=f"param:{key[1]}", conflicts=conflicts
        )
        if chosen is None:
            continue
        raw, provenance = chosen

        # Only a SUBCOMMAND's alternatives are values. For an option they are
        # spellings of the flag itself, and copying them into `enum` would
        # invite a planner to pass "-m" as the value of --message.
        enum_values = raw.choices if raw.kind is ParamKind.SUBCOMMAND else ()

        flag = raw.long or raw.short if raw.kind is ParamKind.OPTION else None
        if raw.kind is ParamKind.OPTION and not flag:
            # Unrenderable; Param would reject it. Drop with a record rather
            # than raising: one bad parameter must not sink a whole catalog.
            conflicts.append(
                MergeConflict(
                    tool=tool,
                    field=f"param:{key[1]}",
                    kept="<dropped>",
                    kept_source="normalize",
                    dropped=raw.name,
                    dropped_source=provenance.source,
                )
            )
            continue

        out.append(
            Param(
                name=raw.name,
                type=raw.type,
                kind=raw.kind,
                flag=flag,
                description=raw.description,
                enum=enum_values,
                short=raw.short if raw.kind is ParamKind.OPTION else None,
                required=raw.required,
                repeatable=raw.repeatable,
                provenance=provenance,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """One Tool plus the family it applies to."""

    family: Family
    tool: Tool


@dataclass(frozen=True)
class Catalog:
    """The merged result. Deterministic, so two builds diff cleanly."""

    variants: tuple[Variant, ...] = ()
    conflicts: tuple[MergeConflict, ...] = field(default=(), compare=False)
    #: Findings that produced no usable tool, with the reason.
    rejected: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    @cached_property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(v.tool for v in self.variants)

    @cached_property
    def by_name(self) -> Mapping[str, tuple[Variant, ...]]:
        grouped: dict[str, list[Variant]] = defaultdict(list)
        for variant in self.variants:
            grouped[variant.tool.name].append(variant)
        return {name: tuple(v) for name, v in grouped.items()}

    def __len__(self) -> int:
        return len(self.variants)

    def select(
        self, name: str, shell: str | None = None, os_name: str | None = None
    ) -> Tool | None:
        """The best variant of `name` for a given (os, shell) target.

        BOTH axes are needed, and this was a real bug when only `shell` was:
        `dir` exists as a Windows builtin and as a GNU coreutil, pwsh runs on
        Windows and Linux alike, so a pwsh user matched both variants and the
        tie broke alphabetically -- handing a Windows user the GNU page. The
        shell says how to quote; the OS says which tool actually exists. Only
        together do they identify a target.

        Specificity wins: a variant whose family names real platforms beats a
        `common` one, because a platform-specific page was written precisely
        because the behaviour differs there. With neither axis known there is
        nothing to be specific about, so `common` wins -- it is the variant
        that is true everywhere.
        """
        variants = self.by_name.get(name)
        if not variants:
            return None

        candidates = list(variants)
        if shell is not None:
            candidates = [v for v in candidates if shell in v.tool.platforms]
            if not candidates:
                return None

        wanted = family_for_os_name(os_name) if os_name else None
        if wanted is not None and wanted is not Family.UNKNOWN:
            # A variant from a different concrete family is not a worse match,
            # it is the wrong tool: GNU `dir` on Windows does not exist. Drop
            # those outright rather than ranking them below.
            candidates = [
                v
                for v in candidates
                if v.family is wanted or not v.family.is_concrete
            ]
            if not candidates:
                return None

        candidates.sort(key=lambda v: (not v.family.is_concrete, v.family.value))
        if shell is None and os_name is None:
            for variant in candidates:
                if variant.family is Family.COMMON:
                    return variant.tool
        return candidates[0].tool

    def for_target(
        self, shell: str, os_name: str | None = None
    ) -> tuple[Tool, ...]:
        """Every tool that could render into this (os, shell), one per name."""
        names = sorted({v.tool.name for v in self.variants if shell in v.tool.platforms})
        out = []
        for name in names:
            tool = self.select(name, shell, os_name)
            if tool is not None:
                out.append(tool)
        return tuple(out)


def normalize(findings: Iterable[RawTool]) -> Catalog:
    """Merge raw findings into a catalog of canonical tools."""
    grouped: dict[tuple[str, Family], list[RawTool]] = defaultdict(list)
    rejected: list[tuple[str, str]] = []

    for finding in findings:
        name = finding.qualified_name
        if not name:
            rejected.append((finding.source, "no usable identifier"))
            continue
        grouped[(name, family_of(finding.os_targets))].append(finding)

    conflicts: list[MergeConflict] = []
    variants: list[Variant] = []

    for (name, family), group in sorted(grouped.items(), key=lambda kv: kv[0]):
        # Deterministic order inside the group, so tie-breaks are stable.
        group = sorted(group, key=lambda f: (int(f.tier), f.source))

        description = _pick(
            [_Claim(f.description, f.tier, f.source) for f in group],
            tool=name,
            field_name="description",
            conflicts=conflicts,
        )
        homepage = _pick(
            [_Claim(f.homepage or "", f.tier, f.source) for f in group],
            tool=name,
            field_name="homepage",
            conflicts=conflicts,
        )

        # Examples are additive, not competing claims: two sources showing
        # different ways to use a tool are both right, and more worked examples
        # is exactly what the planner wants. Deduplicated by template so the
        # same page counted twice does not inflate the list.
        seen_templates: set[str] = set()
        examples: list[str] = []
        for finding in group:
            for example in finding.examples:
                if example.literal in seen_templates:
                    continue
                seen_templates.add(example.literal)
                examples.append(example.literal)

        os_targets: set[str] = set()
        for finding in group:
            os_targets |= finding.os_targets

        # Capabilities are unioned, never picked by tier. They are inferred
        # rather than sourced, and they gate a destructive-command warning: if
        # any source's evidence suggests a tool deletes things, that has to
        # survive the merge. Failing safe here costs a needless warning;
        # failing the other way costs an unflagged `rm -rf`.
        capabilities: set[Capability] = set()
        for finding in group:
            capabilities |= infer_capabilities(
                finding.binary,
                finding.path,
                examples=[e.literal for e in finding.examples],
                description=finding.description,
            )

        params = _merge_params(group, tool=name, conflicts=conflicts)
        best = group[0]

        tool = Tool(
            name=name,
            description=description[0] if description else "",
            binary=best.binary,
            path=best.path,
            params=params,
            capabilities=frozenset(capabilities),
            platforms=shells_for_os(os_targets),
            examples=tuple(examples),
            homepage=homepage[0] if homepage else None,
            provenance=description[1]
            if description
            else Provenance(tier=best.tier, source=best.source),
        )
        variants.append(Variant(family=family, tool=tool))

    return Catalog(
        variants=tuple(variants),
        conflicts=tuple(conflicts),
        rejected=tuple(rejected),
    )
