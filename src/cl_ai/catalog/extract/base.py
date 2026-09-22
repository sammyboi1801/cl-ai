"""Stage 2 -- source-tier extractors. Each emits raw findings with provenance.

Source protocol: tier, availability probe, harvest() -> Iterable[RawTool].

Tier ladder (stop at the first hit):
  0 native introspection   pwsh Get-Command, argparse, clap/cobra metadata
  1 machine-readable help  --help=json, doc emitters
  2 completion scripts     bash/zsh/fish -- often carry valid VALUE SETS
  3 man + local HTML docs  e.g. Git's 510 bundled git-doc/*.html
  4 tldr placeholders      argument semantics + examples
  5 --help scraping        opt-in only; requires --exec

Every harvested field carries its tier and a confidence score, so the linter
can say "this enum came from tier 5, do not trust it".

WHY A RAW LAYER AT ALL
----------------------
Extractors could emit `Tool` directly. They deliberately do not. A tier knows
things the canonical schema has nowhere to put -- that an option was written
`{{[-m|--message]}}` rather than as two separate flags, that a page was filed
under `linux/` rather than `common/` -- and that detail is exactly what stage 3
needs in order to merge two tiers sensibly. Collapsing to `Tool` inside each
extractor would throw the evidence away before the merge that needs it, and
leave conflicts unexplainable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# ParamKind lives in ir, not here: it describes the canonical schema, so the
# contract owns it and extractors speak it. Re-exported below because every
# tier needs it.
from cl_ai.ir import Capability, ParamKind, Provenance, SourceTier

__all__ = [
    "DESTRUCTIVE_HINTS",
    "OS_TO_SHELLS",
    "ParamKind",
    "RawExample",
    "RawParam",
    "RawTool",
    "Source",
    "infer_capabilities",
    "shells_for_os",
]


# Spellings for tools whose name is punctuation. Shell builtins and syntax
# (`.`, `:`, `[`, `!`, `|`, `((`) are documented as commands, and they need
# identifiers that are readable and distinct from one another.
_PUNCT_NAMES: dict[str, str] = {
    ".": "dot", "?": "question", "!": "bang", "$": "dollar", "%": "percent",
    "(": "lparen", ")": "rparen", ",": "comma", "[": "lbracket",
    "]": "rbracket", "^": "caret", ":": "colon", ">": "gt", "<": "lt",
    "|": "pipe", "{": "lbrace", "}": "rbrace", "~": "tilde", "@": "at",
    "#": "hash", "&": "amp", "*": "star", "+": "plus", "=": "eq",
    "/": "slash", "\\": "backslash", '"': "quote", "'": "squote",
    ";": "semi", "-": "dash", " ": "space",
}

_IDENT_STRIP = re.compile(r"[^0-9A-Za-z]+")


def _identifier(segment: str) -> str:
    """Turn one name segment into an identifier, never returning empty.

    Falls back to spelling punctuation out rather than dropping it, so a
    punctuation-only name still yields a distinct key. Unknown characters
    become their codepoint, which is ugly but unambiguous -- and far better
    than two different tools silently sharing a catalog entry.
    """
    cleaned = _IDENT_STRIP.sub("_", segment).strip("_")
    if cleaned:
        return cleaned
    spelled = "_".join(
        _PUNCT_NAMES.get(ch, f"u{ord(ch):04x}") for ch in segment
    )
    return spelled or "unnamed"


@dataclass(frozen=True)
class RawParam:
    """One parameter as a single tier saw it, before canonicalisation.

    `short` and `long` are kept apart rather than folded into one `name`
    because tiers disagree about which to present, and the choice is a
    rendering decision made later -- long forms read better in a suggestion a
    human is about to inspect, short forms are what terse docs show.
    """

    name: str
    kind: ParamKind
    #: An IR param type: string | integer | number | boolean | array. Set by
    #: the tier, because only the tier sees the evidence -- tldr knows
    #: `{{100}}` is numeric and `{{file1 file2 ...}}` is repeatable, and that
    #: information is gone by the time normalisation sees a bare name.
    type: str = "string"
    short: str | None = None          # "-m"
    long: str | None = None           # "--message"
    value_name: str | None = None     # "message" from {{[-m|--message]}} {{message}}
    description: str = ""
    repeatable: bool = False
    required: bool = False
    # Every spelling the source offered, e.g. ("-m", "--message") or the
    # subcommand aliases ("add", "install"). A genuine value set rather than an
    # inference, which is why it survives into Param.enum: tiers that supply
    # one are the only trustworthy source of valid values.
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RawParam.name must be non-empty")

    @property
    def display(self) -> str:
        """The form to show a human: long if we have it, else short."""
        return self.long or self.short or self.name


@dataclass(frozen=True)
class RawExample:
    """One worked example: a human description plus the command that does it.

    `template` keeps the placeholders (`git commit {{[-m|--message]}} "{{message}}"`)
    and `literal` has them reduced to their slot names
    (`git commit --message "message"`). Both are retained on purpose: the
    template is what a planner fills in, the literal is what a human reads and
    what we embed for retrieval. Deriving one from the other at use time would
    mean re-running the parser in two places that must agree.
    """

    description: str
    template: str
    literal: str
    slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.template:
            raise ValueError("RawExample.template must be non-empty")


@dataclass(frozen=True)
class RawTool:
    """Everything one tier learned about one leaf command."""

    binary: str
    path: tuple[str, ...]
    description: str
    tier: SourceTier
    source: str
    examples: tuple[RawExample, ...] = ()
    params: tuple[RawParam, ...] = ()
    os_targets: frozenset[str] = frozenset()
    homepage: str | None = None
    # Parse problems that did not justify dropping the whole page. Surfaced by
    # the linter rather than logged and forgotten.
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.binary:
            raise ValueError("RawTool.binary must be non-empty")

    @property
    def qualified_name(self) -> str:
        """`git commit` -> `git_commit`, `pacman --sync` -> `pacman_sync`.

        The retrieval and plan-step key, so it must always be non-empty and
        distinct. Three rules, each for a measured reason:

        * Leading dashes are stripped per segment, so a flag-spelled
          subcommand reads as `pacman_sync` rather than `pacman___sync`.
        * Case is preserved. `pacman -Q` and `pacman -q` are different
          commands, and folding them would merge two unrelated tools.
        * Punctuation is spelled out. The corpus documents 20 builtins whose
          whole name is punctuation (`.`, `|`, `[[`, `((`), and naive
          sanitising turned `.` into the bare identifier `_` -- a key that is
          neither readable nor reliably distinct from the next such tool.
        """
        segments: list[str] = [self.binary]
        for seg in self.path:
            segments.append(seg.lstrip("-") or seg)
        joined = "_".join(_identifier(seg) for seg in segments if seg)
        return joined or "unnamed"

    @property
    def invocation(self) -> str:
        """`git commit` -- how a human writes it."""
        return " ".join((self.binary, *self.path))

    def provenance(self, confidence: float = 1.0) -> Provenance:
        return Provenance(tier=self.tier, source=self.source, confidence=confidence)


@runtime_checkable
class Source(Protocol):
    """A tier. Implementations must be side-effect free unless tier is HELPTEXT.

    `available()` is separate from `harvest()` so the pipeline can report "tier
    3 was skipped because Git's HTML docs are not installed" rather than
    silently producing a thinner catalog -- an absent tier and an empty tier
    look identical in the output but mean very different things.
    """

    tier: SourceTier
    name: str

    def available(self) -> bool:
        """Cheap probe. Must not execute any harvested binary."""
        ...

    def harvest(self, binaries: Iterable[str] | None = None) -> Iterator[RawTool]:
        """Yield findings, optionally restricted to `binaries`.

        Yields rather than returns: the full tldr corpus is ~7k pages, and a
        caller indexing incrementally should not wait for all of them.
        """
        ...


# --------------------------------------------------------------------------
# OS -> shell mapping
# --------------------------------------------------------------------------

# Documentation sources target an OPERATING SYSTEM; `Tool.platforms` records
# the SHELL PROFILES a tool renders to. These axes are orthogonal -- pwsh runs
# on Linux, bash runs on Windows -- so no mapping between them is lossless.
#
# The table below is therefore a deliberate over-approximation: it answers
# "which shells could plausibly invoke this?", not "which shells will?". It
# errs toward including a shell, because wrongly excluding one makes a real
# tool invisible, while wrongly including one costs a candidate that the
# renderer or the availability check rejects later. A missing suggestion is
# the failure a user cannot diagnose; a filtered one is one they never see.
OS_TO_SHELLS: dict[str, frozenset[str]] = {
    "common": frozenset({"bash", "zsh", "fish", "powershell", "pwsh", "cmd"}),
    "linux": frozenset({"bash", "zsh", "fish", "pwsh"}),
    "osx": frozenset({"bash", "zsh", "fish", "pwsh"}),
    "freebsd": frozenset({"bash", "zsh", "fish"}),
    "netbsd": frozenset({"bash", "zsh", "fish"}),
    "openbsd": frozenset({"bash", "zsh", "fish"}),
    "sunos": frozenset({"bash", "zsh", "fish"}),
    "android": frozenset({"bash", "zsh"}),
    "windows": frozenset({"powershell", "pwsh", "cmd"}),
    "dos": frozenset({"cmd"}),
    # Network-appliance CLIs. Not a host shell at all; deliberately empty so
    # they never surface as runnable suggestions.
    "cisco-ios": frozenset(),
}


def shells_for_os(os_targets: Iterable[str]) -> frozenset[str]:
    """Union the shell profiles implied by a set of OS tags.

    Unknown OS tags contribute nothing rather than everything. A tag we do not
    recognise is a tag we cannot reason about, and guessing "all shells" would
    surface appliance or platform-specific commands on machines that cannot
    run them.
    """
    out: set[str] = set()
    for tag in os_targets:
        out |= OS_TO_SHELLS.get(tag.strip().lower(), frozenset())
    return frozenset(out)


# --------------------------------------------------------------------------
# Capability inference
# --------------------------------------------------------------------------

# Tier-independent, so it lives here rather than in any one extractor.
#
# These are HINTS, not a security boundary. They drive the "destructive" marker
# the widget shows, and the cost of a false positive (a needless warning) is
# far below a false negative (an unflagged `rm -rf` one Enter away), so the
# lists lean inclusive.
DESTRUCTIVE_HINTS: frozenset[str] = frozenset({
    "rm", "rmdir", "rmi", "del", "erase", "unlink", "shred", "srm", "wipe",
    "mkfs", "fdisk", "parted", "dd", "format", "diskpart",
    "kill", "killall", "pkill", "taskkill", "shutdown", "reboot", "halt",
    "truncate", "drop", "dropdb", "destroy", "purge", "prune",
})

_WRITE_HINTS = frozenset({
    "write", "create", "add", "set", "update", "install", "copy", "move",
    "rename", "save", "commit", "push", "modify", "edit", "append", "mkdir",
})
# Tools whose entire purpose is to touch the network. Safe to tag on identity.
_NETWORK_HINTS = frozenset({
    "curl", "wget", "ssh", "scp", "rsync", "ftp", "sftp", "nc", "ncat",
    "ping", "dig", "nslookup", "telnet", "http", "httpie", "traceroute",
})

# Multi-purpose drivers where the SUBCOMMAND decides, not the binary. Tagging
# these on identity alone is wrong in the common case: `git commit` and
# `docker ps` are entirely local, and marking them network-dependent would
# make the planner avoid them when offline -- suppressing the most ordinary
# commands there are.
_NETWORK_DRIVERS = frozenset({
    "git", "npm", "pnpm", "yarn", "pip", "pip3", "gem", "cargo", "go",
    "apt", "apt-get", "dnf", "yum", "pacman", "brew", "choco", "winget",
    "docker", "podman", "kubectl", "helm", "aws", "gcloud", "az", "gh",
})
_NETWORK_VERBS = frozenset({
    "push", "pull", "fetch", "clone", "install", "uninstall", "update",
    "upgrade", "add", "remove", "search", "publish", "login", "logout",
    "download", "sync", "deploy", "apply", "get", "list", "send", "copy",
    "cp", "upload", "refresh", "remote",
})
_ELEVATED_HINTS = frozenset({"sudo", "doas", "runas", "pkexec", "su"})

# Word-boundary match so `format` does not fire on `formatter` and `rm` does
# not fire on `rmarkdown` -- substring matching here produces warnings on
# harmless commands, which trains users to ignore the warning that matters.
_WORD = re.compile(r"[a-z0-9.]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def infer_capabilities(
    binary: str,
    path: Sequence[str] = (),
    examples: Iterable[str] = (),
    description: str = "",
) -> frozenset[Capability]:
    """Best-effort behavioural tags for a tool.

    Reads the tool's own identity first and its examples second. Identity is
    the stronger signal: `rm` is destructive regardless of how an example
    happens to be phrased, whereas an example may merely mention a word.
    """
    caps: set[Capability] = set()
    identity = _words(" ".join((binary, *path)))
    example_words: set[str] = set()
    for ex in examples:
        example_words |= _words(ex)
    prose = _words(description)

    # DESTRUCTIVE is judged on IDENTITY and prose only, never on examples.
    #
    # A bare binary's examples span its entire surface, so one destructive
    # subcommand poisons the parent: `docker` was flagged because its examples
    # include `docker rm`, and `ps` because its examples pipe into `kill`.
    # Observed live -- "list running containers" returned `docker` and `ps`
    # both carrying a destructive marker, on a query that destroys nothing.
    #
    # That is the failure this inference is supposed to avoid. A marker shown
    # on harmless commands is worse than no marker at all, because it teaches
    # the user to dismiss the one that matters. Identity still catches the
    # cases that count: `rm`, `shred`, `git push --force`, `docker rmi`.
    if identity & DESTRUCTIVE_HINTS or (prose & DESTRUCTIVE_HINTS):
        caps.add(Capability.DESTRUCTIVE)
        caps.add(Capability.WRITES)
    if identity & _WRITE_HINTS or (prose & _WRITE_HINTS):
        caps.add(Capability.WRITES)
    if identity & _NETWORK_HINTS or example_words & _NETWORK_HINTS:
        caps.add(Capability.NEEDS_NETWORK)
    elif binary.lower() in _NETWORK_DRIVERS and (_words(" ".join(path)) & _NETWORK_VERBS):
        # `git push` yes, `git commit` no. Judged on the subcommand, and only
        # from the tool's own identity -- an example that merely mentions
        # "install" in its prose is not evidence the command needs a network.
        caps.add(Capability.NEEDS_NETWORK)
    if identity & _ELEVATED_HINTS or example_words & _ELEVATED_HINTS:
        caps.add(Capability.ELEVATED)

    # Everything reads something. Stated explicitly so an empty capability set
    # unambiguously means "we never analysed this tool", not "this tool is
    # inert" -- the same distinction Tool.schematized draws.
    caps.add(Capability.READS)
    return frozenset(caps)
