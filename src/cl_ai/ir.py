"""The intermediate representation. The only currency the core speaks.

This is the load-bearing decision of the whole system: every other component
depends on it, and it is what makes the LLM swappable. The model never emits
shell text -- it emits a Plan, and renderers turn that into syntax. That single
choice solves pipelines and cross-platform support together, and it keeps the
model boundary narrow enough to swap.

Nothing here imports a backend, a shell, or a model. Pure data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

# --------------------------------------------------------------------------
# Tool schemas (canonical dialect)
# --------------------------------------------------------------------------

class SourceTier(int, Enum):
    """Where a harvested fact came from. Lower is more trustworthy."""
    NATIVE = 0        # pwsh Get-Command, argparse, clap/cobra metadata
    MACHINE_HELP = 1  # --help=json, doc emitters
    COMPLETIONS = 2   # bash/zsh/fish completion scripts
    MANDOCS = 3       # man pages, bundled HTML manuals
    TLDR = 4          # tldr placeholders
    HELPTEXT = 5      # --help scraping (opt-in)
    HANDWRITTEN = 6   # a human wrote it


@dataclass(frozen=True)
class Provenance:
    """Why we believe a field. Emitted per field, not per tool.

    `conflicts` records values proposed by lower-priority tiers and overridden
    under highest-tier-wins. Keeping them is what makes a bad schema debuggable
    six months later; discarding them is how catalogs rot.
    """
    tier: SourceTier
    source: str                       # e.g. "git-doc/git-commit.html#OPTIONS"
    confidence: float = 1.0
    conflicts: tuple[str, ...] = ()

    def enum_untrustworthy(self, enum: tuple[str, ...]) -> bool:
        """Whether a value set from this source should be treated as a guess.

        Completion scripts and native introspection state valid values
        outright, so an enum from tier 0-2 is a fact. Prose sources -- man
        pages, tldr, scraped help -- only ever show *some* values in passing,
        so treating those as the complete set would make a planner reject
        arguments that are perfectly valid.
        """
        return bool(enum) and self.tier >= SourceTier.MANDOCS


class ParamKind(str, Enum):
    """How a parameter is written at the call site.

    Load-bearing for renderers: an OPTION carries its own name into the
    command, a POSITIONAL is placed by ordinal, and a SUBCOMMAND is part of
    the tool's identity rather than its arguments. Conflating them is how a
    renderer emits `git --commit message` instead of `git commit -m message`.
    """

    OPTION = "option"
    POSITIONAL = "positional"
    SUBCOMMAND = "subcommand"


@dataclass(frozen=True)
class Param:
    name: str
    type: str                          # string | integer | number | boolean | array
    kind: ParamKind = ParamKind.POSITIONAL
    #: The exact token a renderer must emit, e.g. "--gpg-sign" or "/list".
    #:
    #: Stored rather than derived, because it cannot be derived. `name` is a
    #: slug -- `--gpg-sign` becomes `gpg_sign` -- and reconstructing the flag
    #: from the slug has to guess whether each underscore was a hyphen. Guess
    #: wrong and the rendered command carries a flag the tool will reject,
    #: which is the confidently-wrong-command failure this whole design exists
    #: to avoid. None for positionals, which have no token of their own.
    flag: str | None = None
    description: str = ""
    #: Valid VALUES for this parameter -- never alternative spellings of its
    #: flag. `-m` and `--message` are two ways to write one option and belong
    #: in `flag`/`short`; `{{[add|install]}}` really is a value set. Mixing the
    #: two would let a planner pass "-m" as the value of --message.
    enum: tuple[str, ...] = ()
    #: The short spelling, when a source gave both. Presentation only: renderers
    #: prefer `flag` because a suggestion a human is about to read and run is
    #: clearer as `--message` than `-m`.
    short: str | None = None
    required: bool = False
    repeatable: bool = False
    default: Any = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Param.name must be non-empty")
        if self.kind is ParamKind.OPTION and not (self.flag or self.short):
            # An option with no spelling cannot be rendered at all. Catching it
            # here turns a silent no-op in the renderer into a build-time error
            # in the catalog, where there is a source path to point at.
            raise ValueError(
                f"option parameter {self.name!r} has neither flag nor short form"
            )


class Capability(str, Enum):
    """Behavioural tags. Drive confirmation prompts and platform filtering."""
    READS = "reads"
    WRITES = "writes"
    DESTRUCTIVE = "destructive"
    NEEDS_NETWORK = "needs_network"
    ELEVATED = "elevated"


@dataclass(frozen=True)
class Example:
    """A worked example: what someone wanted, and the command that did it.

    The description is not decoration. It is the only text in the whole
    catalog that states an INTENT in the words a user would actually type --
    "commit staged files with the specified message" against `git commit -m`.
    A tool's own description says what the tool is; an example description
    says what you would want it for, which is what a natural-language query
    is made of.

    Keeping only the command text, as an earlier version did, measurably
    crippled retrieval: queries phrased as intent had nothing to match except
    tool names, so `copy a file` found the tool literally named `file`.
    """

    description: str
    command: str

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("Example.command must be non-empty")

    def __str__(self) -> str:
        return self.command


@dataclass(frozen=True)
class Tool:
    """One leaf command, e.g. `aws_s3_cp`.

    Leaves are fine-grained on purpose. Tool count costs the model nothing --
    retrieval only ever declares the top few -- so granularity is a retrieval
    concern, not a prompt-budget one.
    """
    name: str
    description: str
    binary: str                        # "aws"      -- stage-1 retrieval key
    path: tuple[str, ...] = ()         # ("s3","cp") -- subcommand path
    params: tuple[Param, ...] = ()
    capabilities: frozenset[Capability] = frozenset()
    platforms: frozenset[str] = frozenset()   # shell profile ids it renders to
    examples: tuple[Example, ...] = ()
    #: Upstream documentation, when a source supplied it. Not used to render
    #: anything -- it is what lets the UI answer "where did this come from?",
    #: which is the honest response when a user does not trust a suggestion.
    homepage: str | None = None
    provenance: Provenance | None = None

    @property
    def invocation(self) -> str:
        """How the command is actually typed: `git commit`, not `git_commit`.

        `name` is an identifier and cannot be reversed into this -- a
        subcommand may legitimately contain an underscore, so splitting on one
        corrupts it. Derived here because retrieval and suggestion both need
        it and had each rebuilt it inline.
        """
        return " ".join((self.binary, *self.path))

    @property
    def schematized(self) -> bool:
        """False for a tool we know exists but cannot fill in.

        This state is a feature. It lets the system say "git clone exists, I
        just do not have arguments for it" instead of substituting a neighbour
        -- the failure that turns a missing catalog entry into a confidently
        wrong command.
        """
        return bool(self.params) or bool(self.examples)


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

class StreamKind(str, Enum):
    """What flows between pipeline stages.

    POSIX pipes text; PowerShell pipes objects. These are not syntactic
    variants -- a text-oriented plan rendered naively into PowerShell runs and
    returns wrong results, which is worse than an error. Renderers use this to
    pick the idiomatic form, or to refuse when no faithful translation exists.
    """
    NONE = "none"
    LINES = "lines"
    PATHS = "paths"
    OBJECTS = "objects"


class Join(str, Enum):
    PIPE = "pipe"        # ls | grep
    AND = "and"          # git add && git commit
    THEN = "then"        # a ; b


@dataclass(frozen=True)
class Step:
    """One command in a plan.

    `arguments` is a read-only mapping and is excluded from __hash__ while
    remaining part of __eq__. Both halves matter: frozen=True protects the
    field binding but not a dict behind it, so a plain dict would let callers
    mutate a "frozen" Step; and a dict field would make Step unhashable, which
    breaks deduplicating candidates -- at runtime, in the assembler, rather
    than here. Equal Steps still hash equal; unequal ones may collide, which is
    permitted and cheap at these sizes.
    """

    tool: str
    arguments: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )
    emits: StreamKind = StreamKind.NONE
    join_to_next: Join | None = None

    def __post_init__(self) -> None:
        # Callers pass ordinary dicts; wrap so the promise of frozen holds.
        if not isinstance(self.arguments, MappingProxyType):
            object.__setattr__(
                self, "arguments", MappingProxyType(dict(self.arguments))
            )


@dataclass(frozen=True)
class Plan:
    """One candidate suggestion. A ranked list of these is what the user cycles.

    `confidence` and `reasoning` are DEBUG-ONLY and must not drive control flow.
    Observed twice with Needle 3: the reasoning string describes a different
    tool than the one actually called, and a wrong call reported 0.99.
    """
    steps: tuple[Step, ...] = ()
    refused: bool = False
    confidence: float | None = None
    reasoning: str | None = None
    #: The backend's untouched response, for debugging only. Excluded from
    #: hashing (it is a dict) and from equality (two plans that differ only in
    #: provider noise are the same suggestion to a user cycling the list).
    raw: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def is_empty(self) -> bool:
        return self.refused or not self.steps


@dataclass(frozen=True)
class ContextFacts:
    """What is true about this terminal, right now."""
    os: str = ""
    shell: str = ""
    cwd: str = ""
    in_git_repo: bool = False
    markers: tuple[str, ...] = ()          # package.json, Dockerfile, go.mod...
    installed: frozenset[str] = frozenset()  # hard gate, not a score
    recent_commands: tuple[str, ...] = ()    # local only, never transmitted


@dataclass(frozen=True)
class PlanRequest:
    query: str
    tools: tuple[Tool, ...]
    context: ContextFacts = field(default_factory=ContextFacts)
    max_steps: int = 3
    deadline_ms: int = 120


@dataclass(frozen=True)
class Capabilities:
    """What a backend can do. Core programs against the WEAKEST backend and
    treats these as opportunistic upgrades -- never as assumptions.
    """
    grammar_constrained: bool = False   # can arguments be trusted unvalidated?
    confidence: bool = False            # is the score meaningful?
    multi_call: bool = False            # can it emit multi-stage plans?
    embeddings: bool = False
    max_tools: int = 5
    stateful: bool = False
