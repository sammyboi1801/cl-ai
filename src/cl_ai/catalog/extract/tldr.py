"""Tier 4. tldr placeholders -- `{{branch_name}}` names the argument semantically
and marks the span as a slot.

WHY THIS TIER IS WORTH MORE THAN ITS POSITION SUGGESTS
------------------------------------------------------
Tier 4 sits low on the ladder because tldr is human-curated prose, not the
tool's own metadata: it is incomplete by design, it documents the common case
rather than the full surface, and nothing enforces that it matches the
installed version. Any of tiers 0-3 should beat it on the same field.

But it is the only tier that supplies *worked examples with the arguments
labelled*. A man page tells you `-m` takes `<msg>`; tldr tells you that people
write `git commit -m "message"` to do the thing called "commit staged files
with the specified message". That mapping from intent to invocation is exactly
what a natural-language planner needs, and it is what makes the corpus the
right place to start: ~7.3k commands, no execution, no network.

GRAMMAR, AS MEASURED RATHER THAN ASSUMED
----------------------------------------
Verified against all 7,367 English pages in the corpus: every page has a `# `
title, at least one `> ` description line, and bullet/command pairs where the
bullet ends in `:` and the command is wrapped in backticks. There were zero
violations, so this parser treats a structural deviation as a real anomaly
worth reporting rather than a routine case to paper over.

The one genuine ambiguity is brace escaping. tldr writes literal braces as
`\\{\\{` and `\\}\\}`, but placeholder content can itself contain single braces
(`{{{1..3}}}` is a shell brace-expansion inside a slot) and even a trailing
backslash (`{{A:\\}}` on the DOS `mount` page). Treating any `\\}` as an escape
mis-parses the latter; ignoring escapes entirely mis-parses the former. So the
rule here is: a backslash escapes a brace ONLY as a doubled `\\{\\{` / `\\}\\}`
pair, matching what the tldr spec actually specifies. That resolves both.

A handful of pages (~25, chiefly the `aws dynamodb` JSON examples) are
internally inconsistent -- they escape a closing brace whose opening partner is
unescaped -- and cannot be parsed under any single rule. Those examples are
DROPPED with a warning rather than guessed at. A dropped example costs a
suggestion the user never sees; a mis-parsed one puts a corrupted command in
their prompt buffer with Enter one keystroke away.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from cl_ai.ir import SourceTier

from .base import (
    ParamKind,
    RawExample,
    RawParam,
    RawTool,
    infer_capabilities,
    shells_for_os,
)

__all__ = [
    "ENGLISH_DIRS",
    "MAX_PAGE_BYTES",
    "Placeholder",
    "PlaceholderError",
    "TldrSource",
    "capabilities_for",
    "default_root",
    "infer_type",
    "is_flag_token",
    "parse_page",
    "parse_placeholders",
    "platforms_for",
    "slot_identifier",
]

# A page is prose; anything this large is not a tldr page and reading it would
# be a denial-of-service on catalog build rather than a useful finding.
MAX_PAGE_BYTES = 256 * 1024

# English pages only. Translations describe the same commands with the same
# placeholders, so they add parse risk and duplicate identities for no schema
# gain. The retrieval layer is where multilingual input belongs.
ENGLISH_DIRS = ("pages", "pages.en")


class PlaceholderError(ValueError):
    """A command line whose placeholder braces cannot be resolved."""


@dataclass(frozen=True)
class Placeholder:
    """One `{{...}}` span.

    `alternatives` is populated only for the option-alternation form
    `{{[-m|--message]}}`, which tldr uses to say "these two spellings are the
    same flag". That is real schema information -- it pairs a short form with
    its long form, which most other tiers state separately if at all.
    """

    content: str
    start: int
    end: int
    alternatives: tuple[str, ...] = ()

    @property
    def is_alternation(self) -> bool:
        return bool(self.alternatives)

    @property
    def is_option(self) -> bool:
        """Whether this slot names a flag rather than a value or subcommand.

        An alternation is NOT automatically an option. The corpus has 386
        distinct alternations whose members are subcommand aliases rather than
        flags -- `{{[images|image ls]}}`, `{{[add|install]}}` -- and treating
        those as options produced parameters with kind=OPTION and no flag at
        all, which a renderer cannot emit.
        """
        if self.alternatives:
            return any(is_flag_token(a) for a in self.alternatives)
        return self.content.startswith("-")


_ALTERNATION = re.compile(r"^\[([^\]]+)\]$")

# `-v`, `--verbose`, and the DOS/cmd convention `/L`, `/list`. The slash form
# only counts inside an alternation, where the surrounding brackets make the
# intent unambiguous -- a bare `/etc` is a path, not a flag.
#
# A short flag may be a digit: `pm list packages -3` selects third-party
# packages, `gzip -9` sets the level. Note this is deliberately laxer than
# _LITERAL_FLAG, which scans free command text and must NOT read the `-5` in
# `tool -5 file` as an option -- inside a placeholder alternation the brackets
# or pipe already establish that the token is a flag.
_DASH_FLAG = re.compile(r"^--?[A-Za-z0-9][\w-]*$")
_SLASH_FLAG = re.compile(r"^/[A-Za-z][\w-]*$")


def is_flag_token(token: str) -> bool:
    """Whether a token is spelled like an option."""
    token = token.strip()
    return bool(_DASH_FLAG.match(token) or _SLASH_FLAG.match(token))


def _is_escaped_pair(text: str, index: int) -> bool:
    """True at the start of a `\\{\\{` or `\\}\\}` literal-brace sequence.

    Only the DOUBLED form counts as an escape. tldr's own spec escapes literal
    braces in pairs, and single-brace content genuinely occurs inside slots --
    `{{A:\\}}` on the DOS mount page is a drive letter ending in a backslash,
    not an escape. Treating any `\\}` as an escape mis-parses that page.
    """
    return (
        index + 3 < len(text)
        and text[index] == "\\"
        and text[index + 1] in "{}"
        and text[index + 2] == "\\"
        and text[index + 3] == text[index + 1]
    )


def _balanced(text: str) -> bool:
    """Whether single braces in placeholder content pair up.

    Escaped pairs are skipped: they are literal characters, not structure.
    """
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if _is_escaped_pair(text, i):
            i += 4
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def parse_placeholders(text: str) -> tuple[Placeholder, ...]:
    """Extract every `{{...}}` span from a command template.

    Raises PlaceholderError when braces do not balance under the doubled-escape
    rule, so callers can drop the example rather than emit a partial parse.
    """
    out: list[Placeholder] = []
    i = 0
    n = len(text)
    while i < n:
        # Doubled escape: `\{\{` or `\}\}` is a literal brace pair, never a slot.
        if _is_escaped_pair(text, i):
            i += 4
            continue
        if not text.startswith("{{", i):
            i += 1
            continue

        # Close at the first `}}` whose enclosed content has balanced braces.
        #
        # A plain depth scan is wrong here, and Hypothesis found the case:
        # given `{{}[}` it let two NON-ADJACENT closing braces drop the depth
        # to zero, returning a span that did not end in `}}` at all. A
        # placeholder opens with two braces, so it must close with two
        # adjacent ones; requiring the content between to balance is what
        # keeps `{{{1..3}}}` (brace expansion inside a slot) intact while
        # still rejecting malformed input.
        end = -1
        j = i + 2
        while j < n - 1:
            if _is_escaped_pair(text, j):
                j += 4
                continue
            if text[j] == "}" and text[j + 1] == "}" and _balanced(text[i + 2 : j]):
                end = j + 2
                break
            j += 1

        if end == -1:
            raise PlaceholderError(f"unbalanced placeholder braces in {text!r}")

        content = text[i + 2 : end - 2]
        alternatives: tuple[str, ...] = ()
        stripped = content.strip()
        match = _ALTERNATION.match(stripped)
        if match:
            alternatives = tuple(
                part.strip() for part in match.group(1).split("|") if part.strip()
            )
        elif "|" in stripped:
            # Some pages omit the brackets: `{{-s|-3}}` on android/pm-list.md.
            # Accepted only when EVERY member is flag-shaped. A bare
            # `{{yes|no}}` is a set of values for a positional, not a pair of
            # option spellings, and conflating the two would invent a flag
            # that does not exist.
            parts = tuple(p.strip() for p in stripped.split("|") if p.strip())
            if len(parts) > 1 and all(is_flag_token(p) for p in parts):
                alternatives = parts
        out.append(
            Placeholder(
                content=content,
                start=i,
                end=end,
                alternatives=alternatives,
            )
        )
        i = end
    return tuple(out)


_NON_WORD = re.compile(r"[^0-9a-z]+")


def slot_identifier(content: str) -> str:
    """`path/to/commit_message_file` -> `commit_message_file`.

    tldr writes paths as `path/to/x` to signal "this is a filesystem path".
    The prefix is a convention, not part of the name, so it is stripped for the
    identifier while the path-ness is preserved by infer_type().
    """
    text = content.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        text = text.split("|")[-1]
    text = text.lstrip("-")
    # Repeatable forms: `path/to/file1 path/to/file2 ...` names one thing.
    text = text.replace("...", " ")
    parts = [p for p in text.split() if p]
    if parts:
        text = parts[0]
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = _NON_WORD.sub("_", text.lower()).strip("_")
    # Trailing ordinals from `file1 file2 ...` -- the parameter is `file`.
    text = re.sub(r"_?\d+$", "", text) or text
    return text or "value"


_INT = re.compile(r"^-?\d+$")
_NUM = re.compile(r"^-?\d*\.\d+$")
_PATHY = ("path/to/", "file", "directory", "dir", "folder")


def infer_type(content: str) -> str:
    """Map placeholder text to an IR param type.

    Conservative: anything not clearly numeric is a string. A wrong `integer`
    makes the renderer reject a perfectly good value, which the user sees as
    the tool refusing to complete something obvious.
    """
    text = content.strip().lower()
    if _INT.match(text):
        return "integer"
    if _NUM.match(text):
        return "number"
    if "..." in text or re.search(r"\b\w+1\s+\w+2\b", text):
        return "array"
    if any(hint in text for hint in _PATHY):
        return "string"
    return "string"


_LITERAL_FLAG = re.compile(r"^--?[A-Za-z][\w-]*$")


def _literal_flags(template: str, placeholders: Iterable[Placeholder]) -> list[str]:
    """Flags written plainly, e.g. the `--amend` in `git commit --amend`.

    These carry no placeholder because they take no value, and they are a large
    share of the real option surface -- skipping them would leave the catalog
    knowing only the flags that happen to need arguments.
    """
    spans = [(p.start, p.end) for p in placeholders]

    def inside(idx: int) -> bool:
        return any(s <= idx < e for s, e in spans)

    flags: list[str] = []
    for match in re.finditer(r"\S+", template):
        if inside(match.start()):
            continue
        token = match.group()
        if _LITERAL_FLAG.match(token):
            flags.append(token)
    return flags


def _params_from_template(
    template: str, placeholders: tuple[Placeholder, ...], description: str
) -> list[RawParam]:
    params: list[RawParam] = []
    seen: set[str] = set()

    for index, ph in enumerate(placeholders):
        if ph.is_alternation and not ph.is_option:
            # Subcommand aliases: `{{[images|image ls]}}`, `{{[add|install]}}`.
            # Recorded as a SUBCOMMAND with the full value set, so a renderer
            # knows these are interchangeable spellings of one step rather
            # than a flag it must emit.
            name = slot_identifier(ph.alternatives[0])
            key = f"sub:{name}"
            if key in seen:
                continue
            seen.add(key)
            params.append(
                RawParam(
                    name=name,
                    kind=ParamKind.SUBCOMMAND,
                    description=description,
                    choices=ph.alternatives,
                )
            )
            continue

        if ph.is_option:
            flags = [a for a in ph.alternatives if is_flag_token(a)]
            short = next(
                (a for a in flags if re.fullmatch(r"[-/][A-Za-z0-9]", a)), None
            )
            long = next((a for a in flags if a.startswith("--")), None)
            if long is None:
                # DOS/cmd spells long options with a single slash: `/list`.
                long = next(
                    (a for a in flags if a.startswith("/") and len(a) > 2), None
                )
            if not ph.alternatives:
                token = ph.content.strip()
                # Only a genuinely flag-shaped token becomes a flag. Anything
                # else that merely starts with a dash is content we do not
                # understand, and inventing `--foo|bar` as an option name would
                # put a flag in a rendered command that the tool will reject.
                if not is_flag_token(token):
                    params.append(
                        RawParam(
                            name=slot_identifier(token),
                            kind=ParamKind.POSITIONAL,
                            value_name=slot_identifier(token),
                            description=description,
                        )
                    )
                    continue
                if token.startswith("--"):
                    long = token
                else:
                    short = token
            name = slot_identifier(long or short or ph.content)

            # An option placeholder immediately followed by a value placeholder
            # means the flag takes that value: `{{[-o|--output]}} {{path/to/file}}`.
            value_name = None
            if index + 1 < len(placeholders):
                nxt = placeholders[index + 1]
                if not nxt.is_option:
                    value_name = slot_identifier(nxt.content)

            key = f"opt:{long or short or name}"
            if key in seen:
                continue
            seen.add(key)
            params.append(
                RawParam(
                    name=name,
                    kind=ParamKind.OPTION,
                    short=short,
                    long=long,
                    value_name=value_name,
                    description=description,
                    repeatable=False,
                    choices=ph.alternatives,
                )
            )
        else:
            name = slot_identifier(ph.content)
            key = f"pos:{name}"
            if key in seen:
                continue
            seen.add(key)
            params.append(
                RawParam(
                    name=name,
                    kind=ParamKind.POSITIONAL,
                    value_name=name,
                    description=description,
                    repeatable=infer_type(ph.content) == "array",
                )
            )

    # Values consumed by a preceding option are not independent positionals.
    consumed = {p.value_name for p in params if p.kind is ParamKind.OPTION and p.value_name}
    params = [
        p
        for p in params
        if not (p.kind is ParamKind.POSITIONAL and p.name in consumed)
    ]

    for flag in _literal_flags(template, placeholders):
        long = flag if flag.startswith("--") else None
        short = None if long else flag
        key = f"opt:{flag}"
        if key in seen:
            continue
        seen.add(key)
        params.append(
            RawParam(
                name=slot_identifier(flag),
                kind=ParamKind.OPTION,
                short=short,
                long=long,
                description=description,
            )
        )
    return params


def _literalise(template: str, placeholders: tuple[Placeholder, ...]) -> str:
    """Reduce a template to readable text: `{{[-m|--message]}}` -> `--message`."""
    out: list[str] = []
    cursor = 0
    for ph in placeholders:
        out.append(template[cursor : ph.start])
        if ph.alternatives:
            # For flags, show the long form -- a suggestion a human is about to
            # read and run is clearer as `--message` than `-m`. For subcommand
            # aliases there is no "long" form, so take the first, which is the
            # spelling the page led with.
            long = next((a for a in ph.alternatives if a.startswith("--")), None)
            if long is None and ph.is_option:
                long = next(
                    (a for a in ph.alternatives if is_flag_token(a) and len(a) > 2),
                    None,
                )
            out.append(long or ph.alternatives[0])
        else:
            out.append(ph.content)
        cursor = ph.end
    out.append(template[cursor:])
    text = "".join(out)
    return text.replace("\\{", "{").replace("\\}", "}")


_MORE_INFO = re.compile(r"More information:\s*<(?P<url>[^>]+)>", re.IGNORECASE)


def parse_page(text: str, source: str, os_target: str) -> RawTool | None:
    """Parse one tldr page. Returns None when it is not a usable page.

    None rather than an exception: a malformed page in a 7k-page corpus is a
    data problem to count, not a build failure to abort on.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    nonblank = [line for line in lines if line.strip()]
    if not nonblank or not nonblank[0].startswith("# "):
        return None

    title = nonblank[0][2:].strip()
    if not title:
        return None

    # The TITLE, not the filename, determines where the binary ends and the
    # subcommand path begins. Measured: `apt-get.md` is one binary with a
    # hyphen, `adb-devices.md` is `adb devices`, and `pm-install-commit.md` is
    # `pm install-commit` -- a subcommand that itself contains a hyphen. No
    # filename rule can separate those; the title states it outright.
    # Flag-shaped tokens stay in the path. Some tools spell their subcommands
    # as flags -- `pacman --sync`, `pacman --query`, `acme.sh --dns` are eight
    # separate documented commands -- so dropping them collapsed all of
    # pacman's surface into a single bare `pacman` identity and lost it. The
    # invocation genuinely is `pacman --sync`, so the token belongs in the
    # path; only the derived identifier strips the dashes.
    tokens = title.split()
    binary = tokens[0]
    path: list[str] = list(tokens[1:])

    description_lines: list[str] = []
    homepage: str | None = None
    for line in nonblank:
        if not line.startswith("> "):
            continue
        body = line[2:].strip()
        match = _MORE_INFO.search(body)
        if match:
            homepage = match.group("url")
            continue
        description_lines.append(body.rstrip("."))
    description = ". ".join(description_lines)

    warnings: list[str] = []
    examples: list[RawExample] = []
    params: list[RawParam] = []
    param_keys: set[str] = set()

    pending: str | None = None
    for line in nonblank[1:]:
        if line.startswith("- "):
            pending = line[2:].strip().rstrip(":")
            continue
        if not line.startswith("`"):
            continue
        template = line.strip()
        if template.startswith("`") and template.endswith("`") and len(template) >= 2:
            template = template[1:-1]
        else:
            warnings.append(f"command not backtick-wrapped: {line[:60]}")
            continue

        try:
            placeholders = parse_placeholders(template)
        except PlaceholderError as exc:
            warnings.append(str(exc))
            pending = None
            continue

        example_description = pending or description
        examples.append(
            RawExample(
                description=example_description,
                template=template,
                literal=_literalise(template, placeholders),
                slots=tuple(slot_identifier(p.content) for p in placeholders),
            )
        )
        for param in _params_from_template(template, placeholders, example_description):
            key = f"{param.kind.value}:{param.long or param.short or param.name}"
            if key in param_keys:
                continue
            param_keys.add(key)
            params.append(param)
        pending = None

    return RawTool(
        binary=binary,
        path=tuple(path),
        description=description,
        tier=SourceTier.TLDR,
        source=source,
        examples=tuple(examples),
        params=tuple(params),
        os_targets=frozenset({os_target}),
        homepage=homepage,
        warnings=tuple(warnings),
    )


def default_root() -> Path | None:
    """Locate a tldr corpus without guessing wildly.

    Checked in order: an explicit override, then the standard client caches.
    Returns None rather than a default path so the caller can report "tier 4
    unavailable" instead of failing later on a directory that never existed.
    """
    override = os.environ.get("CL_AI_TLDR_ROOT")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None

    candidates: list[Path] = []
    home = Path.home()
    candidates.append(home / ".cache" / "tldr")
    candidates.append(home / ".tldr" / "cache" / "pages")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "tldr")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(Path(xdg) / "tldr")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


class TldrSource:
    """Tier-4 source over a local tldr page tree.

    Takes a root rather than discovering one internally so tests and the build
    pipeline can point it at a fixture. Implements the Source protocol.
    """

    tier = SourceTier.TLDR
    name = "tldr"

    def __init__(self, root: Path | str | None = None) -> None:
        self._root = Path(root) if root is not None else default_root()

    @property
    def root(self) -> Path | None:
        return self._root

    def available(self) -> bool:
        return self._root is not None and self._page_dirs() != []

    def _page_dirs(self) -> list[Path]:
        """The English page directory, whether rooted at the repo or at `pages/`.

        Returns at most one directory. A tldr checkout can contain BOTH
        `pages/` and `pages.en/`, and measurement says they are byte-identical
        -- harvesting both yielded 14,734 tools for 7,153 distinct names, so
        every single tool was duplicated. ENGLISH_DIRS is therefore a priority
        order, not a set to union.
        """
        if self._root is None:
            return []
        for name in ENGLISH_DIRS:
            candidate = self._root / name
            if candidate.is_dir():
                return [candidate]
        # The root may already BE a pages directory (the layout the tldr client
        # uses), in which case its children are the OS folders.
        if self._root.is_dir() and any(
            (self._root / os_dir).is_dir() for os_dir in ("common", "linux", "windows")
        ):
            return [self._root]
        return []

    def harvest(self, binaries: Iterable[str] | None = None) -> Iterator[RawTool]:
        wanted = {b.lower() for b in binaries} if binaries is not None else None
        for pages in self._page_dirs():
            for os_dir in sorted(p for p in pages.iterdir() if p.is_dir()):
                os_target = os_dir.name
                for page in sorted(os_dir.glob("*.md")):
                    try:
                        if page.stat().st_size > MAX_PAGE_BYTES:
                            continue
                        text = page.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        # An unreadable page is one missing tool, not a failed
                        # catalog build.
                        continue
                    tool = parse_page(
                        text, source=f"tldr/{os_target}/{page.name}", os_target=os_target
                    )
                    if tool is None:
                        continue
                    if wanted is not None and tool.binary.lower() not in wanted:
                        continue
                    yield tool


def platforms_for(tool: RawTool) -> frozenset[str]:
    """Shell profiles this finding plausibly applies to."""
    return shells_for_os(tool.os_targets)


def capabilities_for(tool: RawTool) -> frozenset[str]:
    """Behavioural tags inferred from identity, examples and prose."""
    return infer_capabilities(
        tool.binary,
        tool.path,
        examples=[e.literal for e in tool.examples],
        description=tool.description,
    )
