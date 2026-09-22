"""Per-shell quoting and escaping.

This is the one place in the system where a bug is dangerous rather than
annoying. Everything else degrades visibly: retrieval picks a wrong tool and
the user presses a key; the daemon dies and Tab behaves like Tab. A quoting bug
produces a command that *looks* right, lands in the buffer, and gets run.

Two rules, both load-bearing:

  1. Never concatenate a user value into a command. Every value goes through
     quote().

  2. Refuse rather than approximate. If a value cannot be represented safely in
     a target shell -- and cmd.exe has several genuine cases -- raise. A visible
     refusal beats a command that half works.
"""

from __future__ import annotations

import re

from ..platform_ import QuoteStyle, ShellProfile


class QuotingError(ValueError):
    """A value cannot be safely represented in the target shell.

    Carries the reason in plain language: it is surfaced to the user, who may
    not know what a shell metacharacter is.
    """


# Characters that never need quoting in a POSIX shell. Mirrors shlex's set.
# Note the omissions, each deliberate:
#   ~  home expansion            *?[  globbing
#   !  history expansion         ^   csh-era negation
#
# Anchor with \Z, never with $. In Python, `$` also matches immediately before
# a trailing newline, so `^[\w]+$` happily accepts a value ending in one. That
# value would then be emitted BARE, the unquoted newline would terminate the
# command, and anything the renderer appended became a second command -- a
# command injection. Found by the property tests on their first run; the whole
# family of anchors below is \Z for this reason.
_POSIX_SAFE = re.compile(r"^[\w@%+=:,./-]+\Z", re.ASCII)

# PowerShell's bare-word rules are stricter than POSIX's in practice: `,` and
# `%` are operators/aliases, `@` starts a splat or array, `:` can form a drive
# or scope reference, `.` is safe only mid-token.
_PS_SAFE = re.compile(r"^[\w/=+-]+\Z", re.ASCII)

# cmd.exe: anything beyond this gets wrapped, and several things get refused.
_CMD_SAFE = re.compile(r"^[\w@+=:,./-]+\Z", re.ASCII)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _reject_control(value: str, shell: str) -> None:
    """NUL and most control characters cannot survive any shell's argv."""
    if "\x00" in value:
        raise QuotingError(
            f"value contains a NUL byte, which cannot be passed to {shell}"
        )
    found = _CONTROL.search(value)
    if found:
        raise QuotingError(
            f"value contains the control character {found.group()!r}, "
            f"which cannot be safely passed to {shell}"
        )


def quote_posix(value: str) -> str:
    """Quote for bash / zsh / fish.

    Single quotes are absolute in POSIX shells: no expansion of any kind occurs
    inside them, so the only character needing care is the single quote itself,
    which is closed, escaped, and reopened.
    """
    _reject_control(value, "a POSIX shell")
    if value == "":
        return "''"
    # A leading dash must be quoted even though the characters are otherwise
    # safe. quote() is for VALUES; the renderer emits flags itself. Left bare, a
    # value like "--force" or "-rf" would be read by the target program as an
    # option rather than as data. shlex.quote does leave these bare, which is
    # correct for its purpose and wrong for ours.
    if _POSIX_SAFE.match(value) and not value.startswith("-"):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def quote_powershell(value: str) -> str:
    """Quote for PowerShell / pwsh.

    PowerShell's single-quoted strings are literal -- `$`, backtick and `"` do
    not expand inside them -- and an embedded single quote is written by
    doubling it. This is why we prefer single quotes over double: the
    double-quoted form would require escaping `$` and backtick as well.
    """
    _reject_control(value, "PowerShell")
    if value == "":
        return "''"
    if _PS_SAFE.match(value) and not value.startswith("-"):
        return value
    return "'" + value.replace("'", "''") + "'"


def quote_cmd(value: str) -> str:
    """Quote for cmd.exe -- the one that genuinely cannot always succeed.

    cmd.exe performs environment expansion *before* the quoted string reaches
    the program, and there is no escape for `%` that is correct both
    interactively and in a batch file. Delayed expansion does the same for `!`
    when it is enabled, which we cannot detect from here. Newlines cannot be
    represented in a single command line at all.

    Rather than emit something that works in one context and silently misfires
    in another, these cases raise.
    """
    _reject_control(value, "cmd.exe")
    if "\n" in value or "\r" in value:
        raise QuotingError("cmd.exe cannot represent a newline inside an argument")
    if "%" in value:
        raise QuotingError(
            "cmd.exe expands %VAR% even inside quotes and offers no reliable "
            "escape; this value cannot be passed safely"
        )
    if "!" in value:
        raise QuotingError(
            "cmd.exe expands !VAR! when delayed expansion is enabled, which "
            "cannot be detected here; this value cannot be passed safely"
        )
    if value == "":
        return '""'
    if _CMD_SAFE.match(value) and not value.startswith("-"):
        return value
    return '"' + value.replace('"', '""') + '"'


_QUOTERS = {
    QuoteStyle.POSIX: quote_posix,
    QuoteStyle.POWERSHELL: quote_powershell,
    QuoteStyle.CMD: quote_cmd,
}


def quote(value: str, profile: ShellProfile) -> str:
    """Quote a single argument for the given shell.

    Non-string values are a programming error rather than a user input problem,
    so they raise TypeError instead of QuotingError.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"quote() takes str, got {type(value).__name__}; "
            f"convert explicitly so the caller decides on formatting"
        )
    return _QUOTERS[profile.quote_style](value)


def quote_all(values: list[str], profile: ShellProfile) -> str:
    """Quote every value and join them. The normal entry point for renderers."""
    return " ".join(quote(v, profile) for v in values)
