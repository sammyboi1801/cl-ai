"""Property-based quoting tests.

The hand-written NASTY list can only contain cases someone thought of. Two of
the three bugs found reviewing the first commit were asymmetries between
profiles -- the kind of thing a per-profile example list is structurally bad at
catching, because you write the same examples for each and they agree with you.

Two tiers, because cost differs by three orders of magnitude:

  * pure properties -- no subprocess, hundreds of examples
  * oracle properties -- spawns a real shell per example, so a small budget

If Hypothesis finds a failure, add the minimal example to NASTY in
test_quoting.py so it is checked cheaply forever after.
"""

from __future__ import annotations

import pytest

from cl_ai.platform_ import PROFILES, QuoteStyle, is_available
from cl_ai.render import quoting
from cl_ai.render.quoting import QuotingError

from .shell_oracle import roundtrip

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

ALL_SHELLS = ["bash", "zsh", "fish", "powershell", "pwsh", "cmd"]
POSIX_SHELLS = ["bash", "zsh", "fish"]

#: Deliberately dense in metacharacters. Uniform unicode would spend most of its
#: budget on characters no shell treats specially.
DANGEROUS = "'\"`$\\;&|<>(){}[]*?!~#%^@=+,:/ \t\n-"

text = st.text(
    alphabet=st.one_of(
        st.sampled_from(DANGEROUS),
        st.characters(min_codepoint=32, max_codepoint=126),
        st.characters(min_codepoint=161, max_codepoint=0x2FFF),
        # Written as escapes on purpose: a literal RTL override in source
        # reorders this file in any editor that renders bidirectional text.
        st.sampled_from([chr(0x1f600), chr(0x4e2d), chr(0x202e), chr(0xa0)]),
    ),
    min_size=0,
    max_size=60,
)


def _quote_or_skip(value: str, shell_id: str) -> str:
    try:
        return quoting.quote(value, PROFILES[shell_id])
    except QuotingError:
        assume(False)  # refusing is a valid outcome; not a counterexample
        raise


# ------------------------------------------------------------ pure properties

@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(value=text)
@settings(max_examples=300, deadline=None)
def test_quote_is_deterministic(shell_id, value):
    a = _quote_or_skip(value, shell_id)
    b = _quote_or_skip(value, shell_id)
    assert a == b


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(value=text)
@settings(max_examples=300, deadline=None)
def test_quote_never_produces_an_empty_token(shell_id, value):
    """An argument must never disappear from argv."""
    assert _quote_or_skip(value, shell_id) != ""


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(value=text)
@settings(max_examples=300, deadline=None)
def test_quoted_output_has_no_bare_whitespace(shell_id, value):
    """Whitespace outside quotes would split one argument into several."""
    quoted = _quote_or_skip(value, shell_id)
    if quoted.startswith(("'", '"')):
        return
    assert not any(c.isspace() for c in quoted), (
        f"{shell_id} left whitespace unquoted: {quoted!r}"
    )


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(value=text)
@settings(max_examples=300, deadline=None)
def test_unquoted_output_is_returned_verbatim(shell_id, value):
    """If we decline to quote, we must not have altered the value either."""
    quoted = _quote_or_skip(value, shell_id)
    if not quoted.startswith(("'", '"')):
        assert quoted == value


@given(value=text)
@settings(max_examples=300, deadline=None)
def test_profiles_sharing_a_quote_style_agree(value):
    """Cross-profile invariant, stated over quote_style rather than over "POSIX".

    Any two profiles declaring the same QuoteStyle must produce identical
    output; if they should not, the profile is mismodelled and the fix belongs
    in platform_, not here.

    This test previously asserted that bash, zsh AND fish agreed, which was
    wrong -- it encoded the mismodelling rather than catching it. Fish now has
    its own style, so it is no longer in the bash/zsh group.
    """
    groups: dict[object, dict[str, str]] = {}
    for shell in ALL_SHELLS:
        groups.setdefault(PROFILES[shell].quote_style, {})[shell] = _quote_or_skip(
            value, shell
        )
    for style, outs in groups.items():
        assert len(set(outs.values())) == 1, f"{style} profiles disagree: {outs}"


@given(value=st.text(alphabet="ab\\'", min_size=1, max_size=8))
@settings(max_examples=200, deadline=None)
def test_fish_diverges_from_posix_on_backslashes(value):
    """Guards the fix for the bug CI found.

    Fish processes \\ and ' as escapes inside single quotes; bash and zsh do
    not. If these two ever produced the same output for a backslash-bearing
    value again, fish would have silently been re-modelled as POSIX.
    """
    assume("\\" in value)
    assert _quote_or_skip(value, "fish") != _quote_or_skip(value, "bash")


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
# Construct dash-prefixed values directly. Generating freely and filtering with
# assume() throws away ~95% of inputs, which Hypothesis rightly flags as
# distorting the distribution.
@given(value=st.builds(lambda d, rest: d + rest,
                       st.sampled_from(["-", "--"]), text))
@settings(max_examples=200, deadline=None)
def test_flag_like_values_are_never_left_bare(shell_id, value):
    """Regression as a property, not a handful of examples.

    A value beginning with a dash, left unquoted, is read by the target program
    as an option rather than as data.
    """
    quoted = _quote_or_skip(value, shell_id)
    assert quoted.startswith(("'", '"')), (
        f"{shell_id} left a flag-like value bare: {quoted!r}"
    )


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(value=text, ctl=st.sampled_from(["\x00", "\x07", "\x1b", "\x7f"]))
@settings(max_examples=200, deadline=None)
def test_control_characters_are_always_refused(shell_id, value, ctl):
    with pytest.raises(QuotingError):
        quoting.quote(value + ctl, PROFILES[shell_id])


@given(value=st.builds(lambda a, c, b: a + c + b,
                       text, st.sampled_from("%!"), text))
@settings(max_examples=200, deadline=None)
def test_cmd_refuses_everything_it_cannot_escape(value):
    """cmd expands %VAR% and !VAR! inside quotes with no reliable escape."""
    with pytest.raises(QuotingError):
        quoting.quote(value, PROFILES["cmd"])


@pytest.mark.parametrize("shell_id", ALL_SHELLS)
@given(prefix=st.text(alphabet="abc012", min_size=1, max_size=6))
@settings(max_examples=50, deadline=None)
def test_trailing_newline_is_never_left_bare(shell_id, prefix):
    """Pinning a real bug the property tests found on their first run.

    Python's ``$`` matches before a trailing newline, so the "is this safe to
    leave unquoted" regexes accepted a value ending in one and emitted it bare.
    An unquoted newline ends the command, making whatever the renderer appended
    a second command. The anchors are ``\\Z`` now; this keeps them that way.
    """
    value = prefix + "\n"
    try:
        quoted = quoting.quote(value, PROFILES[shell_id])
    except QuotingError:
        return  # cmd refuses newlines outright, which is also correct
    assert quoted.startswith(("'", '"')), (
        f"{shell_id} emitted an unquoted trailing newline: {quoted!r}"
    )


# ---------------------------------------------------------- oracle properties

@pytest.mark.parametrize("shell_id", ["bash", "zsh", "fish", "powershell", "pwsh"])
@given(value=text)
@settings(
    max_examples=40,           # each example spawns a shell; keep CI honest
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
def test_generated_values_survive_the_real_shell(shell_id, value):
    """The strong property: quote it, run it, get the same bytes back."""
    if not is_available(shell_id):
        pytest.skip(f"{shell_id} not installed here")
    _quote_or_skip(value, shell_id)
    recovered = roundtrip(value, shell_id)
    assert recovered == value, (
        f"{shell_id} round-trip mismatch\n"
        f"  sent     : {value!r}\n"
        f"  recovered: {recovered!r}"
    )


@pytest.mark.parametrize("shell_id", ["bash", "zsh", "fish", "powershell", "pwsh"])
@given(
    payload=st.sampled_from([
        "; touch pwned", "&& touch pwned", "| touch pwned",
        "$(touch pwned)", "`touch pwned`", "'; touch pwned; '",
    ]),
    prefix=st.text(alphabet="abc-_. ", max_size=8),
)
@settings(max_examples=25, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture,
                                 HealthCheck.too_slow])
def test_injection_never_executes(shell_id, payload, prefix):
    if not is_available(shell_id):
        pytest.skip(f"{shell_id} not installed here")
    value = prefix + payload
    _quote_or_skip(value, shell_id)
    assert roundtrip(value, shell_id) == value, (
        f"SECURITY: {shell_id} altered a hostile payload {value!r}"
    )


def test_posix_and_powershell_quote_differently_somewhere():
    """Guards the guard.

    If the two quoters ever produced identical output for every input, the
    cross-profile properties above would be vacuously true and would stop
    protecting anything.
    """
    diverged = any(
        quoting.quote(v, PROFILES["bash"]) != quoting.quote(v, PROFILES["powershell"])
        for v in ["it's", 'say "hi"', "a\\b", "x~y"]
    )
    assert diverged
    assert PROFILES["bash"].quote_style is not PROFILES["powershell"].quote_style
    assert PROFILES["cmd"].quote_style is QuoteStyle.CMD
