"""Protocol tests.

The wire format is the contract between a Python daemon and shell scripts that
will be written and updated independently. Most of what matters here is not
"does a round trip work" but "what happens when the two sides disagree" --
because they will, every time a user upgrades the package without restarting
their shell.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from cl_ai.daemon import protocol
from cl_ai.daemon.protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    Kind,
    ProtocolError,
    Request,
    Response,
    Suggestion,
    decode_request,
    decode_response,
    encode,
    version_mismatch,
)


def test_request_round_trip():
    original = Request(buffer="undo my last commit", cursor=7, shell="powershell",
                       cwd="/tmp", limit=3, deadline_ms=90)
    assert decode_request(encode(original)) == original


def test_response_round_trip():
    original = Response(
        suggestions=(
            Suggestion("git reset --soft HEAD~1", "Undo, keep changes", "git", False),
            Suggestion("git reset --hard HEAD~1", "Undo, discard", "git", True),
        ),
        elapsed_ms=12.5,
    )
    decoded = decode_response(encode(original))
    assert decoded == original
    assert decoded.suggestions[1].dangerous is True


def test_encoding_is_one_line():
    """The framing is newline-delimited, so a payload must never contain one.

    A shell reading a line at a time would otherwise desynchronise for the
    rest of the session.
    """
    raw = encode(Response(suggestions=(Suggestion("echo 'a\nb'"),)))
    assert raw.count(b"\n") == 1
    assert raw.endswith(b"\n")


def test_unicode_survives_and_stays_one_line():
    value = "git commit -m 'café \U0001f600 中文'"
    decoded = decode_response(encode(Response(suggestions=(Suggestion(value),))))
    assert decoded.suggestions[0].command == value


# ------------------------------------------------------ disagreeing versions

def test_version_mismatch_is_a_response_not_an_exception():
    """A user who upgrades the package without restarting their shell is the
    normal case, not an error case. The widget shows the message and falls
    back to ordinary Tab."""
    response = version_mismatch(Request(version=PROTOCOL_VERSION + 1))
    assert response is not None
    assert response.ok is False
    assert response.error is ErrorCode.VERSION_MISMATCH
    assert "restart your shell" in response.message.lower()


def test_matching_version_passes():
    assert version_mismatch(Request(version=PROTOCOL_VERSION)) is None


def test_unknown_fields_are_ignored_not_rejected():
    """An old widget must survive a newer daemon. Additive changes are the
    common kind, so they must not require a version bump."""
    payload = json.dumps({
        "kind": "suggest", "buffer": "ls", "version": PROTOCOL_VERSION,
        "a_field_from_the_future": {"nested": [1, 2, 3]},
    })
    assert decode_request(payload).buffer == "ls"


def test_unknown_error_code_degrades_to_internal():
    """A newer daemon's error code is still an error; failing to read the
    message at all would be worse than losing its precise name."""
    payload = json.dumps({"ok": False, "error": "some_future_code"})
    assert decode_response(payload).error is ErrorCode.INTERNAL


def test_unknown_request_kind_is_rejected():
    """Unlike fields, an unknown *kind* cannot be guessed at safely."""
    with pytest.raises(ProtocolError):
        decode_request(json.dumps({"kind": "rm_rf_everything"}))


# --------------------------------------------------------------- bad input

@pytest.mark.parametrize("payload", [
    b"", b"\n", b"not json at all", b"[1,2,3]", b'"a string"', b"null", b"{",
    b"\xff\xfe invalid utf8",
])
def test_malformed_input_raises_protocol_error_not_something_else(payload):
    """One exception type, so callers can catch it exhaustively. A stray
    UnicodeDecodeError escaping to the daemon loop would kill a connection."""
    with pytest.raises(ProtocolError):
        decode_request(payload)
    with pytest.raises(ProtocolError):
        decode_response(payload)


def test_oversized_messages_are_refused_both_ways():
    """A bound means a wedged or hostile peer cannot make us allocate without
    limit."""
    huge = "x" * (protocol.MAX_LINE_BYTES + 10)
    with pytest.raises(ProtocolError):
        encode(Response(suggestions=(Suggestion(huge),)))
    with pytest.raises(ProtocolError):
        decode_request(json.dumps({"buffer": huge}).encode("utf-8"))


def test_wrong_types_are_refused():
    with pytest.raises(ProtocolError):
        decode_request(json.dumps({"kind": 17}))


def test_defaults_let_a_minimal_message_decode():
    """The widget is the layer we least want to debug; it should be able to
    send almost nothing and still be understood."""
    assert decode_request("{}") == Request()
    assert decode_response("{}") == Response()


def test_suggestions_that_are_not_objects_are_dropped():
    decoded = decode_response(json.dumps({"suggestions": ["a string", None, 5]}))
    assert decoded.suggestions == ()


def test_enums_serialise_as_their_values():
    """The other side is a shell script, not Python: it must see plain
    strings, never a repr."""
    payload = json.loads(encode(Request(kind=Kind.PING)).decode("utf-8"))
    assert payload["kind"] == "ping"
    payload = json.loads(encode(Response(error=ErrorCode.UNAVAILABLE)).decode("utf-8"))
    assert payload["error"] == "unavailable"


def test_messages_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Request().buffer = "mutated"  # type: ignore[misc]
