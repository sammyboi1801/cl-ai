"""Wire format between the shell widget and the daemon.

Newline-delimited JSON, one object per line, request/response. Chosen because
every shell we target can produce and parse it with no dependencies, and
because a line-oriented stream is trivial to frame correctly -- the widget is
the layer we least want to debug.

Two rules hold this together:

  * The protocol is versioned, and a version mismatch is a normal outcome
    rather than an error. A user updates the Python package without restarting
    their shell all the time; the widget must notice and fall back to ordinary
    Tab rather than misbehave.

  * Every field the widget needs has a default. An older widget talking to a
    newer daemon must not crash on a field it has never heard of, so decoding
    ignores unknown keys instead of rejecting them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

#: Bumped only for breaking changes. Additive fields do not need it, because
#: both directions tolerate unknown keys.
PROTOCOL_VERSION = 1

#: Nothing we exchange is large, and a bound means a wedged or hostile peer
#: cannot make the other side allocate without limit.
MAX_LINE_BYTES = 1 << 20


class Kind(str, Enum):
    SUGGEST = "suggest"       # buffer -> ranked commands
    PING = "ping"             # liveness and version handshake
    SHUTDOWN = "shutdown"


class ErrorCode(str, Enum):
    NONE = "none"
    BAD_REQUEST = "bad_request"
    VERSION_MISMATCH = "version_mismatch"
    UNAVAILABLE = "unavailable"        # not ready yet: catalog still building
    INTERNAL = "internal"


class ProtocolError(ValueError):
    """A message could not be decoded. Never raised at the widget; the widget
    treats any failure as "no suggestion" and lets Tab behave normally."""


@dataclass(frozen=True)
class Request:
    kind: Kind = Kind.SUGGEST
    #: The line the user has typed so far.
    buffer: str = ""
    #: Where the cursor sits. Reserved for mid-line completion.
    cursor: int = 0
    #: The widget knows exactly which shell it is; the daemon must not guess.
    shell: str = ""
    cwd: str = ""
    #: How many candidates to return for the user to cycle.
    limit: int = 5
    #: Widget-side budget. The daemon returns its best partial answer rather
    #: than overrunning this, because a late suggestion is worse than none.
    deadline_ms: int = 120
    version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class Suggestion:
    """One candidate, already rendered for the requesting shell."""

    command: str
    #: Shown beside the candidate while cycling.
    description: str = ""
    source: str = ""
    #: True when the command is destructive; the widget marks these, since it
    #: lands in the buffer and Enter is one keystroke away.
    dangerous: bool = False


@dataclass(frozen=True)
class Response:
    ok: bool = True
    suggestions: tuple[Suggestion, ...] = ()
    error: ErrorCode = ErrorCode.NONE
    #: Plain language, and shown to the user: they may not know what a daemon
    #: or a protocol version is.
    message: str = ""
    elapsed_ms: float = 0.0
    version: int = PROTOCOL_VERSION


def _enum_safe(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def encode(message: Request | Response) -> bytes:
    """Serialise to one newline-terminated line of UTF-8 JSON."""
    payload = asdict(message)
    payload = {k: _enum_safe(v) for k, v in payload.items()}
    if isinstance(message, Response):
        payload["suggestions"] = [asdict(s) for s in message.suggestions]
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = line.encode("utf-8") + b"\n"
    if len(raw) > MAX_LINE_BYTES:
        raise ProtocolError(f"message of {len(raw)} bytes exceeds the limit")
    return raw


def _coerce(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields the dataclass declares, and fix up enums.

    Dropping unknown keys is what lets an old widget talk to a new daemon.
    """
    known = {f.name: f for f in fields(cls)}
    out: dict[str, Any] = {}
    for key, value in payload.items():
        spec = known.get(key)
        if spec is None:
            continue
        out[key] = value
    return out


def decode_request(raw: bytes | str) -> Request:
    payload = _parse(raw)
    data = _coerce(Request, payload)
    if "kind" in data:
        try:
            data["kind"] = Kind(data["kind"])
        except ValueError as exc:
            raise ProtocolError(f"unknown request kind {data['kind']!r}") from exc
    try:
        return Request(**data)
    except TypeError as exc:
        raise ProtocolError(str(exc)) from exc


def decode_response(raw: bytes | str) -> Response:
    payload = _parse(raw)
    data = _coerce(Response, payload)
    if "error" in data:
        try:
            data["error"] = ErrorCode(data["error"])
        except ValueError:
            # An error code from a newer daemon is still an error; treat it as
            # a generic one rather than failing to read the message at all.
            data["error"] = ErrorCode.INTERNAL
    if "suggestions" in data:
        data["suggestions"] = tuple(
            Suggestion(**_coerce(Suggestion, s))
            for s in data["suggestions"]
            if isinstance(s, dict)
        )
    try:
        return Response(**data)
    except TypeError as exc:
        raise ProtocolError(str(exc)) from exc


def _parse(raw: bytes | str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        if len(raw) > MAX_LINE_BYTES:
            raise ProtocolError("message exceeds the size limit")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"not valid UTF-8: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected an object, got {type(payload).__name__}")
    return payload


def version_mismatch(request: Request) -> Response | None:
    """The handshake. Returns a refusal when the peer speaks another version.

    Deliberately a normal Response rather than an exception: the widget shows
    the message and falls back to ordinary Tab, which is what a user wants
    when they have upgraded the package but not restarted their shell.
    """
    if request.version == PROTOCOL_VERSION:
        return None
    return Response(
        ok=False,
        error=ErrorCode.VERSION_MISMATCH,
        message=(
            f"cl-ai shell integration speaks protocol v{request.version} but "
            f"the daemon speaks v{PROTOCOL_VERSION}. Restart your shell, or "
            f"run `cl-ai init` to reinstall the integration."
        ),
    )
