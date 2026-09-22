"""Local IPC between the shell widget and the daemon.

Named pipe on Windows, unix domain socket on POSIX. Both are local-only by
construction, which matters: this carries the contents of your command line
and your recent shell history, and none of that should ever touch a TCP port
where another machine -- or another user's process -- could reach it.

The governing constraint is that a failure here must be invisible. If the
daemon is missing, slow, wedged or speaking another protocol version, Tab has
to behave like ordinary Tab. A completion that hangs a terminal is far worse
than one that never appears, so every client path has a deadline and every
error returns None rather than raising.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .protocol import (
    MAX_LINE_BYTES,
    ProtocolError,
    Request,
    Response,
    decode_request,
    decode_response,
    encode,
)

log = logging.getLogger("cl_ai.daemon.transport")

WINDOWS = os.name == "nt"

#: Windows named pipes live in a flat kernel namespace, not the filesystem.
_PIPE_PREFIX = r"\\.\pipe"


def default_endpoint(user: str | None = None) -> str:
    """Where the daemon listens, per user.

    Per-user rather than per-machine on purpose. A shared endpoint would let
    one account read another account's command line on a multi-user box.
    """
    user = user or _current_user()
    if WINDOWS:
        return rf"{_PIPE_PREFIX}\cl-ai-{user}"
    # XDG_RUNTIME_DIR is already user-private and cleaned on logout; /tmp is
    # the fallback and gets an explicit 0700 directory below.
    base = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/cl-ai-{user}"
    return str(Path(base) / f"cl-ai-{user}.sock")


#: sockaddr_un.sun_path is a fixed-size char array: 108 bytes on Linux, 104 on
#: macOS and the BSDs. Exceed it and bind() fails with a message that says
#: nothing about length. macOS makes this easy to hit without trying, because
#: TMPDIR there is a long per-session path under /private/var/folders.
_SUN_PATH_MAX = 100


def _check_unix_path_length(path: str) -> None:
    encoded = len(path.encode("utf-8"))
    if encoded > _SUN_PATH_MAX:
        raise OSError(
            f"socket path is {encoded} bytes, over the ~{_SUN_PATH_MAX} byte "
            f"limit for a unix domain socket on this platform: {path}"
        )


def _current_user() -> str:
    for var in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(var)
        if value:
            return "".join(c for c in value if c.isalnum() or c in "-_") or "default"
    return str(os.getuid()) if hasattr(os, "getuid") else "default"


# --------------------------------------------------------------------- client

def request(
    message: Request,
    endpoint: str | None = None,
    timeout_s: float = 0.5,
) -> Response | None:
    """Send a request; return the response, or None if anything at all fails.

    None is the entire error vocabulary by design. The widget cannot usefully
    distinguish "no daemon" from "daemon too slow" from "garbled reply" -- in
    every case it must quietly let Tab do its normal job.
    """
    endpoint = endpoint or default_endpoint()
    deadline = time.monotonic() + timeout_s
    payload = encode(message)

    # One bounded retry. A connect can fail transiently for reasons that say
    # nothing about whether a daemon is there -- it may be mid-restart, or the
    # listener may be between binding and accepting. Retrying once inside the
    # existing deadline costs nothing when there is genuinely no daemon (the
    # connect fails immediately) and removes a class of spurious misses that
    # would otherwise read to the user as Tab randomly not working.
    for attempt in (0, 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            if WINDOWS:
                raw = _request_loopback(payload, endpoint, deadline)
            else:
                raw = _request_unix(payload, endpoint, deadline)
            if raw is not None:
                return decode_response(raw)
        except ProtocolError:
            return None               # a reply we cannot read will not improve
        except OSError:
            # Connect and read failures are retried once. Refused, reset,
            # missing socket file, a listener mid-restart -- none of these
            # distinguish "no daemon" from "bad timing", and one cheap retry
            # inside the existing deadline resolves the second without
            # delaying the first, because a genuine absence fails instantly.
            if attempt == 1:
                return None
        if attempt == 0:
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return None


def _request_unix(payload: bytes, endpoint: str, deadline: float) -> bytes | None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(remaining)
        sock.connect(endpoint)
        sock.sendall(payload)
        return _read_line(sock.recv, deadline)


def _request_loopback(payload: bytes, endpoint: str, deadline: float) -> bytes | None:
    """Windows stand-in for a named pipe. See _serve_pipe for why it is here.

    The port is read from a companion file rather than fixed, so two users on
    one machine do not collide.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    try:
        port = int(Path(port_file_for(endpoint)).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(remaining)
        sock.connect(("127.0.0.1", port))
        sock.sendall(payload)
        return _read_line(sock.recv, deadline)


def _read_line(recv: Callable[[int], bytes], deadline: float) -> bytes | None:
    chunks = bytearray()
    while time.monotonic() < deadline:
        chunk = recv(4096)
        if not chunk:
            break
        chunks.extend(chunk)
        if b"\n" in chunks:
            break
        if len(chunks) > MAX_LINE_BYTES:
            return None
    if not chunks:
        return None
    return bytes(chunks).split(b"\n", 1)[0]


# --------------------------------------------------------------------- server

Handler = Callable[[Request], Response]


class Server:
    """A tiny line-oriented server. One request per connection.

    Threaded rather than async: the work behind a request is a model call that
    holds the GIL anyway, connections are few, and a thread per short-lived
    connection is far easier to reason about than an event loop in a component
    whose main requirement is that it never hangs.
    """

    def __init__(
        self,
        handler: Handler,
        endpoint: str | None = None,
        *,
        backlog: int = 16,
    ) -> None:
        self.handler = handler
        self.endpoint = endpoint or default_endpoint()
        self.backlog = backlog
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._ready = threading.Event()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        # Idempotent. Starting twice used to spin up a second thread that bound
        # the same path, unlinked the first server's socket out from under it,
        # and left both fighting over connections -- which presented as clients
        # intermittently getting no reply, on every platform. A second start is
        # a caller error, but one that must not corrupt a running server.
        if self._thread is not None and self._thread.is_alive():
            return
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve, name="cl-aid", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            # Surface what actually went wrong. "did not come up" on its own
            # sent us hunting for a race when the real cause was a socket path
            # over the length limit -- the error was there, just discarded.
            if self._error is not None:
                raise RuntimeError(
                    f"server failed to bind {self.endpoint}: {self._error}"
                ) from self._error
            raise RuntimeError(f"server did not come up on {self.endpoint}")

    def stop(self) -> None:
        self._stop.set()
        # Unblock accept() by connecting to ourselves; closing the socket from
        # another thread is not reliably enough to wake it on every platform.
        with contextlib.suppress(Exception):
            request(Request(kind=Request().kind), self.endpoint, timeout_s=0.2)
        if self._sock is not None:
            with contextlib.suppress(Exception):
                self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._cleanup()

    def __enter__(self) -> Server:  # noqa: PYI034 - Self needs 3.11, floor is 3.10
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- internals --------------------------------------------------------

    def _cleanup(self) -> None:
        if not WINDOWS:
            with contextlib.suppress(OSError):
                os.unlink(self.endpoint)

    def _serve(self) -> None:
        try:
            if WINDOWS:
                self._serve_pipe()
            else:
                self._serve_unix()
        except BaseException as exc:      # noqa: BLE001 - reported via start()
            self._error = exc
            self._ready.set()             # unblock start(), which re-raises
            log.error("serve loop failed on %s: %s", self.endpoint, exc)

    def _serve_unix(self) -> None:
        path = Path(self.endpoint)
        _check_unix_path_length(str(path))
        # Only tighten a directory we created ourselves. The previous version
        # chmod'd the parent unconditionally, which for an endpoint directly
        # under /tmp meant trying to make the machine's shared temp directory
        # 0700 -- it failed harmlessly as an unprivileged user, but it was an
        # attempt to reconfigure something we do not own.
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(path.parent, 0o700)
        with contextlib.suppress(OSError):
            os.unlink(path)     # a stale socket from a crashed daemon

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(path))
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        self._sock.listen(self.backlog)
        self._sock.settimeout(0.25)
        self._ready.set()
        self._accept_loop()

    def _serve_pipe(self) -> None:
        # Deferred: pywin32 is not a dependency and the stdlib has no named
        # pipe server. The Windows widget therefore uses the fallback path in
        # _serve_loopback until this is implemented properly.
        self._serve_loopback()

    def _serve_loopback(self) -> None:
        """127.0.0.1 on an ephemeral port, with the port written beside the
        endpoint name.

        Strictly a stand-in for a named pipe. It is loopback-only, but unlike
        a pipe it is reachable by any process on the machine, so it is not
        acceptable as the final Windows transport -- see the note in
        _serve_pipe.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Deliberately NOT SO_REUSEADDR. On Windows it does not mean what it
        # means on POSIX: it permits a second socket to bind an address already
        # in use and take over its connections, which Microsoft's own guidance
        # warns against. Binding an ephemeral port gains nothing from it and
        # inherits the hazard -- under rapid server churn a new listener could
        # land on a port a previous one had not finished releasing, and
        # connections went to whichever socket won. That showed up as roughly
        # one request in ten silently getting no reply.
        # SO_EXCLUSIVEADDRUSE asks for the opposite guarantee.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            with contextlib.suppress(OSError):
                self._sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                )
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(self.backlog)
        self._sock.settimeout(0.25)
        Path(self._port_file()).write_text(
            str(self._sock.getsockname()[1]), encoding="ascii"
        )
        self._ready.set()
        self._accept_loop()

    def _accept_loop(self) -> None:
        """Accept connections until asked to stop.

        The error handling here is the whole point. An earlier version caught
        OSError and broke, which meant a single transient failure -- a
        BlockingIOError from a timeout-mode listener under burst, or a client
        vanishing mid-handshake -- silently killed the daemon for the rest of
        the session. It showed up as most of a dozen concurrent clients getting
        no answer, and only under load, which is exactly the shape of bug that
        reaches users and not developers.

        So transient errors continue, and only a genuinely dead listening
        socket ends the loop.
        """
        assert self._sock is not None
        consecutive_failures = 0

        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except (BlockingIOError, InterruptedError, ConnectionError):
                # Transient: the peer went away, or the listener had nothing
                # ready. Neither says anything about our ability to serve.
                continue
            except OSError as exc:
                if self._stop.is_set():
                    break
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    log.error("giving up on %s after repeated accept errors: %s",
                              self.endpoint, exc)
                    break
                # Do not spin hot on a persistent error.
                log.warning("accept failed (%d/10): %s", consecutive_failures, exc)
                time.sleep(0.05)
                continue

            consecutive_failures = 0
            threading.Thread(
                target=self._handle_socket, args=(conn,), daemon=True
            ).start()

    def _port_file(self) -> str:
        return port_file_for(self.endpoint)

    def _handle_socket(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(2.0)
            try:
                raw = _read_line(conn.recv, time.monotonic() + 2.0)
                if raw is None:
                    return
                try:
                    req = decode_request(raw)
                except ProtocolError as exc:
                    conn.sendall(encode(_bad_request(str(exc))))
                    return
                try:
                    reply = self.handler(req)
                except Exception as exc:
                    # A handler that raises must still produce a reply. Letting
                    # the exception escape leaves the client waiting out its
                    # whole timeout for a connection that will never answer --
                    # which, at the widget, is a frozen Tab. Answer immediately
                    # and let it fall back.
                    log.exception("handler raised; replying with an error")
                    reply = _internal_error(str(exc))
                conn.sendall(encode(reply))
            except OSError:
                return


def port_file_for(endpoint: str) -> str:
    """Companion file holding the loopback port, for the Windows stand-in.

    The PowerShell widget computes this independently; the two must agree
    exactly or it never finds the daemon. Keep the fallback order in step with
    Get-ClAiPortFile in cl-ai.psm1.
    """
    safe = "".join(c for c in endpoint if c.isalnum() or c in "-_")
    temp = os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
    return str(Path(temp) / f"{safe}.port")


def _bad_request(detail: str) -> Response:
    from .protocol import ErrorCode

    return Response(
        ok=False,
        error=ErrorCode.BAD_REQUEST,
        message=f"could not read the request: {detail}",
    )


def _internal_error(detail: str) -> Response:
    from .protocol import ErrorCode

    return Response(
        ok=False,
        error=ErrorCode.INTERNAL,
        message=f"the daemon failed to handle that: {detail}",
    )
