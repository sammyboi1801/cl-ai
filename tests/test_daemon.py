"""Daemon and transport tests.

The requirement that dominates this component is not correctness of the happy
path but invisibility of every failure: if the daemon is missing, slow, wedged
or speaking another protocol version, Tab must behave like ordinary Tab. A
completion that hangs a terminal is worse than one that never appears.

So most of what follows deliberately breaks things.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from cl_ai.catalog.discovery import Discovered, EntryKind, Inventory
from cl_ai.daemon import transport
from cl_ai.daemon.protocol import (
    PROTOCOL_VERSION,
    ErrorCode,
    Kind,
    Request,
    Response,
    Suggestion,
)
from cl_ai.daemon.server import Daemon
from cl_ai.daemon.transport import Server, default_endpoint, request

WINDOWS = os.name == "nt"


@pytest.fixture
def endpoint(tmp_path):
    """A private endpoint per test, so tests never collide with each other or
    with a daemon the developer happens to be running."""
    if WINDOWS:
        return str(tmp_path / f"cl-ai-test-{os.getpid()}-{threading.get_ident()}")
    return str(tmp_path / "s.sock")


def serve(handler, endpoint) -> Server:
    server = Server(handler, endpoint)
    server.start()
    return server


def fake_inventory(*names: str) -> Inventory:
    return Inventory(entries=tuple(
        Discovered(name=n, kind=EntryKind.EXECUTABLE, path=f"/usr/bin/{n}",
                   source="path:/usr/bin")
        for n in names
    ))


# ------------------------------------------------------------ the happy path

def test_round_trip_through_a_real_socket(endpoint):
    def handler(req: Request) -> Response:
        return Response(suggestions=(Suggestion(f"echo {req.buffer}"),))

    with serve(handler, endpoint):
        reply = request(Request(buffer="hello"), endpoint, timeout_s=5)
    assert reply is not None
    assert reply.suggestions[0].command == "echo hello"


def test_several_requests_on_one_server(endpoint):
    with serve(lambda r: Response(suggestions=(Suggestion(r.buffer),)), endpoint):
        for i in range(10):
            reply = request(Request(buffer=str(i)), endpoint, timeout_s=5)
            assert reply.suggestions[0].command == str(i)


def test_concurrent_clients_do_not_interleave(endpoint):
    """One request per connection, so replies must not cross wires."""
    with serve(lambda r: Response(suggestions=(Suggestion(r.buffer),)), endpoint):
        results: dict[int, str] = {}

        def ask(n: int) -> None:
            reply = request(Request(buffer=f"q{n}"), endpoint, timeout_s=10)
            if reply and reply.suggestions:
                results[n] = reply.suggestions[0].command

        threads = [threading.Thread(target=ask, args=(i,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

    expected = {i: f"q{i}" for i in range(12)}
    assert results == expected, (
        f"missing={sorted(set(expected) - set(results))} "
        f"wrong={ {k: v for k, v in results.items() if expected.get(k) != v} }"
    )


# --------------------------------------------------- every failure is silent

def test_no_daemon_returns_none_rather_than_raising(endpoint):
    """The commonest case in the wild: the user has not started it."""
    assert request(Request(buffer="x"), endpoint, timeout_s=0.3) is None


def test_a_slow_daemon_times_out_instead_of_hanging(endpoint):
    def molasses(_req: Request) -> Response:
        time.sleep(5)
        return Response()

    with serve(molasses, endpoint):
        started = time.monotonic()
        reply = request(Request(buffer="x"), endpoint, timeout_s=0.4)
        elapsed = time.monotonic() - started

    assert reply is None
    assert elapsed < 2.0, f"client blocked for {elapsed:.1f}s; Tab would freeze"


def test_a_crashing_handler_does_not_kill_the_server(endpoint):
    calls = {"n": 0}

    def flaky(req: Request) -> Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return Response(suggestions=(Suggestion("recovered"),))

    with serve(flaky, endpoint):
        first = request(Request(buffer="a"), endpoint, timeout_s=2)
        second = request(Request(buffer="b"), endpoint, timeout_s=5)

    # The client gets an immediate structured error rather than waiting out
    # its timeout on a connection that will never answer -- at the widget, a
    # silent hang is a frozen Tab.
    assert first is not None
    assert first.ok is False
    assert first.error is ErrorCode.INTERNAL
    assert second is not None                 # the server survived
    assert second.suggestions[0].command == "recovered"


def test_garbage_on_the_wire_gets_a_structured_refusal(endpoint):
    """A shell sending nonsense must get an answer, not a dropped connection:
    a dangling socket is what makes a widget hang."""
    import socket

    with serve(lambda r: Response(), endpoint):
        if WINDOWS:
            port_file = Path(transport.port_file_for(endpoint))
            port = int(port_file.read_text(encoding="ascii").strip())
            sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(endpoint)
        with sock:
            sock.sendall(b"this is not json\n")
            raw = sock.recv(4096)

    from cl_ai.daemon.protocol import decode_response
    reply = decode_response(raw)
    assert reply.ok is False
    assert reply.error is ErrorCode.BAD_REQUEST


def test_version_mismatch_is_reported_over_the_wire(endpoint):
    daemon = Daemon()
    daemon._inventory = fake_inventory("git")
    with serve(daemon.handle, endpoint):
        reply = request(
            Request(buffer="g", version=PROTOCOL_VERSION + 99), endpoint, timeout_s=5
        )
    assert reply.ok is False
    assert reply.error is ErrorCode.VERSION_MISMATCH


def test_server_stops_cleanly_and_releases_the_endpoint(endpoint):
    server = serve(lambda r: Response(), endpoint)
    assert request(Request(kind=Kind.PING), endpoint, timeout_s=5) is not None
    server.stop()
    assert request(Request(kind=Kind.PING), endpoint, timeout_s=0.3) is None
    if not WINDOWS:
        assert not os.path.exists(endpoint), "stale socket left behind"


@pytest.mark.skipif(WINDOWS, reason="POSIX permissions")
def test_socket_is_private_to_the_user(endpoint):
    """It carries the user's command line. On a shared machine the default
    umask is not sufficient protection."""
    with serve(lambda r: Response(), endpoint):
        mode = os.stat(endpoint).st_mode & 0o777
    assert mode == 0o600, f"socket mode {oct(mode)} is readable by others"


def test_endpoint_is_per_user():
    a = default_endpoint("alice")
    b = default_endpoint("bob")
    assert a != b
    assert "alice" in a and "bob" in b


# ---------------------------------------------------------- daemon behaviour

def test_unavailable_before_the_inventory_is_ready():
    """A cold start must answer immediately rather than queue a keypress."""
    reply = Daemon().handle(Request(buffer="gi"))
    assert reply.ok is False
    assert reply.error is ErrorCode.UNAVAILABLE
    assert "indexing" in reply.message


def test_suggestions_prefer_a_prefix_match():
    daemon = Daemon()
    daemon._inventory = fake_inventory("agit", "git", "github", "zgit")
    reply = daemon.handle(Request(buffer="git", limit=5))
    commands = [s.command for s in reply.suggestions]
    assert commands[0] == "git"
    assert commands.index("github") < commands.index("agit")


def test_limit_is_respected():
    daemon = Daemon()
    daemon._inventory = fake_inventory(*[f"git{i}" for i in range(50)])
    assert len(daemon.handle(Request(buffer="git", limit=3)).suggestions) == 3


def test_empty_buffer_yields_nothing_quietly():
    daemon = Daemon()
    daemon._inventory = fake_inventory("git")
    reply = daemon.handle(Request(buffer="   "))
    assert reply.ok is True
    assert reply.suggestions == ()


def test_destructive_commands_are_flagged():
    """The suggestion lands in the buffer and Enter is one keystroke away."""
    daemon = Daemon()
    daemon._inventory = fake_inventory("rm", "ls")
    flags = {s.command: s.dangerous for s in
             daemon.handle(Request(buffer="", limit=9)).suggestions}
    flags.update({s.command: s.dangerous for s in
                  daemon.handle(Request(buffer="rm", limit=9)).suggestions})
    assert flags.get("rm") is True


def test_handler_never_raises_whatever_the_inventory_does():
    """Total function: the socket loop must never see an exception."""
    daemon = Daemon()

    class Exploding:
        @property
        def usable(self):
            raise RuntimeError("inventory is on fire")

    daemon._inventory = Exploding()  # type: ignore[assignment]
    reply = daemon.handle(Request(buffer="x"))
    assert reply.ok is False
    assert reply.error is ErrorCode.INTERNAL


def test_deadline_is_respected_on_a_large_inventory():
    """Partial beats late: the user has typed on by the time a slow answer
    arrives."""
    daemon = Daemon()
    daemon._inventory = fake_inventory(*[f"cmd{i:05d}" for i in range(200_000)])
    started = time.monotonic()
    reply = daemon.handle(Request(buffer="cmd", limit=5, deadline_ms=60))
    elapsed = (time.monotonic() - started) * 1000
    assert reply.ok is True
    assert elapsed < 400, f"took {elapsed:.0f}ms against a 60ms budget"


def test_ping_works_without_an_inventory():
    reply = Daemon().handle(Request(kind=Kind.PING))
    assert reply.ok is True


def test_elapsed_is_reported():
    daemon = Daemon()
    daemon._inventory = fake_inventory("git")
    assert daemon.handle(Request(buffer="git")).elapsed_ms >= 0


def test_warm_builds_a_real_inventory():
    daemon = Daemon()
    daemon.warm(background=False)
    assert daemon.inventory is not None
    assert daemon.inventory.entries, "found no commands on the real PATH"


def test_warm_failure_leaves_the_daemon_serving(monkeypatch):
    """A broken scan must degrade to UNAVAILABLE, not take the process down."""
    import cl_ai.daemon.server as server_mod

    monkeypatch.setattr(server_mod, "discover",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    daemon = Daemon()
    daemon.warm(background=False)
    assert daemon.inventory is None
    assert daemon.handle(Request(buffer="x")).error is ErrorCode.UNAVAILABLE


# ------------------------------------------------------------------ latency

def test_end_to_end_latency_is_within_the_tab_budget(endpoint):
    """The number the whole architecture rests on.

    Tab has to land under ~100ms to read as completion rather than as a
    network call. This measures a real socket round trip against a real
    inventory -- the part a process-per-invocation design could never reach,
    since spawning one costs 579ms before it answers anything.
    """
    daemon = Daemon()
    daemon.warm(background=False)

    with serve(daemon.handle, endpoint):
        request(Request(buffer="g"), endpoint, timeout_s=5)      # warm the path
        timings = []
        for _ in range(20):
            started = time.perf_counter()
            reply = request(Request(buffer="gi"), endpoint, timeout_s=5)
            timings.append((time.perf_counter() - started) * 1000)
            assert reply is not None

    timings.sort()
    median, worst = timings[len(timings) // 2], timings[-1]
    print(f"\n  round trip: median {median:.1f}ms  worst {worst:.1f}ms")
    assert median < 100, f"median {median:.1f}ms exceeds the Tab budget"
