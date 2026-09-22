"""The resident daemon.

Measured justification, not a preference: a per-invocation process costs 579ms
before it answers anything, while a warm one embeds in 9.7ms and completes in
68ms. Tab has to land under ~100ms to read as completion rather than as a
network call, so the process has to already exist.

Current scope is deliberately narrow. This serves suggestions from the catalog
inventory -- enough to prove the interaction end to end and to measure real
latency through a real transport -- and does not yet involve a model. The
planner, retrieval and renderers plug in behind the same handler.

Two rules the daemon is built around:

  * Never exceed the caller's deadline. A partial answer beats a late one,
    because a late one arrives after the user has already typed on.
  * Never let an exception reach the socket loop. A crashed handler must
    degrade to "no suggestion", which the widget renders as ordinary Tab.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from ..catalog.discovery import Inventory, discover
from ..platform_ import PROFILES
from .protocol import ErrorCode, Kind, Request, Response, Suggestion, version_mismatch
from .transport import Server, default_endpoint

log = logging.getLogger("cl_ai.daemon")

#: Leave headroom inside the caller's budget for encoding and the socket hop.
_OVERHEAD_MS = 15


class Daemon:
    """Holds the warm state and answers requests."""

    def __init__(self) -> None:
        self._inventory: Inventory | None = None
        self._lock = threading.Lock()
        self._building = threading.Event()

    # -- warm state -------------------------------------------------------

    def warm(self, background: bool = True) -> None:
        """Build the inventory.

        In the background by default: on an unfamiliar machine the first scan
        should never be something the user waits behind. Until it lands,
        requests are answered with UNAVAILABLE rather than being queued, so a
        keypress is never blocked on a cold start.
        """
        if background:
            threading.Thread(target=self._build, name="cl-aid-warm",
                             daemon=True).start()
        else:
            self._build()

    def _build(self) -> None:
        self._building.set()
        try:
            inventory = discover()
            with self._lock:
                self._inventory = inventory
            log.info("inventory ready: %d commands in %.2fs",
                     len(inventory.entries), inventory.duration_s)
        except Exception:
            log.exception("inventory build failed; serving nothing")
        finally:
            self._building.clear()

    @property
    def inventory(self) -> Inventory | None:
        with self._lock:
            return self._inventory

    # -- request handling -------------------------------------------------

    def handle(self, request: Request) -> Response:
        """Total function: every path returns a Response, none raise."""
        started = time.monotonic()

        mismatch = version_mismatch(request)
        if mismatch is not None:
            return mismatch

        try:
            if request.kind is Kind.PING:
                return self._finish(Response(), started)
            if request.kind is Kind.SHUTDOWN:
                return self._finish(Response(message="shutting down"), started)
            return self._finish(self._suggest(request, started), started)
        except Exception as exc:
            log.exception("handler failed")
            return self._finish(
                Response(ok=False, error=ErrorCode.INTERNAL,
                         message=f"internal error: {exc}"),
                started,
            )

    def _suggest(self, request: Request, started: float) -> Response:
        inventory = self.inventory
        if inventory is None:
            return Response(
                ok=False,
                error=ErrorCode.UNAVAILABLE,
                message="still indexing the commands on this machine",
            )

        query = request.buffer.strip()
        if not query:
            return Response()

        budget = (request.deadline_ms - _OVERHEAD_MS) / 1000.0
        profile = PROFILES.get(request.shell)
        suggestions: list[Suggestion] = []

        # Placeholder ranking: prefix then substring over installed commands.
        # Retrieval replaces this wholesale; what it proves today is the shape
        # of the loop and the latency through a real transport.
        needle = query.lower()
        prefix, contains = [], []
        for entry in inventory.usable:
            if time.monotonic() - started > budget:
                break                          # partial beats late
            name = entry.name.lower()
            if name.startswith(needle):
                prefix.append(entry)
            elif needle in name:
                contains.append(entry)

        for entry in (*prefix, *contains):
            if len(suggestions) >= max(1, request.limit):
                break
            suggestions.append(
                Suggestion(
                    command=entry.name,
                    description=entry.kind.value,
                    source=entry.source,
                    dangerous=_looks_destructive(entry.name),
                )
            )

        if profile is None and request.shell:
            log.debug("unknown shell %r from widget", request.shell)

        return Response(suggestions=tuple(suggestions))

    @staticmethod
    def _finish(response: Response, started: float) -> Response:
        from dataclasses import replace

        return replace(response, elapsed_ms=(time.monotonic() - started) * 1000)


#: Marked in the UI because the suggestion lands in the buffer ready to run and
#: Enter is one keystroke away. Replaced by the catalog's capability tags once
#: stage 2 exists; a crude list now is better than nothing at all.
_DESTRUCTIVE = frozenset({
    "rm", "rmdir", "del", "erase", "format", "mkfs", "dd", "shred",
    "kill", "killall", "taskkill", "shutdown", "reboot", "halt",
    "fdisk", "diskpart", "truncate", "chown", "chmod",
})


def _looks_destructive(name: str) -> bool:
    return name.lower() in _DESTRUCTIVE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cl-aid", description="cl-ai daemon")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--foreground", action="store_true",
                        help="build the inventory before serving")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    daemon = Daemon()
    daemon.warm(background=not args.foreground)

    endpoint = args.endpoint or default_endpoint()
    server = Server(daemon.handle, endpoint)
    server.start()
    log.info("listening on %s", endpoint)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
