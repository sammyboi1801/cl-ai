"""The resident daemon.

Measured justification, not a preference: a per-invocation process costs 579ms
before it answers anything, while a warm one embeds in 9.7ms and completes in
68ms. Tab has to land under ~100ms to read as completion rather than as a
network call, so the process has to already exist.

Current scope: real retrieval over the built catalog, with no model. A query
is ranked by `ToolIndex`, the example that best answers it is chosen, and that
line goes in the buffer. The planner plugs in behind the same handler to
improve ARGUMENTS; it is not needed to produce a command, because the
extractor already substituted placeholders with concrete legal values.

WARM STATE IS TWO THINGS, NOT ONE
The inventory (what is on PATH) and the catalog (what those commands can do)
have different costs and different lifetimes, so they are built separately
and the request path tolerates having either one alone. An index is then
per-(os, shell) and memoised on first use, because variant resolution happens
at index build time -- one machine can run both pwsh and bash, and the
Windows `dir` must not be offered to the latter.

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

from ..catalog.build import build
from ..catalog.discovery import Inventory, discover
from ..catalog.normalize import Catalog
from ..ir import ContextFacts
from ..platform_ import PROFILES
from ..retrieval.examples import command_for
from ..retrieval.index import ToolIndex
from .protocol import ErrorCode, Kind, Request, Response, Suggestion, version_mismatch
from .transport import Server, default_endpoint

log = logging.getLogger("cl_ai.daemon")

#: Leave headroom inside the caller's budget for encoding and the socket hop.
_OVERHEAD_MS = 15


class Daemon:
    """Holds the warm state and answers requests."""

    def __init__(self) -> None:
        self._inventory: Inventory | None = None
        self._catalog: Catalog | None = None
        #: One index per (os, shell). Built on demand: a machine typically
        #: sees one shell, so building every variant up front would be work
        #: for nobody.
        self._indexes: dict[tuple[str, str], ToolIndex] = {}
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
            try:
                inventory = discover()
            except Exception:
                log.exception("inventory build failed; serving nothing")
                return
            with self._lock:
                self._inventory = inventory
            log.info("inventory ready: %d commands in %.2fs",
                     len(inventory.entries), inventory.duration_s)
            # After the inventory is published, not before: the catalog takes
            # seconds, and name completion should be available for all of them.
            self._build_catalog(inventory)
        finally:
            self._building.clear()

    def _build_catalog(self, inventory: Inventory) -> None:
        """Schematise what is on PATH.

        Separate from the inventory scan and separately guarded: a catalog
        that fails to build costs ranked suggestions, not the daemon. There
        is no useful state between "we have a catalog" and "we do not", so
        the failure just leaves `_catalog` as None and the request path
        already handles that.
        """
        started = time.monotonic()
        try:
            result = build(binaries=sorted(inventory.names), discovered=inventory.entries)
        except Exception:
            log.exception("catalog build failed; suggestions will be unranked")
            return
        with self._lock:
            self._catalog = result.catalog
            self._indexes.clear()
        log.info(
            "catalog ready: %d tools from %s in %.2fs",
            len(result.catalog.tools),
            "+".join(result.used) or "no sources",
            time.monotonic() - started,
        )

    @property
    def inventory(self) -> Inventory | None:
        with self._lock:
            return self._inventory

    @property
    def catalog(self) -> Catalog | None:
        with self._lock:
            return self._catalog

    def index_for(self, shell: str, os_name: str) -> ToolIndex | None:
        """The index for one target, built once and reused.

        The build is done OUTSIDE the lock. It takes ~1s on a real catalog,
        and holding the lock across it would stall every other keystroke
        behind the first one; two threads racing here waste one build and
        then agree, which is much cheaper than serialising them.
        """
        key = (os_name, shell)
        with self._lock:
            catalog = self._catalog
            existing = self._indexes.get(key)
        if existing is not None or catalog is None:
            return existing

        try:
            built = ToolIndex.from_catalog(catalog, shell=shell, os_name=os_name)
        except Exception:
            log.exception("index build failed for %s/%s", os_name, shell)
            return None

        with self._lock:
            return self._indexes.setdefault(key, built)

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
        if profile is None and request.shell:
            log.debug("unknown shell %r from widget", request.shell)

        limit = max(1, request.limit)
        index = self.index_for(request.shell, sys.platform)
        if index is None:
            # The catalog is still building, or failed. Name completion over
            # PATH is a genuinely useful fraction of what Tab is for, so it
            # is what we fall back to rather than returning nothing.
            return Response(
                suggestions=self._by_name(inventory, query, limit, started, budget)
            )

        context = ContextFacts(
            os=sys.platform,
            shell=request.shell,
            cwd=request.cwd,
            installed=inventory.names,
        )
        results = index.search(query, limit=limit, context=context)
        if not results:
            # An honest empty answer, not a fallback. The score floor fired,
            # which means nothing in the catalog matches -- and substituting a
            # name that merely shares a substring is how a missing tool turns
            # into a confidently wrong command. Name completion still runs
            # when the buffer is a plain prefix, because that is not a guess.
            return Response(
                suggestions=self._by_name(inventory, query, limit, started, budget)
            )

        suggestions: list[Suggestion] = []
        for candidate in results:
            if suggestions and time.monotonic() - started > budget:
                break                          # partial beats late
            tool = candidate.tool
            suggestions.append(
                Suggestion(
                    command=command_for(tool, query),
                    description=tool.description,
                    source=_source_of(tool),
                    # The catalog's judgement, widened by the static list. A
                    # missed marking is the direction that costs a user their
                    # files, so the two are unioned rather than ranked.
                    dangerous=candidate.dangerous or _looks_destructive(tool.binary),
                )
            )
        return Response(suggestions=tuple(suggestions))

    @staticmethod
    def _by_name(
        inventory: Inventory,
        query: str,
        limit: int,
        started: float,
        budget: float,
    ) -> tuple[Suggestion, ...]:
        """Prefix-then-substring completion over PATH, with no catalog.

        This was the whole of `_suggest` before retrieval existed. It survives
        as the degraded path because it needs nothing but the inventory, and
        because on a cold start it is the difference between a Tab key that
        does less and a Tab key that does nothing.
        """
        needle = query.lower()
        if not needle:
            return ()
        prefix: list[str] = []
        contains: list[str] = []
        for entry in inventory.usable:
            if time.monotonic() - started > budget:
                break
            name = entry.name.lower()
            if name.startswith(needle):
                prefix.append(entry.name)
            elif needle in name:
                contains.append(entry.name)

        return tuple(
            Suggestion(
                command=name,
                description="command on PATH",
                source="inventory",
                dangerous=_looks_destructive(name),
            )
            for name in (*prefix, *contains)[:limit]
        )

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


def _source_of(tool: object) -> str:
    """Which tier schematised this tool, for "where did this come from?".

    A user who does not trust a suggestion is owed an answer, and a tool
    without provenance should say nothing rather than claim a source.
    """
    provenance = getattr(tool, "provenance", None)
    source = getattr(provenance, "source", "")
    return str(source) if source else ""


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
