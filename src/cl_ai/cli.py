"""`cl-ai` entry point: fetch | index | init | doctor | suggest | daemon.

The install path is `fetch` -> `index` -> `init`, and `doctor` says which of
those has not happened yet.

WHY `doctor` IS THE CENTREPIECE
Every failure in this system is designed to degrade silently. No daemon means
ordinary Tab; no catalog means name completion; an unknown shell means
unbound keys; a version mismatch means the widget stands down. Each of those
is the right behaviour for someone mid-keystroke, and together they are
awful to diagnose: when Tab does nothing, there is no thread to pull.

`doctor` is where that silence is explained. It is the one command allowed to
be slow and verbose, it checks each layer independently so a failure names
itself, and it ends with a live end-to-end suggestion -- because every layer
can pass its own check while the whole still returns nothing.

It never raises. A diagnostic that crashes on a broken system is a diagnostic
that only works when it is not needed, so every probe is caught and reported
as a line of output.

EXIT CODES
    0  everything works
    1  degraded: usable, but something is off
    2  broken: the thing you asked for cannot be done
Machine-readable on purpose, so the install can be checked from a script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["main"]

OK = 0
DEGRADED = 1
BROKEN = 2

#: Written into a shell profile so `init --apply` can recognise its own work
#: and refuse to add it twice.
_MARKER = "# >>> cl-ai >>>"
_END_MARKER = "# <<< cl-ai <<<"


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

@dataclass
class Check:
    """One diagnostic line. `status` drives the exit code, not the wording."""

    label: str
    status: str            # "ok" | "warn" | "fail" | "info"
    detail: str = ""

    def render(self, colour: bool) -> str:
        mark = {"ok": "ok  ", "warn": "warn", "fail": "FAIL", "info": "    "}[
            self.status
        ]
        if colour and self.status in ("ok", "warn", "fail"):
            code = {"ok": "32", "warn": "33", "fail": "31"}[self.status]
            mark = f"\033[{code}m{mark}\033[0m"
        line = f"  {mark}  {self.label}"
        return f"{line}\n        {self.detail}" if self.detail else line


def _use_colour(stream: object) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def _probe(label: str, fn: Callable[[], Check]) -> Check:
    """Run one probe, converting any exception into a reported failure.

    The whole point of doctor is to work on a broken machine, so a probe that
    raises must become a line of output rather than a traceback that hides
    every check after it.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - see docstring
        return Check(label, "fail", f"{type(exc).__name__}: {exc}")


def _check_environment() -> Check:
    return Check(
        "environment",
        "info",
        f"python {sys.version.split()[0]} on {sys.platform} ({os.name})",
    )


def _check_shell() -> Check:
    from .platform_ import detect

    profile = detect()
    widget = _widget_path(profile.id)
    if widget is None:
        return Check(
            f"shell: {profile.id}",
            "warn",
            "no widget ships for this shell yet; only powershell is implemented",
        )
    return Check(f"shell: {profile.id}", "ok", str(widget))


def _check_inventory() -> Check:
    from .catalog.discovery import discover

    started = time.monotonic()
    inventory = discover()
    elapsed = (time.monotonic() - started) * 1000
    if not inventory.names:
        return Check("PATH scan", "fail", "found no runnable commands on PATH")
    return Check(
        "PATH scan",
        "ok",
        f"{len(inventory.names)} commands in {elapsed:.0f}ms"
        + (f", {len(inventory.skipped)} dirs skipped" if inventory.skipped else ""),
    )


def _check_sources() -> Check:
    from .catalog.build import default_sources

    available: list[str] = []
    missing: list[str] = []
    for source in default_sources():
        name = getattr(source, "name", type(source).__name__)
        try:
            (available if source.available() else missing).append(name)
        except Exception:  # noqa: BLE001
            missing.append(name)
    if not available:
        # The commonest state on a fresh machine, and the one that makes
        # everything else look broken: no corpus means no catalog means Tab
        # completes names and nothing else. Name the fix.
        return Check(
            "catalog sources",
            "fail",
            f"none available ({', '.join(missing) or 'no tiers'}). "
            "Run `cl-ai fetch` to download the tldr corpus, "
            "or set CL_AI_TLDR_ROOT to an existing checkout",
        )
    detail = ", ".join(available)
    if missing:
        detail += f" (unavailable: {', '.join(missing)})"
    return Check("catalog sources", "ok", detail)


def _check_catalog() -> Check:
    from .catalog.store import default_path, load

    path = default_path()
    found = load(path)
    if found is None:
        return Check(
            "catalog",
            "warn",
            f"not built yet at {path}; run `cl-ai index`",
        )
    try:
        age = (time.time() - path.stat().st_mtime) / 3600.0
        when = f", {age:.0f}h old" if age >= 1 else ", fresh"
    except OSError:
        when = ""
    return Check(
        "catalog",
        "ok",
        f"{len(found.catalog.tools)} tools from "
        f"{'+'.join(found.built_with) or 'unknown'}{when} at {path}",
    )


def _check_daemon() -> Check:
    from .daemon.protocol import Kind, Request
    from .daemon.transport import default_endpoint, request

    endpoint = default_endpoint()
    started = time.monotonic()
    reply = request(Request(kind=Kind.PING), endpoint, timeout_s=1.0)
    elapsed = (time.monotonic() - started) * 1000
    if reply is None:
        return Check(
            "daemon",
            "warn",
            f"not running at {endpoint}; start it with `cl-ai daemon start`",
        )
    if not reply.ok:
        return Check("daemon", "fail", f"{reply.error.value}: {reply.message}")
    return Check("daemon", "ok", f"responded in {elapsed:.0f}ms at {endpoint}")


def _check_embedder() -> Check:
    chosen = os.environ.get("CL_AI_EMBEDDER", "").strip()
    if not chosen:
        return Check(
            "embedder",
            "info",
            "off (default). Needle's embeddings measured hit@1 0.704 -> 0.444; "
            "see src/cl_ai/embedding/needle_embedder.py",
        )
    from .embedding import build_embedder

    built = build_embedder()
    if built is None:
        return Check(
            "embedder",
            "warn",
            f"CL_AI_EMBEDDER={chosen!r} but no usable model was found",
        )
    return Check("embedder", "warn", f"{chosen} enabled; measured to rank worse")


def _check_end_to_end(shell: str) -> list[Check]:
    """The check that matters: does a real query produce a real command?

    Every layer above can pass and this still return nothing, because the
    score floor is allowed to reject everything. Run last, and run through the
    daemon when there is one so it exercises the path the widget uses.
    """
    from .daemon.protocol import Kind, Request
    from .daemon.transport import default_endpoint, request

    checks: list[Check] = []
    probes = ("git com", "list files")

    reply = request(Request(kind=Kind.PING), default_endpoint(), timeout_s=1.0)
    via_daemon = reply is not None and reply.ok

    if via_daemon:
        for query in probes:
            answer = request(
                Request(buffer=query, shell=shell, limit=1),
                default_endpoint(),
                timeout_s=2.0,
            )
            checks.append(_end_to_end_check(query, answer, "daemon"))
        return checks

    # No daemon: do it in process, so `doctor` is still useful before the
    # first `daemon start`. Slower, and says so.
    engine = _Engine.build(shell)
    if engine is None:
        checks.append(
            Check("suggestion", "fail", "no daemon and no catalog to fall back on")
        )
        return checks
    for query in probes:
        suggestions = engine.suggest(query, limit=1)
        checks.append(
            Check(
                f"suggestion {query!r}",
                "ok" if suggestions else "warn",
                (suggestions[0] if suggestions else "no match")
                + "   (in-process; no daemon)",
            )
        )
    return checks


def _end_to_end_check(query: str, answer: object, via: str) -> Check:
    label = f"suggestion {query!r}"
    if answer is None:
        return Check(label, "fail", f"no reply from the {via}")
    ok = getattr(answer, "ok", False)
    if not ok:
        error = getattr(getattr(answer, "error", None), "value", "?")
        return Check(label, "fail", f"{error}: {getattr(answer, 'message', '')}")
    suggestions = getattr(answer, "suggestions", ())
    if not suggestions:
        return Check(label, "warn", f"no match (via {via})")
    elapsed = getattr(answer, "elapsed_ms", 0.0)
    return Check(label, "ok", f"{suggestions[0].command}   ({elapsed:.0f}ms via {via})")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .platform_ import detect

    colour = _use_colour(sys.stdout)
    shell = args.shell or detect().id

    checks = [
        _probe("environment", _check_environment),
        _probe("shell", _check_shell),
        _probe("PATH scan", _check_inventory),
        _probe("catalog sources", _check_sources),
        _probe("catalog", _check_catalog),
        _probe("daemon", _check_daemon),
        _probe("embedder", _check_embedder),
    ]
    try:
        checks.extend(_check_end_to_end(shell))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("suggestion", "fail", f"{type(exc).__name__}: {exc}"))

    if args.json:
        print(
            json.dumps(
                [{"label": c.label, "status": c.status, "detail": c.detail}
                 for c in checks],
                indent=2,
            )
        )
    else:
        print("cl-ai doctor\n")
        for check in checks:
            print(check.render(colour))
        print()

    if any(c.status == "fail" for c in checks):
        return BROKEN
    if any(c.status == "warn" for c in checks):
        return DEGRADED
    return OK


# --------------------------------------------------------------------------
# a daemon-free engine, shared by `suggest` and `doctor`
# --------------------------------------------------------------------------

class _Engine:
    """Retrieval without a daemon, for one-shot use.

    Exists so `cl-ai suggest` is scriptable and so `doctor` can answer the
    only question that matters before a daemon has ever been started. Slow --
    it scans PATH and builds a catalog every time -- which is exactly why the
    daemon exists and is not a reason to avoid having this.
    """

    def __init__(self, index: object, context: object) -> None:
        self._index = index
        self._context = context

    @classmethod
    def build(cls, shell: str, *, rebuild: bool = False) -> _Engine | None:
        from .catalog.build import build
        from .catalog.discovery import discover
        from .catalog.store import cache_key, load
        from .ir import ContextFacts
        from .retrieval.index import ToolIndex

        inventory = discover()
        catalog = None
        if not rebuild:
            found = load()
            key = cache_key(inventory.entries)
            # Reuse only a catalog built from THIS machine's PATH and this
            # schema. A stale one would suggest tools that are gone.
            if found is not None and found.is_valid_for(key):
                catalog = found.catalog
        if catalog is None:
            catalog = build(
                binaries=sorted(inventory.names), discovered=inventory.entries
            ).catalog
        if not catalog.tools:
            return None

        index = ToolIndex.from_catalog(catalog, shell=shell, os_name=sys.platform)
        context = ContextFacts(
            os=sys.platform, shell=shell, installed=inventory.names
        )
        return cls(index, context)

    def suggest(self, query: str, *, limit: int = 5) -> list[str]:
        from .retrieval.examples import command_for

        results = self._index.search(  # type: ignore[attr-defined]
            query, limit=limit, context=self._context
        )
        return [command_for(c.tool, query) for c in results]

    def detailed(self, query: str, *, limit: int = 5) -> list[tuple[str, str, bool]]:
        from .daemon.server import _looks_destructive
        from .retrieval.examples import best_example, command_for, is_destructive

        results = self._index.search(  # type: ignore[attr-defined]
            query, limit=limit, context=self._context
        )
        rows: list[tuple[str, str, bool]] = []
        for candidate in results:
            chosen = best_example(candidate.tool, query)
            rows.append(
                (
                    command_for(candidate.tool, query),
                    candidate.tool.description,
                    # Must agree with the daemon: the same query through two
                    # paths marking differently would be worse than either
                    # rule on its own.
                    candidate.dangerous
                    or _looks_destructive(candidate.tool.binary)
                    or (chosen is not None and is_destructive(chosen)),
                )
            )
        return rows


# --------------------------------------------------------------------------
# suggest
# --------------------------------------------------------------------------

def cmd_suggest(args: argparse.Namespace) -> int:
    from .daemon.protocol import Request
    from .daemon.transport import default_endpoint, request
    from .platform_ import detect

    query = " ".join(args.query).strip()
    if not query:
        print("nothing to suggest for an empty query", file=sys.stderr)
        return BROKEN
    shell = args.shell or detect().id

    rows: list[tuple[str, str, bool]] = []
    if not args.no_daemon:
        reply = request(
            Request(buffer=query, shell=shell, limit=args.limit),
            default_endpoint(),
            timeout_s=2.0,
        )
        if reply is not None and reply.ok:
            rows = [(s.command, s.description, s.dangerous) for s in reply.suggestions]
        elif reply is not None:
            print(f"daemon: {reply.message}", file=sys.stderr)

    if not rows:
        engine = _Engine.build(shell)
        if engine is None:
            print("no catalog; run `cl-ai index`", file=sys.stderr)
            return BROKEN
        rows = engine.detailed(query, limit=args.limit)

    if args.json:
        print(
            json.dumps(
                [{"command": c, "description": d, "dangerous": x} for c, d, x in rows],
                indent=2,
            )
        )
        return OK if rows else DEGRADED

    if not rows:
        print("(no match)")
        return DEGRADED
    for command, description, dangerous in rows:
        print(command + ("   # destructive" if dangerous else ""))
        if args.verbose and description:
            print(f"    {description}")
    return OK


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> int:
    from .catalog.build import build
    from .catalog.discovery import discover

    started = time.monotonic()
    print("scanning PATH...", end=" ", flush=True)
    inventory = discover()
    print(f"{len(inventory.names)} commands")

    print("building catalog...", end=" ", flush=True)
    result = build(
        binaries=None if args.all else sorted(inventory.names),
        discovered=inventory.entries,
    )
    print(result.summary())

    if result.skipped:
        print(f"  skipped tiers: {', '.join(result.skipped)}")
    if result.path:
        print(f"  written to {result.path}")
    print(f"  {time.monotonic() - started:.1f}s")

    if not result.catalog.tools:
        print("no tools were schematised; is a source available?", file=sys.stderr)
        return BROKEN
    return OK if result.ok else DEGRADED


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------

def cmd_daemon(args: argparse.Namespace) -> int:
    from .daemon.protocol import Kind, Request
    from .daemon.transport import default_endpoint, request

    endpoint = default_endpoint()

    if args.action == "status":
        reply = request(Request(kind=Kind.PING), endpoint, timeout_s=1.0)
        if reply is None:
            print(f"not running ({endpoint})")
            return DEGRADED
        print(f"running ({endpoint}), replied in {reply.elapsed_ms:.1f}ms")
        return OK

    if args.action == "stop":
        reply = request(Request(kind=Kind.SHUTDOWN), endpoint, timeout_s=2.0)
        if reply is None:
            print("not running")
            return DEGRADED
        # Wait for the endpoint to actually go away. The daemon answers the
        # shutdown request BEFORE it unbinds, so returning here would make
        # `stop; start` a race -- observed: start found the old daemon still
        # listening and reported "already running", leaving the user with the
        # process they had just asked to replace.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if request(Request(kind=Kind.PING), endpoint, timeout_s=0.3) is None:
                print("stopped")
                return OK
            time.sleep(0.1)
        print("asked it to stop, but it is still listening", file=sys.stderr)
        return BROKEN

    # start
    if request(Request(kind=Kind.PING), endpoint, timeout_s=0.5) is not None:
        print(f"already running ({endpoint})")
        return OK

    if args.foreground:
        from .daemon.server import main as daemon_main

        return daemon_main(["--foreground"])

    # Detached, so a shell profile can call this without blocking startup.
    #
    # The branch is written against sys.platform rather than os.name so that a
    # type checker narrows it too: CREATE_NO_WINDOW does not exist in the
    # POSIX typeshed stubs, and start_new_session does not exist on Windows.
    # transport.py does the same thing for the same reason.
    command = [sys.executable, "-m", "cl_ai.daemon.server"]
    if sys.platform == "win32":
        # Without these the child dies with the console that launched it,
        # which is precisely the console the user is about to keep using.
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS,
        )
    else:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if request(Request(kind=Kind.PING), endpoint, timeout_s=0.5) is not None:
            print(f"started ({endpoint})")
            return OK
        time.sleep(0.2)
    print("started, but it did not answer within 10s", file=sys.stderr)
    return BROKEN


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------

def cmd_fetch(args: argparse.Namespace) -> int:
    from .catalog.fetch import DEFAULT_URL, LICENCE, FetchError, default_target, fetch

    args.url = args.url or DEFAULT_URL
    target = Path(args.target) if args.target else default_target()
    # Printed before the request, not after: a user is entitled to know what
    # is about to be downloaded and where it will be written.
    print(f"downloading {args.url}")
    print(f"        to {target}")
    started = time.monotonic()
    try:
        result = fetch(args.url, target, force=args.force)
    except FetchError as exc:
        print(f"\nfetch failed: {exc}", file=sys.stderr)
        return BROKEN
    print(
        f"\n{result.pages} pages, {result.bytes_downloaded / 1_000_000:.1f}MB, "
        f"{time.monotonic() - started:.1f}s"
    )
    print(LICENCE)
    print("\nnow run `cl-ai index` to build the catalog")
    return OK


def _widget_path(shell_id: str) -> Path | None:
    """The shipped widget for a shell, or None if there is not one yet."""
    root = Path(__file__).parent / "shell"
    candidates = {
        "powershell": ("powershell", "cl-ai.psm1"),
        "pwsh": ("powershell", "cl-ai.psm1"),
        "bash": ("bash", "cl-ai.bash"),
        "zsh": ("zsh", "cl-ai.zsh"),
        "fish": ("fish", "cl-ai.fish"),
    }
    parts = candidates.get(shell_id)
    if parts is None:
        return None
    path = root.joinpath(*parts)
    return path if path.is_file() else None


def _profile_path(shell_id: str) -> Path | None:
    """Where that shell reads its startup file."""
    home = Path.home()
    if shell_id in ("powershell", "pwsh"):
        found = _powershell_profile(shell_id)
        if found is not None:
            return found
        base = "WindowsPowerShell" if shell_id == "powershell" else "PowerShell"
        return home / "Documents" / base / "Microsoft.PowerShell_profile.ps1"
    return {
        "bash": home / ".bashrc",
        "zsh": home / ".zshrc",
        "fish": home / ".config" / "fish" / "config.fish",
    }.get(shell_id)


def _powershell_profile(shell_id: str) -> Path | None:
    """Ask PowerShell itself, rather than guessing at OneDrive redirection.

    $PROFILE moves: OneDrive relocates Documents, and pwsh and Windows
    PowerShell use different directories. Guessing puts the line in a file
    that is never read, which looks exactly like the widget not working.
    """
    executable = shutil.which(shell_id) or shutil.which("pwsh")
    if executable is None:
        return None
    try:
        done = subprocess.run( 
            [executable, "-NoProfile", "-NonInteractive", "-Command", "$PROFILE"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.strip()
    return Path(value) if done.returncode == 0 and value else None


def _snippet(shell_id: str, widget: Path) -> str:
    """The lines that go in a shell profile.

    Starts the daemon as well as binding the keys. Without that, a new shell
    has no daemon and Tab is just Tab until the user runs `cl-ai daemon
    start` by hand -- which nobody will do every morning, and which makes
    the whole thing look broken.

    Backgrounded, and every failure swallowed. This runs on the startup path
    of an interactive shell: a slow or broken cl-ai must cost the user
    nothing more than the feature, never a hang or an error on every prompt.
    """
    if shell_id in ("powershell", "pwsh"):
        body = (
            f"Import-Module '{widget}'\n"
            "Register-ClAiKeyHandlers | Out-Null\n"
            "Start-Job { cl-ai daemon start } | Out-Null\n"
        )
    elif shell_id == "fish":
        body = f"source '{widget}'\ncl-ai daemon start >/dev/null 2>&1 &\ndisown\n"
    else:
        body = f'. "{widget}"\n(cl-ai daemon start >/dev/null 2>&1 &)\n'
    return f"{_MARKER}\n{body}{_END_MARKER}\n"


def cmd_init(args: argparse.Namespace) -> int:
    from .platform_ import detect

    shell = args.shell or detect().id
    widget = _widget_path(shell)
    if widget is None:
        print(
            f"no widget ships for {shell!r} yet. Implemented: powershell.",
            file=sys.stderr,
        )
        return BROKEN

    profile = _profile_path(shell)
    if profile is None:
        print(f"could not work out where {shell!r} reads its profile", file=sys.stderr)
        return BROKEN

    snippet = _snippet(shell, widget)
    if not args.apply:
        # Printing is the default because this edits a file the user did not
        # name, in a shell they are about to depend on. Showing the exact two
        # lines and the exact path costs one paste and removes all of the
        # surprise.
        print(f"# add to {profile}\n")
        print(snippet, end="")
        print("# or run `cl-ai init --apply` to append it for you")
        return OK

    try:
        existing = profile.read_text(encoding="utf-8") if profile.exists() else ""
    except OSError as exc:
        print(f"cannot read {profile}: {exc}", file=sys.stderr)
        return BROKEN
    if _MARKER in existing:
        print(f"already installed in {profile}")
        return OK

    try:
        profile.parent.mkdir(parents=True, exist_ok=True)
        separator = "" if not existing or existing.endswith("\n") else "\n"
        with profile.open("a", encoding="utf-8") as handle:
            handle.write(f"{separator}\n{snippet}")
    except OSError as exc:
        print(f"cannot write {profile}: {exc}", file=sys.stderr)
        return BROKEN

    print(f"installed in {profile}")
    print("open a new shell, then press Tab after typing what you want")
    return OK


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cl-ai",
        description="Natural language at your shell prompt. Nothing is ever run for you.",
    )
    parser.add_argument("--shell", default=None, help="override the detected shell")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="diagnose the installation")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    fetch = sub.add_parser("fetch", help="download the tldr command corpus")
    fetch.add_argument("--url", default=None, help="archive to download")
    fetch.add_argument("--target", default=None, help="where to unpack it")
    fetch.add_argument(
        "--force", action="store_true", help="replace an existing corpus"
    )
    fetch.set_defaults(func=cmd_fetch)

    index = sub.add_parser("index", help="build the command catalog")
    index.add_argument(
        "--all",
        action="store_true",
        help="schematise every documented tool, not only what is installed",
    )
    index.set_defaults(func=cmd_index)

    suggest = sub.add_parser("suggest", help="one-shot suggestion, printed")
    suggest.add_argument("query", nargs="*")
    suggest.add_argument("-n", "--limit", type=int, default=5)
    suggest.add_argument("-v", "--verbose", action="store_true")
    suggest.add_argument("--json", action="store_true")
    suggest.add_argument(
        "--no-daemon", action="store_true", help="always answer in process"
    )
    suggest.set_defaults(func=cmd_suggest)

    daemon = sub.add_parser("daemon", help="start, stop or query the daemon")
    daemon.add_argument("action", choices=("start", "stop", "status"))
    daemon.add_argument("--foreground", action="store_true")
    daemon.set_defaults(func=cmd_daemon)

    init = sub.add_parser("init", help="install the shell widget")
    init.add_argument(
        "--apply", action="store_true", help="append to the profile instead of printing"
    )
    init.set_defaults(func=cmd_init)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return OK
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
