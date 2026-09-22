"""Stage 1 of the catalog pipeline: what commands exist on this machine.

Open (whatever is actually installed, not a curated list) and exec-free
(listing and stat-ing files, never running them). Discovery answers only
"does this exist and how would you invoke it" -- extracting a schema for it is
stage 2's job, and the two are separate outputs on purpose.

A command discovered WITHOUT a schema is a first-class result, not a failure.
It is what lets the system say "I know `git clone` exists but have no arguments
for it" instead of substituting the nearest neighbour it does have -- the
failure that turns a missing catalog entry into a confidently wrong command.

Cross-platform notes, each of which is a trap that bites in practice:

* Windows executability is decided by PATHEXT, not a permission bit. `git` on
  PATH is really `git.exe`, and the invocable name drops the extension.
* `os.access(X_OK)` is meaningless on Windows -- it reports success for any
  readable file -- so it is only consulted on POSIX.
* %LOCALAPPDATA%\\Microsoft\\WindowsApps is on PATH by default and holds zero
  byte App Execution Aliases: NTFS reparse points that look like executables
  and actually open the Microsoft Store. `bash.exe` there is the WSL launcher
  that, with no distribution installed, runs nothing at all -- the exact stub
  that broke this project's Windows CI. They are reported, and excluded from
  the usable set by default.
* PATH is routinely broken: this machine has 54 entries, 7 pointing at
  directories that do not exist and 7 duplicated. A scan must survive all of
  it and say what it skipped.
* An empty PATH entry means "current directory" on Windows, which is a
  well-known hijack vector. Always skipped, always reported.
"""

from __future__ import annotations

import os
import stat as stat_module
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ..platform_ import ShellProfile, detect

#: Reparse tag identifying a Windows App Execution Alias. Present in `stat`
#: since 3.8; defined defensively so this module imports anywhere.
_APPEXECLINK = getattr(stat_module, "IO_REPARSE_TAG_APPEXECLINK", 0x8000001B)

#: Shell builtins have no file on disk, so a PATH scan cannot see them, and
#: enumerating them for real (`compgen -b`, `builtin -n`) means running a
#: shell. These lists are curated so discovery can stay exec-free -- a
#: deliberate trade of completeness for not executing anything.
_BUILTINS: Mapping[str, frozenset[str]] = MappingProxyType({
    "posix": frozenset({
        "alias", "bg", "bind", "break", "builtin", "cd", "command", "continue",
        "declare", "dirs", "echo", "eval", "exec", "exit", "export", "false",
        "fg", "getopts", "hash", "help", "history", "jobs", "kill", "let",
        "local", "logout", "popd", "printf", "pushd", "pwd", "read",
        "readonly", "return", "set", "shift", "shopt", "source", "test",
        "times", "trap", "true", "type", "typeset", "ulimit", "umask",
        "unalias", "unset", "wait",
    }),
    "fish": frozenset({
        "and", "argparse", "begin", "bg", "bind", "block", "break", "builtin",
        "case", "cd", "command", "commandline", "complete", "contains",
        "continue", "count", "echo", "else", "emit", "end", "eval", "exec",
        "exit", "false", "fg", "for", "function", "functions", "history",
        "if", "jobs", "math", "not", "or", "printf", "pwd", "read", "return",
        "set", "source", "status", "string", "switch", "test", "true", "type",
        "ulimit", "wait", "while",
    }),
    # PowerShell has hundreds of cmdlets and enumerating them means running
    # Get-Command. This is the core set a CLI assistant plausibly needs; the
    # long tail arrives via stage 2 when it is allowed to execute.
    "powershell": frozenset({
        "Get-ChildItem", "Get-Content", "Get-Command", "Get-Help",
        "Get-Location", "Get-Process", "Get-Service", "Set-Location",
        "Copy-Item", "Move-Item", "Remove-Item", "New-Item", "Rename-Item",
        "Select-String", "Select-Object", "Where-Object", "ForEach-Object",
        "Sort-Object", "Measure-Object", "Start-Process", "Stop-Process",
        "Invoke-WebRequest", "Invoke-RestMethod", "Test-Path", "Out-File",
        "Write-Output", "Write-Host", "Get-Item", "Set-Content",
        "Add-Content", "Compress-Archive", "Expand-Archive",
    }),
    "cmd": frozenset({
        "assoc", "call", "cd", "chdir", "cls", "color", "copy", "date", "del",
        "dir", "echo", "endlocal", "erase", "exit", "for", "ftype", "goto",
        "if", "md", "mkdir", "mklink", "move", "path", "pause", "popd",
        "prompt", "pushd", "rd", "rem", "ren", "rename", "rmdir", "set",
        "setlocal", "shift", "start", "time", "title", "type", "ver", "vol",
    }),
})


class EntryKind(str, Enum):
    EXECUTABLE = "executable"   # a real file on PATH
    BUILTIN = "builtin"         # provided by the shell, no file exists
    STUB = "stub"               # looks executable, runs nothing (see SkipReason)


class SkipReason(str, Enum):
    EMPTY_PATH_ENTRY = "empty_path_entry"   # means CWD on Windows: a hijack vector
    MISSING = "missing"
    NOT_A_DIRECTORY = "not_a_directory"
    PERMISSION_DENIED = "permission_denied"
    DUPLICATE = "duplicate"
    UNREADABLE = "unreadable"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class Discovered:
    """One invocable name."""

    name: str                       # as typed: "git", not "git.exe"
    kind: EntryKind
    path: str | None = None         # None for builtins
    source: str = ""                # "path:<dir>" or "builtin:<family>"
    #: Same name found later in PATH and therefore shadowed. Kept because
    #: "which python did it pick" is a question users genuinely ask.
    shadowed_by_order: tuple[str, ...] = ()
    #: Cheap identity for cache keying: size and mtime, never a version string.
    #: Reading a version would mean executing the binary, and identity also
    #: catches a rebuilt binary whose version number did not change.
    size: int | None = None
    mtime_ns: int | None = None

    @property
    def usable(self) -> bool:
        return self.kind is not EntryKind.STUB


@dataclass(frozen=True)
class SkippedDir:
    path: str
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True)
class Inventory:
    """The result of a scan. Deterministic, so two runs diff cleanly."""

    entries: tuple[Discovered, ...] = ()
    skipped: tuple[SkippedDir, ...] = ()
    shell: str = ""
    duration_s: float = 0.0
    truncated: bool = False

    @property
    def usable(self) -> tuple[Discovered, ...]:
        return tuple(e for e in self.entries if e.usable)

    @property
    def names(self) -> frozenset[str]:
        """The hard gate the context engine applies: never suggest what is
        not installed."""
        return frozenset(e.name for e in self.entries if e.usable)

    def get(self, name: str) -> Discovered | None:
        key = name.lower() if os.name == "nt" else name
        for entry in self.entries:
            candidate = entry.name.lower() if os.name == "nt" else entry.name
            if candidate == key:
                return entry
        return None


def _pathext(env: Mapping[str, str]) -> tuple[str, ...]:
    """Extensions Windows considers executable, lowercased, in PATH order."""
    raw = env.get("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    return tuple(e.lower() for e in raw.split(os.pathsep) if e.strip())


def _is_app_exec_alias(path: str) -> bool:
    """A Windows App Execution Alias: zero-byte reparse point, not a program.

    Detected by reparse tag rather than by size. Size alone would also catch
    legitimately empty files, and the tag is what the OS itself keys on.
    """
    if os.name != "nt":
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return getattr(st, "st_reparse_tag", 0) == _APPEXECLINK


def _invocable_name(filename: str, pathext: Sequence[str]) -> str | None:
    """The name a user would type, or None if this file is not executable.

    On Windows that means stripping a PATHEXT extension; on POSIX it means
    holding the execute bit.
    """
    if os.name == "nt":
        lowered = filename.lower()
        for ext in pathext:
            if lowered.endswith(ext):
                return filename[: -len(ext)]
        return None
    return filename


def iter_path_dirs(
    env: Mapping[str, str] | None = None,
    extra: Iterable[str] = (),
) -> Iterator[tuple[str, SkipReason | None, str]]:
    """Yield (directory, skip_reason, detail) for every PATH entry, in order.

    Yields skipped entries too rather than silently dropping them: "why is my
    tool not found" is answered by the skip list far more often than by the
    hit list.
    """
    env = os.environ if env is None else env
    raw = env.get("PATH", "")
    seen: set[str] = set()

    for entry in [*raw.split(os.pathsep), *extra]:
        if not entry.strip():
            # On Windows an empty PATH entry means the current directory, a
            # long-standing hijack vector. Never scanned, always reported.
            yield entry, SkipReason.EMPTY_PATH_ENTRY, "resolves to the current directory"
            continue

        expanded = os.path.expandvars(os.path.expanduser(entry))
        key = os.path.normcase(os.path.abspath(expanded))
        if key in seen:
            yield expanded, SkipReason.DUPLICATE, "already scanned earlier in PATH"
            continue
        seen.add(key)

        try:
            if not os.path.exists(expanded):
                yield expanded, SkipReason.MISSING, ""
            elif not os.path.isdir(expanded):
                yield expanded, SkipReason.NOT_A_DIRECTORY, ""
            else:
                yield expanded, None, ""
        except OSError as exc:            # unreadable, or a dead network mount
            yield expanded, SkipReason.PERMISSION_DENIED, str(exc)


def discover(
    env: Mapping[str, str] | None = None,
    profile: ShellProfile | None = None,
    *,
    include_builtins: bool = True,
    include_stubs: bool = True,
    extra_dirs: Iterable[str] = (),
    max_entries: int = 50_000,
) -> Inventory:
    """Scan for every invocable command. Never executes anything.

    `max_entries` is a guard, not a tuning knob: a PATH entry pointing at a
    huge tree (or a network mount) should degrade to a truncated inventory
    rather than hanging the daemon at startup.
    """
    started = time.monotonic()
    env = os.environ if env is None else env
    profile = profile or detect(dict(env))
    pathext = _pathext(env)
    windows = os.name == "nt"

    # name -> first winning entry. The shell takes the first match in PATH
    # order, so discovery must agree with it or every later stage is wrong.
    winners: dict[str, Discovered] = {}
    shadowed: dict[str, list[str]] = {}
    skipped: list[SkippedDir] = []
    truncated = False
    count = 0

    for directory, reason, detail in iter_path_dirs(env, extra_dirs):
        if reason is not None:
            skipped.append(SkippedDir(directory, reason, detail))
            continue
        if truncated:
            skipped.append(SkippedDir(directory, SkipReason.BUDGET_EXHAUSTED, ""))
            continue

        try:
            listing = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as exc:
            skipped.append(SkippedDir(directory, SkipReason.UNREADABLE, str(exc)))
            continue

        for item in listing:
            if count >= max_entries:
                truncated = True
                break
            try:
                if item.is_dir():
                    continue
            except OSError:
                continue

            name = _invocable_name(item.name, pathext)
            if name is None:
                continue

            if not windows:
                # POSIX: the execute bit is the whole answer. On Windows
                # os.access(X_OK) is true for anything readable, so asking
                # would actively mislead.
                try:
                    if not os.access(item.path, os.X_OK):
                        continue
                except OSError:
                    continue

            key = name.lower() if windows else name
            if key in winners:
                shadowed.setdefault(key, []).append(item.path)
                continue

            is_stub = _is_app_exec_alias(item.path)
            try:
                st = item.stat()
                size, mtime_ns = st.st_size, st.st_mtime_ns
            except OSError:
                size = mtime_ns = None

            winners[key] = Discovered(
                name=name,
                kind=EntryKind.STUB if is_stub else EntryKind.EXECUTABLE,
                path=item.path,
                source=f"path:{directory}",
                size=size,
                mtime_ns=mtime_ns,
            )
            count += 1

    entries = [
        Discovered(
            name=e.name, kind=e.kind, path=e.path, source=e.source,
            shadowed_by_order=tuple(shadowed.get(key, ())),
            size=e.size, mtime_ns=e.mtime_ns,
        )
        for key, e in winners.items()
    ]

    if include_builtins:
        family = _builtin_family(profile)
        for builtin in sorted(_BUILTINS.get(family, ())):
            key = builtin.lower() if windows else builtin
            if key in winners:
                continue      # a real binary shadows the builtin for our purposes
            entries.append(
                Discovered(name=builtin, kind=EntryKind.BUILTIN,
                           source=f"builtin:{family}")
            )

    if not include_stubs:
        entries = [e for e in entries if e.kind is not EntryKind.STUB]

    # Sorted so two inventories diff cleanly and cache keys are stable.
    entries.sort(key=lambda e: (e.name.lower(), e.source))

    return Inventory(
        entries=tuple(entries),
        skipped=tuple(skipped),
        shell=profile.id,
        duration_s=time.monotonic() - started,
        truncated=truncated,
    )


def _builtin_family(profile: ShellProfile) -> str:
    if profile.id in ("powershell", "pwsh"):
        return "powershell"
    if profile.id == "cmd":
        return "cmd"
    if profile.id == "fish":
        return "fish"
    return "posix"
