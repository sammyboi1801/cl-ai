"""Shell profiles: the single place platform differences are allowed to live.

Capability-based, never OS-sniffing. The unit of targeting is the (os, shell)
pair, and the two are orthogonal: PowerShell runs on Linux, bash runs on
Windows via Git Bash and WSL, and one machine can legitimately present three
different targets at once.

No other module in the codebase branches on the operating system. A renderer
receives a ShellProfile and asks it questions; it never asks where it is.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class QuoteStyle(str, Enum):
    """How a shell delimits a literal argument.

    These are genuinely different algorithms, not dialects of one another --
    see render.quoting for the implementations.
    """
    POSIX = "posix"              # '...' with '\'' for embedded quotes
    FISH = "fish"                # '...' but \ and ' are STILL escapes inside
    POWERSHELL = "powershell"    # '...' with '' for embedded quotes
    CMD = "cmd"                  # "..." with "" -- and several impossible cases


class PathStyle(str, Enum):
    POSIX = "posix"      # /, case-sensitive, ~ for home
    WINDOWS = "windows"  # \, case-insensitive, drive letters, UNC


class PipelineKind(str, Enum):
    """What a pipe actually carries.

    This is the deep difference, not a syntactic one. POSIX pipes bytes;
    PowerShell pipes .NET objects. A text-oriented plan rendered naively into
    PowerShell runs and produces wrong results -- worse than an error, because
    nothing signals the failure.
    """
    TEXT = "text"
    OBJECT = "object"
    NONE = "none"        # cmd.exe: pipes exist but are not usefully typed


class EnvSyntax(str, Enum):
    DOLLAR = "dollar"        # $VAR
    PS_ENV = "ps_env"        # $env:VAR
    PERCENT = "percent"      # %VAR%


@dataclass(frozen=True)
class ShellProfile:
    """Everything platform-specific, resolved into one object.

    Hashable, so it can be used as a cache key -- the daemon keys warm state by
    profile. That is why `builtin_map` is excluded from comparison: a dict field
    would make the whole dataclass unhashable despite frozen=True.
    """

    id: str
    quote_style: QuoteStyle
    path_style: PathStyle
    pipeline: PipelineKind
    env_syntax: EnvSyntax
    line_ending: str
    comment_prefix: str
    #: Command used to run a single command string, e.g. ("bash", "-c").
    exec_flags: tuple[str, ...] = ()
    #: Shell-native names for common operations, for tools that have no binary.
    #: Read-only: profiles share these tables, so a plain dict would let a write
    #: through one profile silently alter every other profile that shares it.
    builtin_map: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), compare=False
    )

    @property
    def is_posix_like(self) -> bool:
        """Text-stream shells with POSIX-ish syntax. Includes fish.

        Note this is not the same question as "does it quote like POSIX" --
        fish is POSIX-like in structure while quoting differently. Use
        `quote_style` for anything about quoting.
        """
        return self.quote_style in (QuoteStyle.POSIX, QuoteStyle.FISH)


_POSIX_BUILTINS: Mapping[str, str] = MappingProxyType({})
_PS_BUILTINS = MappingProxyType({
    "ls": "Get-ChildItem",
    "grep": "Select-String",
    "cat": "Get-Content",
    "cp": "Copy-Item",
    "mv": "Move-Item",
    "rm": "Remove-Item",
    "ps": "Get-Process",
    "kill": "Stop-Process",
    "pwd": "Get-Location",
    "which": "Get-Command",
})
_CMD_BUILTINS = MappingProxyType({
    "ls": "dir",
    "cat": "type",
    "cp": "copy",
    "mv": "move",
    "rm": "del",
    "ps": "tasklist",
    "kill": "taskkill",
    "pwd": "cd",
    "which": "where",
})


PROFILES: dict[str, ShellProfile] = {
    "bash": ShellProfile(
        id="bash", quote_style=QuoteStyle.POSIX, path_style=PathStyle.POSIX,
        pipeline=PipelineKind.TEXT, env_syntax=EnvSyntax.DOLLAR,
        line_ending="\n", comment_prefix="#", exec_flags=("bash", "-c"),
        builtin_map=_POSIX_BUILTINS,
    ),
    "zsh": ShellProfile(
        id="zsh", quote_style=QuoteStyle.POSIX, path_style=PathStyle.POSIX,
        pipeline=PipelineKind.TEXT, env_syntax=EnvSyntax.DOLLAR,
        line_ending="\n", comment_prefix="#", exec_flags=("zsh", "-c"),
        builtin_map=_POSIX_BUILTINS,
    ),
    "fish": ShellProfile(
        id="fish", quote_style=QuoteStyle.FISH, path_style=PathStyle.POSIX,
        pipeline=PipelineKind.TEXT, env_syntax=EnvSyntax.DOLLAR,
        line_ending="\n", comment_prefix="#", exec_flags=("fish", "-c"),
        builtin_map=_POSIX_BUILTINS,
    ),
    "powershell": ShellProfile(
        id="powershell", quote_style=QuoteStyle.POWERSHELL,
        path_style=PathStyle.WINDOWS, pipeline=PipelineKind.OBJECT,
        env_syntax=EnvSyntax.PS_ENV, line_ending="\r\n", comment_prefix="#",
        exec_flags=("powershell", "-NoProfile", "-NonInteractive", "-Command"),
        builtin_map=_PS_BUILTINS,
    ),
    "pwsh": ShellProfile(
        id="pwsh", quote_style=QuoteStyle.POWERSHELL,
        path_style=PathStyle.POSIX, pipeline=PipelineKind.OBJECT,
        env_syntax=EnvSyntax.PS_ENV, line_ending="\n", comment_prefix="#",
        exec_flags=("pwsh", "-NoProfile", "-NonInteractive", "-Command"),
        builtin_map=_PS_BUILTINS,
    ),
    "cmd": ShellProfile(
        id="cmd", quote_style=QuoteStyle.CMD, path_style=PathStyle.WINDOWS,
        pipeline=PipelineKind.NONE, env_syntax=EnvSyntax.PERCENT,
        line_ending="\r\n", comment_prefix="REM", exec_flags=("cmd", "/c"),
        builtin_map=_CMD_BUILTINS,
    ),
}

#: Profiles whose quoting can be verified by actually running the shell.
VERIFIABLE = ("bash", "zsh", "fish", "powershell", "pwsh", "cmd")


def get_profile(shell_id: str) -> ShellProfile:
    """Look up a profile by id. Raises KeyError with the valid set listed."""
    try:
        return PROFILES[shell_id]
    except KeyError:
        raise KeyError(
            f"unknown shell {shell_id!r}; known: {', '.join(sorted(PROFILES))}"
        ) from None


#: Places a working shell hides when the name on PATH resolves to something
#: else. Windows ships C:\Windows\System32\bash.exe -- a launcher for WSL that,
#: with no distribution installed, exits 1 and writes nothing to stderr. It
#: precedes Git's bash on PATH, so `which bash` finds a binary that cannot run
#: anything. Observed on GitHub's windows-latest runners.
_FALLBACK_PATHS: dict[str, tuple[str, ...]] = {
    "bash": (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ),
}

_EXEC_CACHE: dict[str, str | None] = {}


def _works(executable: str, shell_id: str) -> bool:
    """Run a trivial command and require a distinctive exit code.

    Existence on disk is not usability. We ask for exit code 7 specifically so
    that a launcher failing for its own reasons -- which typically exits 1 --
    cannot be mistaken for success.
    """
    if shell_id in ("powershell", "pwsh"):
        argv = [executable, "-NoProfile", "-NonInteractive", "-Command", "exit 7"]
    elif shell_id == "cmd":
        argv = [executable, "/c", "exit 7"]
    else:
        argv = [executable, "-c", "exit 7"]
    try:
        completed = subprocess.run(
            argv, capture_output=True, timeout=30,
            stdin=subprocess.DEVNULL, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 7


def resolve_executable(shell_id: str) -> str | None:
    """The path to a shell that actually runs, or None.

    Deliberately functional rather than nominal: a name on PATH is a claim, and
    this checks it. Cached, because it costs a process per shell.
    """
    if shell_id in _EXEC_CACHE:
        return _EXEC_CACHE[shell_id]

    profile = PROFILES.get(shell_id)
    resolved: str | None = None
    if profile and profile.exec_flags:
        candidates = []
        found = shutil.which(profile.exec_flags[0])
        if found:
            candidates.append(found)
        candidates.extend(_FALLBACK_PATHS.get(shell_id, ()))
        for candidate in candidates:
            if os.path.exists(candidate) and _works(candidate, shell_id):
                resolved = candidate
                break

    _EXEC_CACHE[shell_id] = resolved
    return resolved


def is_available(shell_id: str) -> bool:
    """Whether this shell can actually be executed here.

    Used by tests to decide between verifying against the real shell and
    skipping with a named marker. Never used to *guess* a profile.
    """
    return resolve_executable(shell_id) is not None


def detect(env: Mapping[str, str] | None = None) -> ShellProfile:
    """Best-effort detection of the current shell.

    Deliberately conservative: we would rather return a POSIX profile we are
    confident about than guess PowerShell from the fact that we are on Windows.
    The caller can always override -- and the shell adapter, which knows
    exactly what it is, always does.
    """
    # Bound to a separate name rather than rebinding the parameter, whose
    # declared type is narrower: os.environ is an _Environ, not a dict.
    source: Mapping[str, str] = os.environ if env is None else env

    # An explicit override always wins.
    forced = source.get("CL_AI_SHELL")
    if forced:
        # Silently ignoring a typo here would hand the user a wrong shell and a
        # wrong quoter, which is exactly the class of failure this module is
        # meant to prevent. Fail loudly instead.
        return get_profile(forced)

    # PowerShell exports these; nothing else does.
    if source.get("PSModulePath"):
        # PowerShell 7+ sets PSEdition=Core, Windows PowerShell 5.1 does not.
        if source.get("PSEdition") == "Core" or source.get("POWERSHELL_DISTRIBUTION_CHANNEL"):
            return PROFILES["pwsh"]
        return PROFILES["powershell"]

    # POSIX shells export SHELL with a path to the binary.
    shell_path = source.get("SHELL", "")
    if shell_path:
        name = os.path.basename(shell_path).lower().removesuffix(".exe")
        if name in PROFILES:
            return PROFILES[name]

    # cmd.exe sets COMSPEC and, unlike PowerShell, no PSModulePath.
    if source.get("COMSPEC") and os.name == "nt":
        return PROFILES["cmd"]

    return PROFILES["bash"]
