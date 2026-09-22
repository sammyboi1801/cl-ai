"""Run a quoted value through a real shell and recover it.

This is the whole point of the quoting test suite. Asserting that quote()
returns the string I expect only tests my beliefs about shells; running the
shell and comparing bytes tests the shell. When PowerShell's escaping rules
turn out not to be what I assumed, these fail instead of agreeing with me.

Mechanism: write a tiny script that writes the argument to a file, execute it,
read the file back. Going through a script file rather than `-c`/`-Command`
avoids a second layer of command-line parsing (notably Python's own argument
quoting on Windows) that would make failures ambiguous.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from cl_ai.platform_ import PROFILES, is_available
from cl_ai.render import quoting

TIMEOUT = 30


class OracleUnavailable(RuntimeError):
    """The shell is not installed here."""


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, timeout=TIMEOUT,
        stdin=subprocess.DEVNULL, check=False,
    )


_WORK_ROOT = Path(__file__).resolve().parent / ".oracle_work"


@contextlib.contextmanager
def _workdir():
    """A scratch directory every shell on this machine can actually reach."""
    _WORK_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=_WORK_ROOT))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


_POSIX_ROOT_CACHE: dict[str, str] = {}


def _posix_path(p: Path, shell: str = "bash") -> str:
    """Translate a Windows path into the POSIX view *this* shell actually has.

    Do not guess this. On one Windows machine `bash` may be Git Bash (drive D
    at /d), WSL (/mnt/d), or Cygwin (/cygdrive/d) -- and `shutil.which` can
    point at one while a shim runs another. Observed here: which() resolves to
    C:\\Program Files\\Git\\usr\\bin\\bash.EXE, but `uname -a` reports WSL2, so
    /d does not exist and /mnt/d does.

    So we probe once per shell: create a marker, ask the shell which candidate
    path can see it, and cache the answer.
    """
    s = str(p)
    if len(s) < 2 or s[1] != ":":
        return s.replace("\\", "/")

    drive, rest = s[0].lower(), s[2:].replace("\\", "/")
    prefix = _POSIX_ROOT_CACHE.get(shell)

    if prefix is None:
        probe_dir = Path(__file__).resolve().parent
        marker = probe_dir / ".oracle_probe"
        marker.write_text("x", encoding="ascii")
        try:
            pd = str(probe_dir)
            pdrive, prest = pd[0].lower(), pd[2:].replace("\\", "/")
            for cand in (f"/mnt/{pdrive}", f"/{pdrive}", f"/cygdrive/{pdrive}"):
                test = f"{cand}{prest}/.oracle_probe"
                r = subprocess.run([shell, "-c", f'test -f "{test}"'],
                                   capture_output=True, timeout=TIMEOUT,
                                   stdin=subprocess.DEVNULL, check=False)
                if r.returncode == 0:
                    prefix = cand[: -len(pdrive)]   # "/mnt/", "/", "/cygdrive/"
                    break
            else:
                prefix = "/"
        finally:
            marker.unlink(missing_ok=True)
        _POSIX_ROOT_CACHE[shell] = prefix

    return f"{prefix}{drive}{rest}"


def roundtrip(value: str, shell_id: str) -> str:
    """Quote `value` for `shell_id`, run it, return what the shell produced.

    A mismatch between the return value and `value` is a real quoting bug.
    """
    if not is_available(shell_id):
        raise OracleUnavailable(shell_id)

    profile = PROFILES[shell_id]
    quoted = quoting.quote(value, profile)

    # Deliberately NOT the system temp dir. Under WSL (and some sandboxes) the
    # Windows %TEMP% is not visible to the Linux side, so a file Python can see
    # does not exist as far as bash is concerned. A directory inside the repo is
    # reachable from every shell we drive.
    with _workdir() as d:
        out = d / "out.bin"
        out_literal = quoting.quote(str(out), profile)

        if shell_id in ("bash", "zsh", "fish"):
            script = d / "s.sh"
            out_literal = quoting.quote(_posix_path(out, profile.exec_flags[0]), profile)
            script.write_text(f"printf %s {quoted} > {out_literal}\n",
                              encoding="utf-8", newline="\n")
            proc = _run([profile.exec_flags[0], _posix_path(script, profile.exec_flags[0])], d)

        elif shell_id in ("powershell", "pwsh"):
            script = d / "s.ps1"
            # WriteAllText with an explicit no-BOM UTF-8 encoder keeps the
            # comparison byte-exact; Out-File would inject an encoding of its
            # own choosing and a trailing newline.
            # Must exercise COMMAND-ARGUMENT position, which is what a renderer
            # actually produces. A .NET method call is *expression* position,
            # where a bare word is a syntax error -- so testing there would
            # report a failure for a perfectly valid unquoted argument.
            body = (
                "$ErrorActionPreference='Stop'\n"
                f"$v = Write-Output -NoEnumerate -InputObject {quoted}\n"
                f"[System.IO.File]::WriteAllText({out_literal}, [string]$v, "
                "(New-Object System.Text.UTF8Encoding $false))\n"
            )
            # Windows PowerShell 5.1 reads a .ps1 as ANSI unless it has a BOM.
            script.write_text(body, encoding="utf-8-sig", newline="\n")
            proc = _run([profile.exec_flags[0], "-NoProfile", "-NonInteractive",
                         "-ExecutionPolicy", "Bypass", "-File", str(script)], d)

        elif shell_id == "cmd":
            script = d / "s.bat"
            # `echo` appends CRLF and cannot emit an empty line without a hack;
            # the caller strips the trailing CRLF.
            script.write_text(f"@echo off\r\necho {quoted}> {out_literal}\r\n",
                              encoding="utf-8", newline="")
            proc = _run(["cmd", "/c", str(script)], d)

        else:
            raise OracleUnavailable(shell_id)

        if proc.returncode != 0:
            raise AssertionError(
                f"{shell_id} exited {proc.returncode} for {value!r}\n"
                f"  quoted: {quoted}\n"
                f"  stderr: {proc.stderr.decode('utf-8', 'replace')[:400]}"
            )
        if not out.exists():
            raise AssertionError(f"{shell_id} produced no output for {value!r}")

        raw = out.read_bytes()
        text = raw.decode("utf-8", "replace")
        if shell_id == "cmd" and text.endswith("\r\n"):
            text = text[:-2]
        return text
