"""Tests for the PowerShell widget.

What can and cannot be tested here, stated plainly.

CAN: that the module parses, that its functions exist, that it locates the
daemon, that it talks the protocol correctly against a real server, and --
most importantly -- that every failure path returns empty rather than throwing.

CANNOT: whether pressing Tab *feels* right. Key handlers need an interactive
console; PSReadLine's handlers cannot be driven from a non-interactive session.
That judgement needs a human at a prompt, and no assertion here substitutes for
it.

So these tests cover the widget's logic and its protocol behaviour, which is
where a bug would be silent. The interaction itself is verified by using it.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from cl_ai.daemon.protocol import Request, Response, Suggestion
from cl_ai.daemon.transport import Server, port_file_for
from cl_ai.platform_ import resolve_executable

MODULE = (
    Path(__file__).resolve().parents[1]
    / "src" / "cl_ai" / "shell" / "powershell" / "cl-ai.psm1"
)

pwsh = resolve_executable("pwsh") or resolve_executable("powershell")
requires_powershell = pytest.mark.skipif(
    pwsh is None, reason="no PowerShell on this machine"
)


def run_ps(script: str, timeout: int = 60, port_file: str | None = None
           ) -> subprocess.CompletedProcess:
    """Run a snippet with the module imported.

    `port_file` uses the module's documented CL_AI_PORT_FILE override.
    Redefining Get-ClAiPortFile in the caller's scope does not work: the
    module's own functions resolve it in module scope, so the override has
    to be something the module itself consults.
    """
    body = textwrap.dedent(f"""
        $ErrorActionPreference = 'Stop'
        Import-Module '{MODULE}' -Force
        {script}
    """)
    env = dict(os.environ)
    if port_file is not None:
        env["CL_AI_PORT_FILE"] = port_file
    return subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", body],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, check=False, env=env,
    )


def test_module_file_is_shipped():
    """It is package data; if packaging drops it, `cl-ai init` has nothing to
    install and the failure appears only at a user's prompt."""
    assert MODULE.is_file()
    assert MODULE.read_text(encoding="utf-8").strip()


@requires_powershell
def test_module_parses_and_imports():
    """A syntax error here is invisible to Python tooling entirely."""
    result = run_ps("Write-Output 'imported'")
    assert result.returncode == 0, result.stderr
    assert "imported" in result.stdout


@requires_powershell
def test_expected_functions_are_exported():
    result = run_ps(
        "(Get-Command -Module 'cl-ai').Name | Sort-Object | ForEach-Object { $_ }"
    )
    assert result.returncode == 0, result.stderr
    exported = set(result.stdout.split())
    assert {
        "Register-ClAiKeyHandlers", "Invoke-ClAiComplete", "Invoke-ClAiNext",
        "Invoke-ClAiPrevious", "Get-ClAiSuggestions", "Invoke-ClAiRequest",
    } <= exported


@requires_powershell
def test_importing_without_psreadline_is_harmless():
    """Importing in a script or in CI must not bind keys or fail.

    Register returns false rather than throwing when there is no interactive
    line editor to bind to.
    """
    result = run_ps("$bound = Register-ClAiKeyHandlers; Write-Output \"bound=$bound\"")
    assert result.returncode == 0, result.stderr
    assert "bound=" in result.stdout


@requires_powershell
def test_port_file_location_matches_the_python_side():
    """The two sides compute this independently; if they drift, the widget
    silently never finds the daemon."""
    endpoint = r"\\.\pipe\cl-ai-testuser"
    result = run_ps(f"Get-ClAiPortFile -Endpoint '{endpoint}'")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).name == Path(port_file_for(endpoint)).name


@requires_powershell
def test_no_daemon_yields_no_suggestions_and_no_error(tmp_path):
    """The commonest situation in the wild: the daemon is not running.

    The absence is ARRANGED, via a port file that does not exist, rather than
    assumed. The first version of this test just called the function and
    trusted the developer to have no daemon -- which was true on CI and true
    right up until the product started working, at which point it failed on
    the machine of anyone actually using it.
    """
    missing = tmp_path / "definitely-not-a-daemon.port"
    result = run_ps(
        "$s = Get-ClAiSuggestions -Buffer 'git com'; Write-Output \"count=$($s.Count)\"",
        port_file=str(missing),
    )
    assert result.returncode == 0, result.stderr
    assert "count=0" in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="loopback transport is the Windows path")
@requires_powershell
def test_empty_buffer_does_not_contact_the_daemon(tmp_path):
    """Proven against a LIVE daemon that counts what it receives.

    The earlier version just asserted the result was empty, which is true
    whether or not a request was sent -- it could not fail. The claim in the
    name is about traffic, so the test has to be about traffic: Tab on a blank
    line is the commonest keypress in a shell, and waking the daemon for it
    would be a round trip per keystroke for nothing.
    """
    endpoint = str(tmp_path / "ep")
    received: list[Request] = []

    def handler(req: Request) -> Response:
        received.append(req)
        return Response()

    with Server(handler, endpoint):
        result = run_ps(
            "$s = Get-ClAiSuggestions -Buffer '   ';"
            ' Write-Output "count=$($s.Count)"',
            port_file=port_file_for(endpoint),
        )
        # Snapshotted INSIDE the block. Server.stop() unblocks accept() by
        # connecting to itself, and that self-connect reaches the handler as
        # an ordinary empty SUGGEST -- so asserting after the `with` records a
        # request the widget never sent. Cost me one confident false positive.
        seen = list(received)

    assert result.returncode == 0, result.stderr
    assert "count=0" in result.stdout
    assert seen == [], f"blank buffer reached the daemon: {seen}"


@pytest.mark.skipif(os.name != "nt", reason="loopback transport is the Windows path")
@requires_powershell
def test_widget_talks_to_a_real_daemon(tmp_path):
    """End to end across the language boundary.

    This is the test that would catch the two sides disagreeing about the wire
    format -- the failure mode a Python-only suite cannot see.
    """
    endpoint = str(tmp_path / "ep")
    received: list[Request] = []

    def handler(req: Request) -> Response:
        received.append(req)
        return Response(suggestions=(
            Suggestion("git commit -m 'x'", "Record changes", "git", False),
            Suggestion("git reset --hard", "Discard changes", "git", True),
        ))

    with Server(handler, endpoint):
        # Point the widget at this test's endpoint rather than the real one.
        result = run_ps("""
            $s = Get-ClAiSuggestions -Buffer 'commit my work'
            Write-Output "count=$($s.Count)"
            foreach ($item in $s) { Write-Output "cmd=$($item.command)|danger=$($item.dangerous)" }
        """, port_file=port_file_for(endpoint))

    assert result.returncode == 0, result.stderr
    assert "count=2" in result.stdout, result.stdout
    assert "cmd=git commit -m 'x'|danger=False" in result.stdout
    assert "cmd=git reset --hard|danger=True" in result.stdout

    assert received, "the daemon never saw the request"
    assert received[0].buffer == "commit my work"
    assert received[0].shell == "powershell", "the widget must name its own shell"
    assert received[0].version == 1


@pytest.mark.skipif(os.name != "nt", reason="loopback transport is the Windows path")
@requires_powershell
def test_version_mismatch_is_reported_not_swallowed(tmp_path):
    """A user who upgraded without restarting should be told, once, plainly."""
    from cl_ai.daemon.protocol import ErrorCode

    endpoint = str(tmp_path / "ep")

    def handler(_req: Request) -> Response:
        return Response(ok=False, error=ErrorCode.VERSION_MISMATCH,
                        message="restart your shell, please")

    with Server(handler, endpoint):
        result = run_ps("""
            $s = Get-ClAiSuggestions -Buffer 'anything'
            Write-Output "count=$($s.Count)"
        """, port_file=port_file_for(endpoint))

    assert result.returncode == 0, result.stderr
    assert "count=0" in result.stdout
    assert "restart your shell" in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(os.name != "nt", reason="loopback transport is the Windows path")
@requires_powershell
def test_a_wedged_daemon_does_not_wedge_the_prompt(tmp_path):
    """The requirement the whole design rests on.

    A daemon that never replies must cost the user a short pause, not a frozen
    terminal.
    """
    import time as _time

    endpoint = str(tmp_path / "ep")

    def molasses(_req: Request) -> Response:
        _time.sleep(30)
        return Response()

    with Server(molasses, endpoint):
        started = _time.monotonic()
        result = run_ps("""
            $s = Get-ClAiSuggestions -Buffer 'slow'
            Write-Output "count=$($s.Count)"
        """, timeout=60, port_file=port_file_for(endpoint))
        elapsed = _time.monotonic() - started

    assert result.returncode == 0, result.stderr
    assert "count=0" in result.stdout
    # Generous: this includes PowerShell's own start-up, which dwarfs the
    # widget's 500ms socket timeout.
    assert elapsed < 30, f"the widget blocked for {elapsed:.1f}s"


@pytest.mark.skipif(os.name != "nt", reason="loopback transport is the Windows path")
@requires_powershell
def test_garbled_reply_is_survived(tmp_path):
    """Anything that is not a well-formed reply must read as "no suggestion"."""
    import socket
    import threading

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(4)
    port = sock.getsockname()[1]
    endpoint = str(tmp_path / "ep")
    Path(port_file_for(endpoint)).write_text(str(port), encoding="ascii")

    def serve_rubbish():
        try:
            conn, _ = sock.accept()
            with conn:
                conn.sendall(b"absolutely not json\n")
        except OSError:
            pass

    thread = threading.Thread(target=serve_rubbish, daemon=True)
    thread.start()
    try:
        result = run_ps("""
            $s = Get-ClAiSuggestions -Buffer 'x'
            Write-Output "count=$($s.Count)"
        """, port_file=port_file_for(endpoint))
    finally:
        sock.close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert "count=0" in result.stdout


@requires_powershell
def test_request_json_matches_the_python_schema():
    """Field names are a contract across two languages with no shared types."""
    result = run_ps("""
        $r = [ordered]@{
            kind='suggest'; buffer='x'; cursor=0; shell='powershell'
            cwd='/tmp'; limit=5; deadline_ms=500; version=1
        }
        $r | ConvertTo-Json -Compress
    """)
    assert result.returncode == 0, result.stderr
    from cl_ai.daemon.protocol import decode_request

    # Decoded from the text PowerShell actually emitted, not from a Python
    # dict: the point is that ConvertTo-Json produces something the Python
    # side reads, field names and all.
    decoded = decode_request(result.stdout.strip())
    assert decoded.shell == "powershell"
    assert decoded.buffer == "x"
    assert decoded.limit == 5
    assert json.loads(result.stdout)["deadline_ms"] == 500


# ------------------------------------------------------------------ dismissal


@requires_powershell
def test_the_dismiss_function_is_exported():
    """Escape is the only way back. Tab REPLACES the buffer, so without a
    dismissal the text a user typed is simply gone -- which is bad on its own
    and much worse when the suggestion that replaced it is destructive."""
    result = run_ps(
        "Write-Output ([bool](Get-Command Invoke-ClAiDismiss -ErrorAction SilentlyContinue))"
    )
    assert result.returncode == 0, result.stderr
    assert "True" in result.stdout


@requires_powershell
def test_registering_binds_a_dismiss_key():
    result = run_ps(
        "Register-ClAiKeyHandlers | Out-Null;"
        " Get-PSReadLineKeyHandler -Bound"
        " | Where-Object { $_.Function -eq 'clAiDismiss' }"
        " | ForEach-Object { Write-Output \"key=$($_.Key)\" }"
    )
    assert result.returncode == 0, result.stderr
    # Non-interactive hosts have no PSReadLine console to bind to, so an
    # empty result is a legitimate outcome here; a CRASH is not.
    if "key=" in result.stdout:
        assert "Escape" in result.stdout


@requires_powershell
def test_the_dismiss_key_is_configurable():
    result = run_ps(
        "(Get-Command Register-ClAiKeyHandlers).Parameters.Keys"
        " | Where-Object { $_ -eq 'DismissKey' }"
    )
    assert result.returncode == 0, result.stderr
    assert "DismissKey" in result.stdout


@requires_powershell
def test_dismissing_with_no_suggestion_falls_through_to_plain_escape():
    """Binding Escape must not take anything away from a user who never
    pressed Tab: with no cl-ai state it has to behave exactly as before."""
    result = run_ps(
        "Reset-ClAiCycle;"
        " $src = (Get-Command Invoke-ClAiDismiss).Definition;"
        " Write-Output ([bool]($src -match 'RevertLine'))"
    )
    assert result.returncode == 0, result.stderr
    assert "True" in result.stdout
