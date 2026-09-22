"""Tests for functional shell resolution.

This code was added to fix a CI failure and shipped without tests of its own,
which is precisely the gap that let the earlier bugs through. It also runs
executables found on PATH, so it deserves more scrutiny than most modules and
not less.

The bug it exists for: Windows ships C:\\Windows\\System32\\bash.exe, a launcher
for WSL that -- with no distribution installed -- exits 1 and writes nothing.
It precedes Git's bash on PATH, so `shutil.which("bash")` returns a binary that
cannot execute anything.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from cl_ai import platform_
from cl_ai.platform_ import PROFILES, is_available, resolve_executable


@pytest.fixture(autouse=True)
def _clear_cache():
    """Resolution is cached process-wide; tests must not inherit each other."""
    platform_._EXEC_CACHE.clear()
    yield
    platform_._EXEC_CACHE.clear()


def test_resolution_is_functional_not_nominal(monkeypatch, tmp_path):
    """A name on PATH is a claim. Resolution must check it.

    Simulates the System32 WSL stub: a binary that exists and is executable but
    exits non-zero and produces no output.
    """
    stub = tmp_path / "fake-bash"
    stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    monkeypatch.setattr(platform_.shutil, "which", lambda _name: str(stub))
    monkeypatch.setattr(platform_, "_FALLBACK_PATHS", {})

    assert resolve_executable("bash") is None
    assert is_available("bash") is False


def test_falls_back_when_the_path_entry_does_not_work(monkeypatch, tmp_path):
    broken = tmp_path / "broken"
    broken.write_text("x", encoding="utf-8")
    working = tmp_path / "working"
    working.write_text("x", encoding="utf-8")

    monkeypatch.setattr(platform_.shutil, "which", lambda _name: str(broken))
    monkeypatch.setattr(platform_, "_FALLBACK_PATHS", {"bash": (str(working),)})
    monkeypatch.setattr(platform_, "_works", lambda exe, _sid: exe == str(working))

    assert resolve_executable("bash") == str(working)


def test_exit_code_seven_is_required_not_merely_success(monkeypatch, tmp_path):
    """Why 7 and not 0.

    A launcher that fails for its own reasons typically exits 1, and one that
    succeeds trivially exits 0. Demanding a distinctive code proves our command
    actually ran, rather than that some process merely started and stopped.
    """
    exe = tmp_path / "always-zero"
    exe.write_text("x", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(platform_.subprocess, "run", fake_run)
    assert platform_._works(str(exe), "bash") is False
    assert "exit 7" in " ".join(calls[0])


def test_probe_failures_do_not_raise(monkeypatch, tmp_path):
    """A hanging or missing shell must degrade to "unavailable", never crash.

    In the daemon this runs behind an interactive keypress; an exception here
    would take out a Tab completion.
    """
    exe = tmp_path / "hangs"
    exe.write_text("x", encoding="utf-8")

    for boom in (OSError("denied"), subprocess.TimeoutExpired("bash", 30)):
        monkeypatch.setattr(
            platform_.subprocess, "run",
            lambda *_a, _b=boom, **_k: (_ for _ in ()).throw(_b),
        )
        assert platform_._works(str(exe), "bash") is False


def test_resolution_is_cached(monkeypatch, tmp_path):
    """One process per shell, not one per call: this sits on the Tab path."""
    exe = tmp_path / "sh"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(platform_.shutil, "which", lambda _name: str(exe))

    count = {"n": 0}

    def counting(_exe, _sid):
        count["n"] += 1
        return True

    monkeypatch.setattr(platform_, "_works", counting)
    for _ in range(5):
        resolve_executable("bash")
    assert count["n"] == 1


def test_unknown_shell_resolves_to_none():
    assert resolve_executable("no-such-shell") is None
    assert is_available("no-such-shell") is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific PATH trap")
def test_windows_does_not_select_the_wsl_stub():
    """Regression for the CI failure itself.

    If bash resolves at all on Windows it must not be the System32 launcher --
    that binary is what made every bash test fail with exit 1 and no stderr.
    """
    resolved = resolve_executable("bash")
    if resolved is None:
        pytest.skip("no working bash on this machine")
    assert "system32" not in resolved.lower(), (
        f"resolved bash to the WSL launcher: {resolved}"
    )


def test_every_profile_resolves_to_something_runnable_or_none():
    """Whatever comes back must actually run; nothing in between."""
    for shell_id in PROFILES:
        resolved = resolve_executable(shell_id)
        if resolved is not None:
            assert platform_._works(resolved, shell_id), (
                f"{shell_id} resolved to {resolved}, which does not run"
            )
