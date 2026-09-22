"""Discovery tests.

Built around a synthetic PATH rather than the host's, so the same assertions
run identically on Linux, macOS and Windows. The handful of tests that need a
real machine are marked and skipped elsewhere.

The most important test in this file is test_discovery_never_executes_anything:
discovery is defined as exec-free, and that guarantee is what lets it run
openly over every binary on a user's PATH without a safety story.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

from cl_ai.catalog import discovery
from cl_ai.catalog.discovery import (
    Discovered,
    EntryKind,
    Inventory,
    SkipReason,
    discover,
    iter_path_dirs,
)
from cl_ai.platform_ import PROFILES

WINDOWS = os.name == "nt"


def make_exe(directory, name: str) -> str:
    """Create a file the host OS will consider executable."""
    path = directory / name
    path.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    if not WINDOWS:
        path.chmod(0o755)
    return str(path)


def env_with(*dirs, **extra) -> dict:
    env = {"PATH": os.pathsep.join(str(d) for d in dirs)}
    if WINDOWS:
        env.setdefault("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    env.update(extra)
    return env


def names(inv: Inventory) -> set[str]:
    return {e.name for e in inv.entries if e.kind is not EntryKind.BUILTIN}


# --------------------------------------------------------------- the guarantee

def test_discovery_never_executes_anything(tmp_path, monkeypatch):
    """The defining property of this module.

    Discovery runs openly across every binary on a user's PATH, including ones
    nobody vetted. It is safe to do that only because it never runs them, so
    that is asserted directly rather than assumed: every process-spawning entry
    point is replaced with a landmine.
    """
    exe_name = "victim.exe" if WINDOWS else "victim"
    make_exe(tmp_path, exe_name)

    def landmine(*args, **kwargs):
        raise AssertionError(f"discovery executed something: {args!r}")

    for target in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, target, landmine)
    monkeypatch.setattr(os, "system", landmine)
    if hasattr(os, "execv"):
        monkeypatch.setattr(os, "execv", landmine)
    monkeypatch.setattr(os, "popen", landmine)

    inv = discover(env_with(tmp_path), PROFILES["bash"])
    assert "victim" in names(inv)


# ------------------------------------------------------------------- the basics

def test_finds_an_executable(tmp_path):
    make_exe(tmp_path, "mytool.exe" if WINDOWS else "mytool")
    inv = discover(env_with(tmp_path), PROFILES["bash"])
    entry = inv.get("mytool")
    assert entry is not None
    assert entry.kind is EntryKind.EXECUTABLE
    assert entry.source.startswith("path:")
    assert entry.size is not None and entry.mtime_ns is not None


def test_identity_is_size_and_mtime_not_a_version(tmp_path):
    """Cache keys must not require running the binary to read --version, and
    must still change when a rebuild keeps the same version number."""
    make_exe(tmp_path, "t.exe" if WINDOWS else "t")
    before = discover(env_with(tmp_path), PROFILES["bash"]).get("t")

    path = tmp_path / ("t.exe" if WINDOWS else "t")
    path.write_text("#!/bin/sh\necho different content entirely\n", encoding="utf-8")
    after = discover(env_with(tmp_path), PROFILES["bash"]).get("t")

    assert (before.size, before.mtime_ns) != (after.size, after.mtime_ns)


def test_output_is_deterministic(tmp_path):
    """Two scans must diff cleanly; the artifact is committed and reviewed."""
    for n in ("b", "a", "c"):
        make_exe(tmp_path, f"{n}.exe" if WINDOWS else n)
    first = discover(env_with(tmp_path), PROFILES["bash"])
    second = discover(env_with(tmp_path), PROFILES["bash"])
    assert [e.name for e in first.entries] == [e.name for e in second.entries]
    assert [e.name for e in first.entries] == sorted(e.name for e in first.entries)


# ------------------------------------------------------------ PATH is a mess

def test_first_entry_in_path_wins_and_the_rest_are_recorded(tmp_path):
    """Must agree with the shell, which takes the first match in PATH order.

    Disagreeing here silently mis-answers every later stage. Recording the
    losers is what makes "which python did it pick" answerable -- the question
    that explained this project's Windows CI failure.
    """
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    name = "dup.exe" if WINDOWS else "dup"
    winner = make_exe(first, name)
    loser = make_exe(second, name)

    inv = discover(env_with(first, second), PROFILES["bash"])
    entry = inv.get("dup")
    assert entry.path == winner
    assert loser in entry.shadowed_by_order


def test_empty_path_entry_is_skipped_and_reported(tmp_path):
    """An empty PATH entry means the current directory on Windows -- a
    long-standing hijack vector. Never scanned, and never silently."""
    env = {"PATH": os.pathsep.join([str(tmp_path), "", str(tmp_path)])}
    if WINDOWS:
        env["PATHEXT"] = ".EXE"
    inv = discover(env, PROFILES["bash"])
    assert any(s.reason is SkipReason.EMPTY_PATH_ENTRY for s in inv.skipped)


def test_missing_and_non_directory_entries_are_survived(tmp_path):
    missing = tmp_path / "does-not-exist"
    a_file = tmp_path / "a-file"
    a_file.write_text("x", encoding="utf-8")
    good = tmp_path / "good"
    good.mkdir()
    make_exe(good, "ok.exe" if WINDOWS else "ok")

    inv = discover(env_with(missing, a_file, good), PROFILES["bash"])
    assert "ok" in names(inv)          # the scan continued
    reasons = {s.reason for s in inv.skipped}
    assert SkipReason.MISSING in reasons
    assert SkipReason.NOT_A_DIRECTORY in reasons


def test_duplicate_path_entries_are_scanned_once(tmp_path):
    make_exe(tmp_path, "once.exe" if WINDOWS else "once")
    inv = discover(env_with(tmp_path, tmp_path, tmp_path), PROFILES["bash"])
    assert sum(1 for e in inv.entries if e.name == "once") == 1
    assert sum(1 for s in inv.skipped if s.reason is SkipReason.DUPLICATE) == 2


def test_unreadable_directory_is_reported_not_fatal(tmp_path, monkeypatch):
    good = tmp_path / "good"
    good.mkdir()
    make_exe(good, "fine.exe" if WINDOWS else "fine")
    bad = tmp_path / "bad"
    bad.mkdir()

    real_scandir = os.scandir

    def selective(path=".", *a, **k):
        if str(path) == str(bad):
            raise PermissionError(13, "denied")
        return real_scandir(path, *a, **k)

    monkeypatch.setattr(discovery.os, "scandir", selective)
    inv = discover(env_with(bad, good), PROFILES["bash"])
    assert "fine" in names(inv)
    assert any(s.reason is SkipReason.UNREADABLE for s in inv.skipped)


def test_truncates_rather_than_hanging(tmp_path):
    """A PATH entry pointing at a huge tree or a network mount must degrade,
    not stall the daemon at startup."""
    for i in range(30):
        make_exe(tmp_path, f"t{i:02d}.exe" if WINDOWS else f"t{i:02d}")
    inv = discover(env_with(tmp_path), PROFILES["bash"], max_entries=10,
                   include_builtins=False)
    assert inv.truncated
    assert len(inv.entries) <= 10


def test_directories_on_path_are_not_reported_as_commands(tmp_path):
    (tmp_path / ("subdir.exe" if WINDOWS else "subdir")).mkdir()
    inv = discover(env_with(tmp_path), PROFILES["bash"], include_builtins=False)
    assert "subdir" not in names(inv)


def test_unicode_filenames_survive(tmp_path):
    make_exe(tmp_path, "café-tool.exe" if WINDOWS else "café-tool")
    inv = discover(env_with(tmp_path), PROFILES["bash"])
    assert "café-tool" in names(inv)


# -------------------------------------------------------------- platform rules

@pytest.mark.skipif(not WINDOWS, reason="PATHEXT is a Windows concept")
def test_windows_uses_pathext_and_strips_the_extension(tmp_path):
    make_exe(tmp_path, "tool.exe")
    make_exe(tmp_path, "script.bat")
    make_exe(tmp_path, "readme.txt")      # not in PATHEXT
    inv = discover(env_with(tmp_path), PROFILES["powershell"],
                   include_builtins=False)
    found = names(inv)
    assert "tool" in found and "script" in found     # extension stripped
    assert "readme" not in found and "readme.txt" not in found


@pytest.mark.skipif(not WINDOWS, reason="PATHEXT is a Windows concept")
def test_windows_lookup_is_case_insensitive(tmp_path):
    make_exe(tmp_path, "MyTool.exe")
    inv = discover(env_with(tmp_path), PROFILES["powershell"])
    assert inv.get("mytool") is not None
    assert inv.get("MYTOOL") is not None


@pytest.mark.skipif(WINDOWS, reason="the execute bit does not exist on Windows")
def test_posix_requires_the_execute_bit(tmp_path):
    """os.access(X_OK) is only consulted on POSIX: on Windows it reports
    success for any readable file, so asking there would mislead."""
    runnable = tmp_path / "runnable"
    runnable.write_text("#!/bin/sh\n", encoding="utf-8")
    runnable.chmod(0o755)
    plain = tmp_path / "plain"
    plain.write_text("just data", encoding="utf-8")
    plain.chmod(0o644)

    inv = discover(env_with(tmp_path), PROFILES["bash"], include_builtins=False)
    found = names(inv)
    assert "runnable" in found
    assert "plain" not in found


# ------------------------------------------------- Windows App Execution Aliases

def test_app_execution_aliases_are_marked_unusable(tmp_path, monkeypatch):
    """Zero-byte reparse points in WindowsApps look like programs and open the
    Microsoft Store instead.

    Not hypothetical: the `bash.exe` alias there is the WSL launcher that, with
    no distribution installed, exits 1 silently -- the stub that made every
    Windows CI job fail. Patched rather than created, because a real reparse
    point cannot be made portably.
    """
    real = make_exe(tmp_path, "real.exe" if WINDOWS else "real")
    stub = make_exe(tmp_path, "stub.exe" if WINDOWS else "stub")
    monkeypatch.setattr(discovery, "_is_app_exec_alias", lambda p: p == stub)

    inv = discover(env_with(tmp_path), PROFILES["bash"])
    assert inv.get("stub").kind is EntryKind.STUB
    assert inv.get("stub").usable is False
    assert inv.get("real").kind is EntryKind.EXECUTABLE
    assert "stub" not in inv.names        # excluded from the hard gate
    assert "real" in inv.names
    assert real  # referenced for clarity


def test_stubs_can_be_excluded_entirely(tmp_path, monkeypatch):
    stub = make_exe(tmp_path, "s.exe" if WINDOWS else "s")
    monkeypatch.setattr(discovery, "_is_app_exec_alias", lambda p: p == stub)
    inv = discover(env_with(tmp_path), PROFILES["bash"], include_stubs=False)
    assert inv.get("s") is None


@pytest.mark.skipif(not WINDOWS, reason="needs a real WindowsApps directory")
def test_real_app_execution_aliases_are_detected_on_this_machine():
    """Against the actual OS, not a patch: proves the reparse-tag check works."""
    windows_apps = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps")
    if not os.path.isdir(windows_apps):
        pytest.skip("no WindowsApps directory here")
    aliases = [
        e.path for e in os.scandir(windows_apps)
        if e.is_file() and discovery._is_app_exec_alias(e.path)
    ]
    if not aliases:
        pytest.skip("no aliases configured on this machine")
    assert all(os.path.getsize(p) == 0 for p in aliases)


# -------------------------------------------------------------------- builtins

def test_builtins_are_reported_without_running_a_shell(tmp_path):
    inv = discover(env_with(tmp_path), PROFILES["bash"])
    builtins = {e.name for e in inv.entries if e.kind is EntryKind.BUILTIN}
    assert {"cd", "export", "alias"} <= builtins
    assert all(e.path is None for e in inv.entries if e.kind is EntryKind.BUILTIN)


def test_builtin_set_follows_the_shell(tmp_path):
    def builtins_for(profile):
        inv = discover(env_with(tmp_path), profile)
        return {e.name for e in inv.entries if e.kind is EntryKind.BUILTIN}

    assert "Get-ChildItem" in builtins_for(PROFILES["powershell"])
    assert "Get-ChildItem" not in builtins_for(PROFILES["bash"])
    assert "funcsave" not in builtins_for(PROFILES["bash"])
    assert "string" in builtins_for(PROFILES["fish"])


def test_a_real_binary_shadows_a_builtin(tmp_path):
    """/usr/bin/echo exists on most systems; the file wins the name."""
    make_exe(tmp_path, "echo.exe" if WINDOWS else "echo")
    inv = discover(env_with(tmp_path), PROFILES["bash"])
    assert inv.get("echo").kind is EntryKind.EXECUTABLE
    assert sum(1 for e in inv.entries if e.name == "echo") == 1


def test_builtins_can_be_excluded(tmp_path):
    inv = discover(env_with(tmp_path), PROFILES["bash"], include_builtins=False)
    assert not any(e.kind is EntryKind.BUILTIN for e in inv.entries)


# --------------------------------------------------------------------- helpers

def test_iter_path_dirs_reports_order_and_reasons(tmp_path):
    good = tmp_path / "g"
    good.mkdir()
    results = list(iter_path_dirs({"PATH": os.pathsep.join([str(good), "", "/nope"])}))
    assert results[0][1] is None
    assert results[1][1] is SkipReason.EMPTY_PATH_ENTRY
    assert results[2][1] is SkipReason.MISSING


def test_pathext_without_a_leading_dot_is_normalised():
    """Regression: PATHEXT is user-writable and entries can lack the dot.

    Unnormalised, `endswith("exe")` is a substring test rather than a suffix
    test, so a file named `someexe` was being reported as the command `some`.
    """
    assert discovery._pathext({"PATHEXT": "EXE;.BAT"}) == (".exe", ".bat")
    assert discovery._pathext({"PATHEXT": " .Com ; exe "}) == (".com", ".exe")
    assert discovery._invocable_name("someexe", (".exe",)) is None
    assert discovery._invocable_name("some.exe", (".exe",)) == "some"


def test_lookups_are_cached_and_stay_correct(tmp_path):
    """These sit on the Tab path, where the whole budget is ~100ms.

    Caching is safe only because an Inventory is immutable; the test checks
    both that it is cached and that it still answers correctly.
    """
    make_exe(tmp_path, "cached.exe" if WINDOWS else "cached")
    inv = discover(env_with(tmp_path), PROFILES["bash"])

    assert inv.names is inv.names            # same object: computed once
    assert inv.usable is inv.usable
    assert inv.get("cached") is not None
    assert inv.get("definitely-absent") is None


def test_get_returns_the_path_winner_not_a_shadowed_duplicate(tmp_path):
    """The lookup index must agree with PATH order, like the scan does."""
    first, second = tmp_path / "1", tmp_path / "2"
    first.mkdir()
    second.mkdir()
    name = "dup.exe" if WINDOWS else "dup"
    winner = make_exe(first, name)
    make_exe(second, name)
    inv = discover(env_with(first, second), PROFILES["bash"])
    assert inv.get("dup").path == winner


def test_names_is_the_hard_gate(tmp_path):
    make_exe(tmp_path, "present.exe" if WINDOWS else "present")
    inv = discover(env_with(tmp_path), PROFILES["bash"])
    assert "present" in inv.names
    assert "definitely-not-installed" not in inv.names


def test_entries_are_immutable():
    """Same reasoning as the IR: a frozen record whose contents can be edited
    is not frozen."""
    entry = Discovered(name="x", kind=EntryKind.EXECUTABLE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.name = "y"  # type: ignore[misc]


@pytest.mark.skipif("CI" in os.environ, reason="host PATH varies on runners")
def test_scanning_the_real_path_is_fast_enough_for_startup():
    """The daemon builds this at startup and rebuilds it in the background."""
    inv = discover()
    assert inv.entries, "found nothing on the real PATH, which cannot be right"
    assert inv.duration_s < 10.0, f"real PATH scan took {inv.duration_s:.1f}s"
    assert sys.executable  # sanity: we are running from somewhere
