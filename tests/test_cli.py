"""The `cl-ai` command line.

The first test here is the one that matters most: pyproject declares
`cl-ai = "cl_ai.cli:main"` and cli.py shipped with no `main`, so installing
the package put a binary on PATH that raised ImportError on every invocation.
Nothing caught it, because nothing had ever imported the thing pyproject
names. That is now asserted directly, for both entry points.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cl_ai import cli
from cl_ai.cli import BROKEN, DEGRADED, OK, Check, build_parser, main

# -- the declared entry points actually exist -----------------------------


def test_every_declared_console_script_resolves() -> None:
    """Parsed out of pyproject rather than hardcoded, so adding a script
    without a target fails here instead of on a user's machine."""
    import importlib

    import tomllib

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]
    assert scripts, "no console scripts declared"

    for name, target in scripts.items():
        module_name, _, attribute = target.partition(":")
        module = importlib.import_module(module_name)
        entry = getattr(module, attribute, None)
        assert callable(entry), f"{name} -> {target} is not callable"


def test_the_bare_command_prints_help_and_succeeds(capsys) -> None:
    """Running `cl-ai` with no arguments is what a new user does first."""
    assert main([]) == OK
    assert "doctor" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["doctor", "--json"],
        ["index", "--all"],
        ["suggest", "list", "files"],
        ["daemon", "status"],
        ["init", "--apply"],
    ],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    """Parsing only -- a typo in a parser definition is a crash at startup."""
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_an_unknown_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["frobnicate"])
    assert caught.value.code != 0


def test_a_keyboard_interrupt_becomes_the_conventional_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during a slow `index` should exit 130, not print a traceback.

    Patched at the parser rather than on the module: set_defaults captures the
    function object when the parser is built, so rebinding cli.cmd_index
    afterwards has no effect -- the first version of this test did exactly
    that, ran the real command, and could not fail.
    """

    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    real_parser = build_parser

    def patched() -> object:
        parser = real_parser()
        for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
            action.choices["index"].set_defaults(func=interrupt)  # type: ignore[attr-defined]
        return parser

    monkeypatch.setattr(cli, "build_parser", patched)
    assert main(["index"]) == 130


# -- doctor's contract ----------------------------------------------------


def test_a_probe_that_raises_becomes_a_reported_failure() -> None:
    """A diagnostic that crashes on a broken system only works when it is not
    needed. Every probe is caught."""

    def explode() -> Check:
        raise RuntimeError("the disk is on fire")

    check = cli._probe("thing", explode)
    assert check.status == "fail"
    assert "the disk is on fire" in check.detail


def test_a_probe_failure_does_not_hide_the_checks_after_it(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def explode() -> Check:
        raise RuntimeError("nope")

    monkeypatch.setattr(cli, "_check_inventory", explode)
    monkeypatch.setattr(cli, "_check_end_to_end", lambda shell: [])
    code = main(["--shell", "bash", "doctor"])
    out = capsys.readouterr().out
    assert code == BROKEN
    assert "PATH scan" in out
    assert "catalog" in out, "checks after the failure must still run"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["ok", "ok", "info"], OK),
        (["ok", "warn"], DEGRADED),
        (["ok", "warn", "fail"], BROKEN),
        (["fail"], BROKEN),
        (["info"], OK),
    ],
)
def test_the_exit_code_reports_the_worst_status(
    monkeypatch: pytest.MonkeyPatch, statuses: list[str], expected: int, capsys
) -> None:
    """Machine readable on purpose: an install can be checked from a script."""
    monkeypatch.setattr(
        cli,
        "_probe",
        lambda label, fn: Check(label, statuses.pop(0) if statuses else "info"),
    )
    monkeypatch.setattr(cli, "_check_end_to_end", lambda shell: [])
    assert main(["--shell", "bash", "doctor"]) == expected


def test_doctor_emits_valid_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import json

    monkeypatch.setattr(cli, "_check_end_to_end", lambda shell: [])
    main(["--shell", "bash", "doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload and all({"label", "status", "detail"} <= set(r) for r in payload)


def test_doctor_never_raises_even_with_everything_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The system this is meant to diagnose is, by assumption, broken."""

    def explode(*_a: object, **_k: object) -> object:
        raise RuntimeError("everything is on fire")

    for name in (
        "_check_environment",
        "_check_shell",
        "_check_inventory",
        "_check_sources",
        "_check_catalog",
        "_check_daemon",
        "_check_embedder",
        "_check_end_to_end",
    ):
        monkeypatch.setattr(cli, name, explode)
    assert main(["--shell", "bash", "doctor"]) == BROKEN


def test_a_check_renders_without_colour_when_not_a_tty() -> None:
    plain = Check("label", "ok", "detail").render(colour=False)
    assert "\033[" not in plain
    assert "label" in plain and "detail" in plain


def test_colour_is_suppressed_by_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://no-color.org/ -- respected because doctor output is pasted
    into issues, where escape codes are noise."""

    class Tty:
        @staticmethod
        def isatty() -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._use_colour(Tty()) is False
    monkeypatch.delenv("NO_COLOR")
    assert cli._use_colour(Tty()) is True


def test_a_stream_with_no_isatty_is_not_coloured() -> None:
    assert cli._use_colour(object()) is False


# -- widgets and profiles -------------------------------------------------


def test_the_powershell_widget_ships_in_the_package() -> None:
    """Packaging regression: the module is data, not Python, so it is only in
    the wheel if the build is configured to include it."""
    found = cli._widget_path("powershell")
    assert found is not None and found.is_file()
    assert found.read_text(encoding="utf-8", errors="replace").strip()


@pytest.mark.parametrize("shell", ["zsh", "bash", "fish", "cmd", "nonsense"])
def test_shells_without_a_widget_report_none(shell: str) -> None:
    """Honest about what is not implemented. Returning a path that does not
    exist would make `init` write a line that fails at every shell start."""
    assert cli._widget_path(shell) is None


def test_pwsh_and_powershell_share_one_widget() -> None:
    assert cli._widget_path("pwsh") == cli._widget_path("powershell")


@pytest.mark.parametrize(
    ("shell", "expected"),
    [
        ("powershell", "Import-Module"),
        ("pwsh", "Register-ClAiKeyHandlers"),
        ("bash", ". "),
        ("zsh", ". "),
        ("fish", "source "),
    ],
)
def test_the_snippet_uses_each_shell_s_own_syntax(shell: str, expected: str) -> None:
    snippet = cli._snippet(shell, Path("/tmp/widget"))
    assert expected in snippet
    assert snippet.startswith(cli._MARKER)
    assert snippet.rstrip().endswith(cli._END_MARKER)


# -- init -----------------------------------------------------------------


@pytest.fixture
def fake_widget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    widget = tmp_path / "cl-ai.psm1"
    widget.write_text("# widget\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_widget_path", lambda shell: widget)
    return widget


def test_init_prints_by_default_and_writes_nothing(
    fake_widget: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The default must not edit a file the user did not name, in a shell they
    are about to depend on."""
    profile = tmp_path / "profile.ps1"
    monkeypatch.setattr(cli, "_profile_path", lambda shell: profile)
    assert main(["--shell", "powershell", "init"]) == OK
    assert not profile.exists()
    assert "--apply" in capsys.readouterr().out


def test_init_apply_writes_the_snippet(
    fake_widget: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = tmp_path / "nested" / "profile.ps1"
    monkeypatch.setattr(cli, "_profile_path", lambda shell: profile)
    assert main(["--shell", "powershell", "init", "--apply"]) == OK
    body = profile.read_text(encoding="utf-8")
    assert cli._MARKER in body and str(fake_widget) in body


def test_init_apply_is_idempotent(
    fake_widget: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running an installer is something people do. Twice-registered key
    handlers are not harmless: the second import shadows the first."""
    profile = tmp_path / "profile.ps1"
    monkeypatch.setattr(cli, "_profile_path", lambda shell: profile)
    main(["--shell", "powershell", "init", "--apply"])
    first = profile.read_text(encoding="utf-8")
    assert main(["--shell", "powershell", "init", "--apply"]) == OK
    assert profile.read_text(encoding="utf-8") == first


def test_init_apply_preserves_what_was_already_there(
    fake_widget: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appending to someone's shell profile must never lose a line of it."""
    profile = tmp_path / "profile.ps1"
    profile.write_text("Set-Alias ll Get-ChildItem", encoding="utf-8")
    monkeypatch.setattr(cli, "_profile_path", lambda shell: profile)
    main(["--shell", "powershell", "init", "--apply"])
    body = profile.read_text(encoding="utf-8")
    assert body.startswith("Set-Alias ll Get-ChildItem")
    assert "\nSet-Alias ll Get-ChildItem\n" not in body[len("Set-Alias") :], body


def test_init_refuses_a_shell_with_no_widget(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_widget_path", lambda shell: None)
    assert main(["--shell", "zsh", "init"]) == BROKEN
    assert "no widget" in capsys.readouterr().err


def test_init_reports_an_unwritable_profile_rather_than_raising(
    fake_widget: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file")
    monkeypatch.setattr(cli, "_profile_path", lambda shell: blocker / "sub" / "p.ps1")
    assert main(["--shell", "powershell", "init", "--apply"]) == BROKEN
    assert "cannot" in capsys.readouterr().err


def test_init_reports_an_unknown_profile_location(
    fake_widget: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli, "_profile_path", lambda shell: None)
    assert main(["--shell", "powershell", "init"]) == BROKEN
    assert "profile" in capsys.readouterr().err


# -- suggest --------------------------------------------------------------


def test_suggest_refuses_an_empty_query(capsys) -> None:
    assert main(["suggest"]) == BROKEN
    assert "empty" in capsys.readouterr().err


def test_suggest_reports_no_match_without_inventing_one(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A nearest-neighbour substitution is how a missing tool becomes a
    confidently wrong command."""

    class Empty:
        @staticmethod
        def detailed(query: str, *, limit: int = 5) -> list[object]:
            return []

    monkeypatch.setattr(cli._Engine, "build", classmethod(lambda cls, s, **k: Empty()))
    code = main(["--shell", "bash", "suggest", "--no-daemon", "zzqqxxyy"])
    assert code == DEGRADED
    assert "no match" in capsys.readouterr().out


def test_suggest_marks_destructive_commands(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    class Dangerous:
        @staticmethod
        def detailed(query: str, *, limit: int = 5) -> list[tuple[str, str, bool]]:
            return [("rm -rf path/to/directory", "Remove files", True)]

    monkeypatch.setattr(
        cli._Engine, "build", classmethod(lambda cls, s, **k: Dangerous())
    )
    assert main(["--shell", "bash", "suggest", "--no-daemon", "delete"]) == OK
    assert "destructive" in capsys.readouterr().out


def test_suggest_without_a_catalog_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(cli._Engine, "build", classmethod(lambda cls, s, **k: None))
    assert main(["--shell", "bash", "suggest", "--no-daemon", "anything"]) == BROKEN
    assert "cl-ai index" in capsys.readouterr().err


def test_suggest_emits_valid_json(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import json

    class One:
        @staticmethod
        def detailed(query: str, *, limit: int = 5) -> list[tuple[str, str, bool]]:
            return [("ls -1", "List directory contents", False)]

    monkeypatch.setattr(cli._Engine, "build", classmethod(lambda cls, s, **k: One()))
    main(["--shell", "bash", "suggest", "--no-daemon", "--json", "list", "files"])
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {"command": "ls -1", "description": "List directory contents",
         "dangerous": False}
    ]


# -- daemon ---------------------------------------------------------------


def test_daemon_status_reports_a_missing_daemon_as_degraded(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "cl_ai.daemon.transport.request", lambda *a, **k: None, raising=True
    )
    assert main(["daemon", "status"]) == DEGRADED
    assert "not running" in capsys.readouterr().out


def test_daemon_stop_reports_a_missing_daemon(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "cl_ai.daemon.transport.request", lambda *a, **k: None, raising=True
    )
    assert main(["daemon", "stop"]) == DEGRADED
    assert "not running" in capsys.readouterr().out


def test_daemon_stop_waits_for_the_endpoint_to_be_released(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Observed: stop returned while the daemon was still listening, so the
    next start found the old process and reported "already running" -- leaving
    the user with exactly the daemon they had asked to replace.
    """
    from cl_ai.daemon.protocol import Kind, Response

    remaining = [True, True, False]

    def fake_request(message, endpoint=None, timeout_s=0.5):
        if message.kind is Kind.SHUTDOWN:
            return Response(message="shutting down")
        return Response() if remaining.pop(0) else None

    monkeypatch.setattr(
        "cl_ai.daemon.transport.request", fake_request, raising=True
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert main(["daemon", "stop"]) == OK
    assert not remaining, "should have polled until the endpoint went away"
    assert "stopped" in capsys.readouterr().out


def test_daemon_stop_gives_up_rather_than_hanging(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from cl_ai.daemon.protocol import Response

    clock = iter([0.0] + [float(i) for i in range(1, 200)])
    monkeypatch.setattr(
        "cl_ai.daemon.transport.request",
        lambda *a, **k: Response(),
        raising=True,
    )
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    assert main(["daemon", "stop"]) == BROKEN
    assert "still listening" in capsys.readouterr().err


def test_daemon_start_is_a_no_op_when_one_is_already_running(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from cl_ai.daemon.protocol import Response

    monkeypatch.setattr(
        "cl_ai.daemon.transport.request", lambda *a, **k: Response(), raising=True
    )

    def forbidden(*_a: object, **_k: object) -> object:
        raise AssertionError("must not spawn a second daemon")

    monkeypatch.setattr(cli.subprocess, "Popen", forbidden)
    assert main(["daemon", "start"]) == OK
    assert "already running" in capsys.readouterr().out
