"""Shell-profile tests.

Every test here corresponds to a bug that was actually present in the first
commit and found during review, not to a hypothetical.
"""

from __future__ import annotations

import pytest

from cl_ai.platform_ import (
    PROFILES,
    PathStyle,
    PipelineKind,
    QuoteStyle,
    detect,
    get_profile,
    is_available,
)


def test_profile_is_hashable():
    """Regression: a dict field made the frozen dataclass unhashable.

    The daemon keys warm state by profile, so an unhashable profile would fail
    only at runtime, in the cache, under load.
    """
    assert len({PROFILES["bash"], PROFILES["zsh"], PROFILES["bash"]}) == 2
    assert {PROFILES["powershell"]: "ok"}[PROFILES["powershell"]] == "ok"


def test_builtin_map_is_not_writable():
    """Regression: profiles shared one mutable dict.

    Writing through bash.builtin_map also altered zsh's and fish's, because
    frozen=True protects the field binding and not the object behind it.
    """
    with pytest.raises(TypeError):
        PROFILES["bash"].builtin_map["ls"] = "nonsense"  # type: ignore[index]


def test_profiles_do_not_leak_into_each_other():
    assert "ls" not in PROFILES["bash"].builtin_map
    assert PROFILES["powershell"].builtin_map["ls"] == "Get-ChildItem"
    assert PROFILES["cmd"].builtin_map["ls"] == "dir"


def test_bad_override_fails_loudly():
    """Regression: a typo'd CL_AI_SHELL silently fell back to bash.

    Silently handing back the wrong shell means the wrong quoter, which is the
    exact failure mode the quoting module exists to prevent.
    """
    with pytest.raises(KeyError) as excinfo:
        detect({"CL_AI_SHELL": "powersehll"})
    assert "powersehll" in str(excinfo.value)
    assert "powershell" in str(excinfo.value)  # lists the valid set


def test_override_wins_over_environment():
    assert detect({"CL_AI_SHELL": "fish", "PSModulePath": "x"}).id == "fish"


def test_detect_distinguishes_powershell_editions():
    assert detect({"PSModulePath": "x"}).id == "powershell"
    assert detect({"PSModulePath": "x", "PSEdition": "Core"}).id == "pwsh"


def test_detect_reads_posix_shell_variable():
    assert detect({"SHELL": "/usr/bin/zsh"}).id == "zsh"
    assert detect({"SHELL": "/bin/bash"}).id == "bash"
    # Windows builds append .exe; the suffix must not defeat the lookup.
    assert detect({"SHELL": "C:/Program Files/Git/usr/bin/bash.exe"}).id == "bash"


def test_detect_falls_back_without_guessing_powershell():
    """An empty environment must not produce a Windows-shaped guess."""
    assert detect({}).id == "bash"


def test_unknown_profile_lists_the_alternatives():
    with pytest.raises(KeyError) as excinfo:
        get_profile("tcsh")
    assert "known:" in str(excinfo.value)


def test_pipeline_kinds_are_not_all_text():
    """The object/text split is the reason renderers cannot be shared."""
    assert PROFILES["bash"].pipeline is PipelineKind.TEXT
    assert PROFILES["powershell"].pipeline is PipelineKind.OBJECT


def test_pwsh_is_posix_pathed_but_powershell_quoted():
    """pwsh on Linux: the (os, shell) pair really is two independent axes."""
    p = PROFILES["pwsh"]
    assert p.path_style is PathStyle.POSIX
    assert p.quote_style is QuoteStyle.POWERSHELL


def test_is_available_never_raises_for_unknown_shell():
    assert is_available("no-such-shell") is False
