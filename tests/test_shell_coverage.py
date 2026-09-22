"""Fail when a shell we claim to cover is not actually being exercised.

A skipped test renders as a pass. That is exactly how the quoting suite came to
look green locally while a third of its matrix -- zsh, fish, pwsh -- had never
run once, and while fish was very likely broken.

CI sets CL_AI_REQUIRED_SHELLS per runner. Locally the variable is unset and
these tests are inert, which is the intended asymmetry: a developer machine may
legitimately lack fish, a CI runner that is supposed to provide it may not.
"""

from __future__ import annotations

import os

import pytest

from cl_ai.platform_ import PROFILES, is_available

REQUIRED = [s for s in os.environ.get("CL_AI_REQUIRED_SHELLS", "").split(",") if s]


@pytest.mark.skipif(not REQUIRED, reason="CL_AI_REQUIRED_SHELLS unset (local run)")
def test_every_required_shell_is_present():
    missing = [s for s in REQUIRED if not is_available(s)]
    assert not missing, (
        f"these shells were required on this runner but are not installed: "
        f"{', '.join(missing)}. Install them or correct CL_AI_REQUIRED_SHELLS -- "
        f"do not let the suite pass by skipping them."
    )


@pytest.mark.skipif(not REQUIRED, reason="CL_AI_REQUIRED_SHELLS unset (local run)")
def test_required_shells_are_known_profiles():
    unknown = [s for s in REQUIRED if s not in PROFILES]
    assert not unknown, f"CL_AI_REQUIRED_SHELLS names unknown profiles: {unknown}"
