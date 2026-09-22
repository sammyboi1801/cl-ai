"""Make CI failures legible from outside the runner.

GitHub only serves raw job logs to authenticated callers, but check-run
*annotations* are public for a public repo. So when running under Actions we
re-emit each test failure as a `::error::` workflow command, which becomes an
annotation. The practical effect is that anyone debugging a red build -- with
or without a token, from a script or a browser -- can read why it failed
instead of only that it did.

Off outside CI, where the normal pytest report is better.
"""

from __future__ import annotations

import os

_ON_CI = os.environ.get("GITHUB_ACTIONS") == "true"
_MAX_ANNOTATIONS = 25          # GitHub renders at most ~50; leave headroom
_MAX_CHARS = 900               # keep each one readable in the UI


def _escape(text: str) -> str:
    """Workflow commands are newline-delimited, so newlines must be encoded."""
    return (
        text.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _ON_CI:
        return

    failed = terminalreporter.stats.get("failed", [])
    errored = terminalreporter.stats.get("error", [])
    reports = (failed + errored)[:_MAX_ANNOTATIONS]

    for report in reports:
        raw = str(getattr(report, "longrepr", "") or "")
        # Keep the assertion, not the source listing around it. pytest prefixes
        # the actual message lines with "E ", and those are the only part that
        # says what went wrong; the rest is context we can already read locally.
        lines = [ln[1:].strip() for ln in raw.splitlines() if ln.startswith("E ")]
        detail = "\n".join(lines) if lines else raw[-_MAX_CHARS:]
        print(
            f"::error title={_escape(report.nodeid)[:200]}::"
            f"{_escape(detail[:_MAX_CHARS])}"
        )

    total = len(failed) + len(errored)
    if total > len(reports):
        print(f"::error::{total - len(reports)} further failures were not annotated")

    # A one-line census, so the shape of a failure is visible at a glance:
    # "every pwsh case" reads very differently from "three fish cases".
    if total:
        by_shell: dict[str, int] = {}
        for report in failed + errored:
            shell = "other"
            for candidate in ("bash", "zsh", "fish", "powershell", "pwsh", "cmd"):
                if f"{candidate}]" in report.nodeid or f"{candidate}-" in report.nodeid:
                    shell = candidate
                    break
            by_shell[shell] = by_shell.get(shell, 0) + 1
        census = " ".join(f"{k}={v}" for k, v in sorted(by_shell.items()))
        print(f"::error title=failure census::{_escape(census)}")
