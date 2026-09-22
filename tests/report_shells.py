"""Print which shells this machine can actually run, and what they really are.

Run as `python -m tests.report_shells`. Exists because "which shell am I
talking to" turned out not to be answerable from the binary name: on the
development machine `which bash` resolves to Git Bash while the process that
runs reports itself as WSL2, which changes where drive D appears in the
filesystem. CI logs should record what was really exercised, not what was
assumed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from cl_ai.platform_ import PROFILES, is_available


def _identify(shell_id: str) -> str:
    profile = PROFILES[shell_id]
    exe = profile.exec_flags[0]
    try:
        if shell_id in ("powershell", "pwsh"):
            argv = [exe, "-NoProfile", "-NonInteractive", "-Command",
                    "$PSVersionTable.PSVersion.ToString()"]
        elif shell_id == "cmd":
            argv = ["cmd", "/c", "ver"]
        else:
            argv = [exe, "-c", "uname -sr 2>/dev/null; echo $0 ${BASH_VERSION}${ZSH_VERSION}"]
        out = subprocess.run(argv, capture_output=True, timeout=30,
                             stdin=subprocess.DEVNULL, check=False)
        text = (out.stdout or out.stderr).decode("utf-8", "replace")
        return " | ".join(line.strip() for line in text.splitlines() if line.strip())[:120]
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        return f"<failed to identify: {exc}>"


def main() -> int:
    print("shell coverage on this machine")
    print("-" * 78)
    for shell_id in PROFILES:
        available = is_available(shell_id)
        path = shutil.which(PROFILES[shell_id].exec_flags[0]) or "-"
        mark = "yes" if available else "NO "
        print(f"  {shell_id:12} {mark}  {path}")
        if available:
            print(f"  {'':12}      {_identify(shell_id)}")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
