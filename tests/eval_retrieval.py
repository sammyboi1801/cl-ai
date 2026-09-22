"""Measure retrieval quality against the FULL tldr corpus.

Not a test -- a tool, like report_shells.py. The committed fixtures are ten
tools, which is enough to catch a regression but far too small to say anything
about ranking at scale. This script runs the same kind of evaluation against a
real corpus so the numbers quoted in test_retrieval_quality.py are reproducible
rather than merely asserted.

    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_retrieval
    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_retrieval --verbose

It reports two configurations, and the gap between them is the finding:

  unrestricted     every documented tool, including thousands never installed
  installed-gated  what a user on this machine actually experiences

Gating to installed binaries is a QUALITY signal, not just a correctness one.
Obscure tools are textually excellent matches -- `gdown` really is for
downloading a file -- and removing the ones that cannot be run promotes the
canonical answer. Measured here: hit@1 0.467 unrestricted, 0.633 gated.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from cl_ai.catalog.discovery import discover
from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.normalize import normalize
from cl_ai.ir import ContextFacts
from cl_ai.retrieval.index import ToolIndex

#: Realistic queries with the answers a person would accept. Several answers
#: per case because several tools genuinely satisfy some of these; insisting on
#: one would measure taste rather than retrieval.
CASES: tuple[tuple[str, frozenset[str]], ...] = (
    # Command fragments being completed.
    ("git com", frozenset({"git_commit"})),
    ("git pus", frozenset({"git_push"})),
    ("pyth", frozenset({"python", "python3"})),
    ("dock", frozenset({"docker"})),
    # Natural language.
    ("list files", frozenset({"ls", "dir"})),
    ("list files in a directory", frozenset({"ls", "dir"})),
    ("list all running containers", frozenset({"docker_ps", "docker_container_ls"})),
    ("commit staged files with a message", frozenset({"git_commit"})),
    ("undo the last commit", frozenset({"git_reset", "git_revert", "git_commit"})),
    ("show disk usage", frozenset({"du", "df", "duf"})),
    ("find text in files", frozenset({"grep", "rg", "ripgrep", "ack", "findstr"})),
    ("search for a pattern recursively", frozenset({"grep", "rg", "ripgrep", "ack"})),
    ("delete a directory recursively", frozenset({"rm", "rmdir"})),
    ("compress a folder into a zip", frozenset({"zip", "7z", "tar"})),
    ("extract a tar archive", frozenset({"tar", "7z"})),
    ("download a file from a url", frozenset({"curl", "wget"})),
    ("show running processes", frozenset({"ps", "tasklist", "top", "htop"})),
    ("kill a process by name", frozenset({"pkill", "killall", "taskkill", "kill"})),
    ("change file permissions", frozenset({"chmod", "icacls"})),
    ("create a new branch", frozenset({"git_branch", "git_switch", "git_checkout"})),
    ("clone a repository", frozenset({"git_clone"})),
    ("copy a file", frozenset({"cp", "copy", "robocopy", "xcopy"})),
    ("move a file", frozenset({"mv", "move"})),
    ("show the last lines of a file", frozenset({"tail"})),
    ("count lines in a file", frozenset({"wc"})),
    ("make a directory", frozenset({"mkdir", "md"})),
    ("check out a specific commit", frozenset({"git_checkout", "git_switch"})),
)


def evaluate(
    index: ToolIndex,
    *,
    context: ContextFacts | None = None,
    verbose: bool = False,
) -> dict[str, float]:
    hit1 = hit3 = hit5 = 0
    reciprocal = 0.0
    latencies: list[float] = []
    misses: list[tuple[str, list[str]]] = []

    for query, wanted in CASES:
        start = time.perf_counter()
        results = index.search(query, limit=5, context=context)
        latencies.append((time.perf_counter() - start) * 1000)
        names = [c.tool.name for c in results]
        rank = next((i for i, n in enumerate(names) if n in wanted), None)
        if rank is None:
            misses.append((query, names))
        else:
            reciprocal += 1.0 / (rank + 1)
            hit1 += rank < 1
            hit3 += rank < 3
            hit5 += rank < 5
        if verbose:
            mark = "ok  " if rank == 0 else ("~   " if rank is not None else "MISS")
            print(f"  {mark} {query!r:40} -> {', '.join(names[:4]) or '(none)'}")

    if misses and not verbose:
        for query, names in misses:
            print(f"  MISS {query!r:40} -> {', '.join(names[:4]) or '(none)'}")

    total = len(CASES)
    latencies.sort()
    return {
        "hit@1": hit1 / total,
        "hit@3": hit3 / total,
        "hit@5": hit5 / total,
        "mrr": reciprocal / total,
        "p50_ms": latencies[total // 2],
        "max_ms": latencies[-1],
    }


def main() -> int:
    root = os.environ.get("CL_AI_TLDR_ROOT")
    if not root:
        print("set CL_AI_TLDR_ROOT to a tldr corpus checkout", file=sys.stderr)
        return 2
    source = TldrSource(Path(root))
    if not source.available():
        print(f"{root} is not a tldr corpus", file=sys.stderr)
        return 2

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    catalog = normalize(source.harvest())
    shell = "powershell" if os.name == "nt" else "bash"
    os_name = "win32" if os.name == "nt" else "linux"
    index = ToolIndex.from_catalog(catalog, shell=shell, os_name=os_name)

    print(f"=== unrestricted ({len(index)} tools, shell={shell}) ===")
    for key, value in evaluate(index, verbose=verbose).items():
        print(f"  {key:8} {value:.3f}")

    inventory = discover()
    context = ContextFacts(
        os=os_name, shell=shell, installed=frozenset(inventory.names)
    )
    print(f"\n=== gated to {len(inventory.names)} installed binaries ===")
    for key, value in evaluate(index, context=context, verbose=verbose).items():
        print(f"  {key:8} {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
