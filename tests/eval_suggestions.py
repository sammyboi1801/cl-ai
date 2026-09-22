"""Score the full evaluation set against real retrieval.

A tool, not a test -- like report_shells.py and eval_retrieval.py. Needs the
full tldr corpus, which is not committed.

    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_suggestions
    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_suggestions --failures
    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_suggestions --os linux
    CL_AI_TLDR_ROOT=/path/to/tldr python -m tests.eval_suggestions --unrestricted

By default it gates to the binaries installed on THIS machine, because that
is what a user experiences, and because the gate is itself a quality signal:
obscure tools are excellent textual matches and removing what cannot be run
promotes the canonical answer.

Cases carrying an `os` are skipped unless they match the target, so running
this on Windows scores the Windows cases and reports the rest as skipped
rather than quietly failing them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from cl_ai.catalog.discovery import discover
from cl_ai.catalog.extract.tldr import TldrSource
from cl_ai.catalog.normalize import normalize
from cl_ai.daemon.server import _looks_destructive
from cl_ai.ir import ContextFacts
from cl_ai.retrieval.examples import best_example, command_for, is_destructive
from cl_ai.retrieval.index import ToolIndex

from .evalset import Case, Outcome, Report, load_cases, score


def run(
    cases: tuple[Case, ...],
    index: ToolIndex,
    context: ContextFacts | None,
    *,
    limit: int = 5,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for case in cases:
        start = time.perf_counter()
        results = index.search(case.query, limit=limit, context=context)
        elapsed = (time.perf_counter() - start) * 1000
        outcomes.append(
            Outcome(
                case=case,
                names=tuple(c.tool.name for c in results),
                commands=tuple(command_for(c.tool, case.query) for c in results),
                dangerous_flags=tuple(_dangerous(c, case.query) for c in results),
                elapsed_ms=elapsed,
            )
        )
    return outcomes


def _dangerous(candidate: object, query: str) -> bool:
    """Exactly the rule the daemon applies.

    Duplicated deliberately rather than imported from the daemon, which
    wants a Request. If the two ever disagree this measures something the
    user never sees, so the union is written out in full in both places and
    a test pins that they agree.
    """
    tool = candidate.tool  # type: ignore[attr-defined]
    chosen = best_example(tool, query)
    return bool(
        candidate.dangerous  # type: ignore[attr-defined]
        or _looks_destructive(tool.binary)
        or (chosen is not None and is_destructive(chosen))
    )


def report(name: str, result: Report, *, verbose: bool = False) -> None:
    example_rate, example_n = result.example_accuracy()
    nothing_rate, nothing_n = result.no_match_accuracy()
    danger_rate, danger_n = result.danger_recall()
    forbidden_rate, forbidden_n = result.forbidden_rate()
    p50, worst = result.latency()

    print(f"\n=== {name} ({len(result.outcomes)} cases) ===")
    print(f"  tool@1          {result.tool_at(1):.3f}")
    print(f"  tool@3          {result.tool_at(3):.3f}")
    print(f"  tool@5          {result.tool_at(5):.3f}")
    print(f"  mrr             {result.mrr():.3f}")
    print(f"  example         {example_rate:.3f}   (of {example_n} judgeable)")
    print(f"  says nothing    {nothing_rate:.3f}   (of {nothing_n} no-match cases)")
    print(f"  marks danger    {danger_rate:.3f}   (of {danger_n} destructive cases)")
    print(f"  WRONG neighbour {forbidden_rate:.3f}   (of {forbidden_n}; lower is better)")
    print(f"  latency         p50 {p50:.1f}ms  max {worst:.1f}ms")

    for attribute in ("category", "style", "persona"):
        print(f"\n  -- tool@1 by {attribute} --")
        for key, (rate, count) in sorted(
            result.by(attribute).items(), key=lambda kv: kv[1][0]
        ):
            bar = "#" * int(rate * 20)
            print(f"    {key:22} {rate:.2f}  n={count:<4} {bar}")

    if verbose:
        print("\n  -- failures --")
        for outcome in result.failures():
            case = outcome.case
            reason = (
                "WRONG-NEIGHBOUR"
                if outcome.forbidden_hit
                else ("BAD-EXAMPLE" if outcome.example_ok is False else "MISS")
            )
            got = ", ".join(outcome.names[:3]) or "(nothing)"
            print(f"    {reason:16} {case.query!r}")
            print(f"    {'':16} want {sorted(case.tools) or '(nothing)'}")
            print(f"    {'':16} got  {got}")
            if outcome.names:
                print(f"    {'':16} cmd  {outcome.commands[0]}")


def main() -> int:
    # Imported here rather than at module scope so `run` and `report` stay
    # usable without the case file -- which is how they were driven while the
    # cases were still being written.
    from .eval_cases import CASES

    parser = argparse.ArgumentParser(prog="eval_suggestions")
    parser.add_argument("--failures", action="store_true", help="list every failure")
    parser.add_argument("--os", default=sys.platform, help="target platform")
    parser.add_argument("--shell", default=None)
    parser.add_argument(
        "--unrestricted",
        action="store_true",
        help="do not gate to installed binaries",
    )
    parser.add_argument("--category", default=None, help="score one category only")
    args = parser.parse_args()

    root = os.environ.get("CL_AI_TLDR_ROOT")
    if not root:
        print("set CL_AI_TLDR_ROOT to a tldr corpus checkout", file=sys.stderr)
        return 2
    source = TldrSource(Path(root))
    if not source.available():
        print(f"{root} is not a tldr corpus", file=sys.stderr)
        return 2

    shell = args.shell or ("powershell" if args.os == "win32" else "bash")
    cases = load_cases(CASES)
    # A case pinned to another platform is not a failure here, it is out of
    # scope. Scoring it anyway would make the Windows run look broken for
    # every Linux case and drown the real signal.
    applicable = tuple(c for c in cases if c.os in (None, args.os))
    if args.category:
        applicable = tuple(c for c in applicable if c.category == args.category)
    skipped = len(cases) - len(applicable)

    catalog = normalize(source.harvest())
    index = ToolIndex.from_catalog(catalog, shell=shell, os_name=args.os)

    context = None
    label = f"unrestricted, {len(index)} tools, {shell}/{args.os}"
    if not args.unrestricted:
        inventory = discover()
        context = ContextFacts(
            os=args.os, shell=shell, installed=inventory.names
        )
        label = f"gated to {len(inventory.names)} installed, {shell}/{args.os}"

    if skipped:
        print(f"({skipped} cases skipped: pinned to another platform)")
    report(label, score(run(applicable, index, context)), verbose=args.failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
