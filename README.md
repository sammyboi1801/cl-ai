# cl-ai

Natural-language command completion for your shell. Type what you want, press
Tab, cycle the suggestions with the arrow keys, and the command lands in your
prompt — ready for you to read and run. It never executes anything itself.

```
> undo my last commit but keep the changes
  $ git reset --soft HEAD~1
```

**Status: early. Not usable yet.** The architecture is settled and two modules
are implemented and tested; most of the tree is stubs that record their
responsibility. See "Where this is" below.

## How it works

A resident daemon holds the model, a tool catalogue and an embedding index warm;
a thin per-shell widget sends the buffer to it and puts the answer back. The
daemon is required rather than an optimisation: a per-invocation process costs
~580 ms before it answers anything, while a warm one answers in ~68 ms, and Tab
has to feel instant.

```
 shell widget (Tab)  ──JSON over pipe/socket──▶  cl-aid
 PSReadLine / ZLE / readline / fish              model + index, warm
```

The model never writes shell text. It emits a structured plan, and per-shell
renderers turn that into syntax. That one decision is what makes pipelines and
cross-platform output the same problem, and what keeps the language model
replaceable — it is one adapter behind a narrow port, not a dependency of the
core.

## Cross-platform

The unit of targeting is the **(os, shell) pair**, because the two are
independent: PowerShell runs on Linux, bash runs on Windows, and a machine with
WSL presents several targets at once. Nothing outside `platform_.py` branches on
the operating system; everything else receives a `ShellProfile` and asks it
questions.

| Profile | Quoting | Pipeline |
|---|---|---|
| bash, zsh, fish | POSIX | text |
| powershell, pwsh | PowerShell | objects |
| cmd | cmd | untyped |

PowerShell's pipeline carries objects, not text, so a text-shaped plan rendered
naively into it *runs and returns the wrong answer* — which is worse than an
error, because nothing signals it. Plans therefore carry a stage type, and a
renderer refuses when no faithful translation exists.

**cmd.exe is not supported yet.** Its two parsing layers — cmd's own, then the
target program's argv rules — disagree about escaping, and the test oracle for
it is not yet faithful. Its tests are `xfail` rather than skipped, so they will
announce themselves when that changes.

## Testing

Quoting is the one place a bug is dangerous rather than annoying: the suggestion
lands in your buffer and you press Enter. So it is not tested against anyone's
beliefs about shells — every value is quoted, **executed in the real shell**,
and compared byte-for-byte with what comes back. Property-based tests generate
adversarial inputs on top of that.

That approach has already paid for itself. It caught, among others:

- `$` in a Python regex also matches before a trailing newline, so values ending
  in one were emitted unquoted — an unquoted newline ends the command and turns
  whatever follows into a second one. A command injection, found by Hypothesis
  on its first run.
- Flag-like values (`-rf`, `--force`) were quoted on PowerShell but left bare on
  POSIX, where the target program would read them as options rather than data.
- `which bash` resolving to Git Bash while the process that runs is WSL, which
  moves drive D from `/d` to `/mnt/d`.

CI runs the suite on Linux, macOS and Windows and **fails if a shell it is
supposed to cover is missing**, because a skipped test reads as a pass and that
is exactly how a third of the matrix went unexercised locally.

```sh
pip install -e ".[dev]"
pytest -q
python -m tests.report_shells      # what this machine can actually verify
```

## Layout

```
src/cl_ai/
  ir.py          the plan representation — the only currency the core speaks
  ports.py       Planner and Embedder, the two swappable seams
  platform_.py   ShellProfile and detection
  catalog/       discovery -> extract (6 source tiers) -> normalize -> lint
  retrieval/     two-stage: pick the binary, then the leaf
  context/       cwd markers, installed binaries, recent history
  planner/       Needle 3 adapter, a hostile adapter, validation
  render/        quoting and per-shell renderers
  daemon/        server, transport, protocol
  shell/         the per-shell widgets
```

## Where this is

Implemented and tested: `platform_.py`, `render/quoting.py`, and the contracts
in `ir.py` / `ports.py`.

Everything else is a stub carrying its responsibility and the measurement that
shaped it. Next up is `catalog/discovery.py` — open, exec-free, cross-platform.

## Licence

Apache-2.0.
