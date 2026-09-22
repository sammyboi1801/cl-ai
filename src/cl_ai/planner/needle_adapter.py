"""Needle 3 backend.

Not implemented yet, on purpose. What follows is what measurement says the
adapter must and must not do, so that whoever writes it does not re-derive
it. Schema generation is already done, in schema.py.

WHAT NEEDLE IS GOOD AT
Extracting argument spans. Cactus are explicit that "a call contains only
values evidenced by the request", and that holds up: given a sane schema and
the right tool, it copies the value out of the query. Measured over eight
queries whose answer is literally in the text, once schema.py replaced the
naive conversion:

    a call with arguments   5/8 -> 7/8
    the span was copied     4/8 -> 5/8

It also refuses cleanly. "what is the weather in paris" and "send an email to
my manager" both return an empty call list with a sensible reason, fast.

WHAT IT IS NOT GOOD AT: CHOOSING THE TOOL
Measured over the evaluation set, counting only cases where the CORRECT tool
was among the five declared -- so retrieval had already done its job:

    picked the right one            23/60  = 0.383
    returned no call at all         14

And with the answer plus four RANDOM distractors, which is as easy as this
gets:

    picked the right one           61/117  = 0.521

Chance is 0.20, so there is signal, but half the time it takes the wrong tool
off a table of five where one is obviously right. The failures are
near-siblings -- git_log -> git_commit_graph, npm_ci -> npm_install_ci_test,
git_rebase -> git_commits_since -- which is the same weakness everything else
in this system has.

CONSEQUENCE FOR THE ADAPTER
Declare ONE tool, not five: the one retrieval chose. That removes the
selection question entirely and leaves Needle doing the thing it is good at.
schema.py caps at five because that is the engine's ceiling, not because
five is a good number to hand it.

CONFIDENCE IS NOT A CORRECTNESS SIGNAL
Do not gate on it. Measured over 61 scored calls:

    correct calls        n=23   p50 0.76   mean 0.70   range 0.12-0.94
    WRONG calls          n=24   p50 0.70   mean 0.68   range 0.25-1.00
    refusals (no call)   n=14   p50 0.09   mean 0.09   range 0.00-0.37

The first two distributions are the same distribution. A threshold sweep
moves precision from 0.489 ungated to at best 0.571 at 0.70 -- and then
INVERTS: at >=0.95 there are three calls and all three are wrong.

What confidence does separate is answering from refusing, cleanly. That is
exactly what a repurposed calibration head would do -- it scores the model's
willingness to commit, not whether the commitment is right. Worth knowing
that this was claimed twice from four samples before anyone measured it.

So validation has to be structural: accept the arguments only when they
validate against the schema AND the call names the tool retrieval chose.
Anything else falls back to the example's placeholders, which is a working
answer rather than no answer.

OTHER QUIRKS TO OWN HERE, SO CORE NEVER SEES THEM
  - agent construction is per-toolset and costs ~0.3-1.4s, so it wants an LRU
    cache keyed by a hash of the declared schemas
  - refusal normalisation: an empty function_calls list is a refusal, and
    there is no free text to fall back on
  - `reasoning` is debug-only and frequently describes a different tool than
    the one actually called, e.g. "Error: wrong tool. Fixing git_commit ->
    git_stamp" on a call to neither
  - this checkpoint has no tool-retrieval head and no embedding head; see
    schema.py MAX_DECLARED_TOOLS and embedding/needle_embedder.py
"""
