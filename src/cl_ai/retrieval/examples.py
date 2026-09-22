"""Choosing WHICH example answers the query.

Retrieval picks a tool; this picks the line that goes in the prompt buffer.
Both halves have to be right and only the first has been measured until now.
`git_commit` is the correct tool for eleven different intents -- amend, sign,
message, from a file -- and returning the wrong one of those is a worse
experience than returning the wrong tool, because it looks confident and
runs.

WHY THIS CAN BE DONE WITHOUT A MODEL
Because the extractor already substituted placeholders with concrete, legal
values. A tldr example is not a template needing a planner to fill it:

    git commit --message "message"
    mkdir --parents path/to/directory1 path/to/directory2 ...

That is runnable syntax with editable stand-ins, which is exactly the right
thing to land in a buffer the user is about to edit. A planner improves the
ARGUMENTS later; it is not needed to produce a command at all.

SCORING
Term overlap against the example's own description, IDF-weighted across the
examples of that one tool. Weighting within the tool is the point: every
`git commit` example contains "commit", so the word carries no information
HERE even though it is highly informative across the corpus. Only the terms
that distinguish one example from its siblings should decide between them.

The command text counts too, at a lower weight, because a user's word often
appears in the flag rather than in the prose -- "message" is in `--message`.

DEFAULTING TO THE FIRST EXAMPLE
When nothing matches, the answer is example zero, and that is a real choice
rather than a shrug. tldr pages are ordered by how common the usage is, so
the first line is the maintainers' answer to "what does someone usually want
from this command". Picking the shortest, or the one with fewest flags, both
measured worse on inspection: the shortest `tar` example is not the one
anybody wants.

KNOWN LIMITATION: A MATCHED NOUN IS NOT A MATCHED INTENT
Measured, "what is my ip" against `ipconfig`:

    +1.0296  ipconfig /renew adapter     Renew the IP addresses for a ...
    +1.0296  ipconfig /release adapter   Free up the IP addresses for a ...
    +0.0000  ipconfig                    List all network adapters
    -0.0000  ipconfig /all               Show a detailed list of network ...

The word "IP" appears only in the examples that CHANGE the address, never in
the ones that show it, so the read intent selects two write actions. Nothing
term-based can fix this: the right answer shares no word with the query, and
the distinction the user cares about lives entirely in the verb.

It is left wrong on purpose. The fix is a floor -- thin evidence should not
override the documented common usage -- but "thin" needs a threshold, and
setting one from this single case is how a corpus-wide regression gets
introduced to fix one query. That waits on an example-selection eval set,
which does not exist yet; the tool-selection set in tests/eval_retrieval.py
scores only which TOOL is returned and would score this case as a hit.
Pinned as an xfail in tests/test_retrieval_examples.py so it stays visible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cl_ai.ir import Example, Tool

from .text import tokenize, tokenize_query, unique

__all__ = ["best_example", "rank_examples"]

#: How much a term appearing in the command itself counts, relative to the
#: same term in the description. Below 1.0 because a description is written as
#: intent and a command is written as syntax -- matching `--message` is
#: evidence, matching the word "message" in prose is better evidence.
_COMMAND_WEIGHT = 0.45


def _idf(df: int, total: int) -> float:
    """Non-negative IDF, matching bm25.py so the two rank consistently."""
    return math.log(1.0 + (total - df + 0.5) / (df + 0.5))


def rank_examples(tool: Tool, query: str) -> list[tuple[float, Example]]:
    """Score every example, best first. Empty when the tool has none."""
    if not tool.examples:
        return []

    terms = set(tokenize_query(query))
    examples = tool.examples
    total = len(examples)

    # Tokenised once per example, then reused for both the document-frequency
    # pass and the scoring pass.
    described = [set(tokenize(e.description)) for e in examples]
    commanded = [set(tokenize(e.command)) for e in examples]

    frequency: dict[str, int] = {}
    for words, syntax in zip(described, commanded):
        for term in words | syntax:
            frequency[term] = frequency.get(term, 0) + 1

    scored: list[tuple[float, Example]] = []
    for position, example in enumerate(examples):
        score = 0.0
        for term in terms:
            df = frequency.get(term, 0)
            if df == 0:
                continue
            weight = _idf(df, total)
            if term in described[position]:
                score += weight
            elif term in commanded[position]:
                score += weight * _COMMAND_WEIGHT
        # Position breaks ties toward the earlier example, which on a tldr
        # page means the more common usage. Small enough that it can never
        # outweigh a real term match.
        scored.append((score - position * 1e-6, example))

    scored.sort(key=lambda item: -item[0])
    return scored


def best_example(tool: Tool, query: str) -> Example | None:
    """The example that best answers `query`, or None if the tool has none."""
    ranked = rank_examples(tool, query)
    return ranked[0][1] if ranked else None


def command_for(tool: Tool, query: str) -> str:
    """The exact text to put in the prompt buffer.

    Falls back to the bare invocation when a tool has no examples. That is an
    honest partial answer -- `git bisect` with no arguments is still what the
    user meant, and is better than substituting a neighbouring tool that does
    have examples. See `Tool.schematized`.
    """
    example = best_example(tool, query)
    return example.command if example is not None else tool.invocation


def matched_terms(tool: Tool, query: str) -> Sequence[str]:
    """Which query terms the chosen example actually accounts for.

    Exposed for explaining a suggestion rather than for ranking it: a user who
    does not trust a candidate is owed a reason, and "it matched these words"
    is one we can give without a model.
    """
    example = best_example(tool, query)
    if example is None:
        return ()
    known = set(tokenize(example.description)) | set(tokenize(example.command))
    return [t for t in unique(tokenize_query(query)) if t in known]
