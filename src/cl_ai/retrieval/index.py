"""Two-stage index.

Stage 1 picks the binary (~dozens, well separated) -- this is where the context
engine is strongest. Stage 2 picks the leaf within it (aws_s3_cp vs aws_s3_mv),
a much easier ranking problem once the binary is fixed.

Fine-grained leaves cost the model nothing (only the top 5 are ever declared),
but they move the discrimination burden onto retrieval. This two-stage split is
what keeps that tractable.

Owns the score floor that yields an honest "no match" instead of the
nearest-neighbour substitution that turns a missing tool into a wrong command.

HOW THE TWO STAGES ARE ACTUALLY COMBINED, AND WHY NOT AS A CASCADE
------------------------------------------------------------------
A strict cascade -- pick the binary, then only ever look inside it -- has an
unrecoverable failure: if stage 1 is wrong, no amount of stage-2 skill helps.
And stage 1 is precisely where the measured error was. `ls` ranking second for
"list all running containers" is a stage-1 mistake; under a cascade, choosing
`ls` means `docker ps` can never be reached.

So the binary signal is used as a BOOST over a flat leaf ranking rather than as
a filter. Leaves are always all reachable, a strong leaf match survives a weak
binary score, and the discrimination benefit of the binary view is still there.
Per-binary capping then does the other job a cascade was meant to do: stop one
binary's fifty subcommands from filling a five-slot list.

THE PREFIX SIGNAL
The buffer is not always a sentence. A user types `git com` and presses Tab,
and they expect `git commit` -- that is a prefix of a name, not a description
match, and BM25 cannot see it because `com` is not a term in the index. So a
separate prefix ranker runs over names and binaries and is fused in. Dropping
it would regress the one behaviour the placeholder implementation already had.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from cl_ai.ir import Capability, ContextFacts, Tool

from .bm25 import Bm25Index, Field
from .fuse import fuse_ranked_ids
from .text import tokenize_query

__all__ = [
    "DEFAULT_FIELDS",
    "MIN_COVERAGE",
    "Candidate",
    "ToolIndex",
]

#: Field weights. The name is worth most: a user typing `git commit` means that
#: tool, not every tool whose description mentions committing. Examples are
#: weighted lowest but not zero -- they carry the natural-language phrasing
#: ("commit staged files") that a description often lacks, which is the whole
#: reason tier 4 was worth building.
#:
#: `b` is lowered for the name field. Length normalisation punishes long fields,
#: and a name is two or three tokens; normalising it like prose would rank
#: `git` above `git commit` for the query "git commit" purely on brevity.
#: The weights are measured, not guessed. Against a 31-query evaluation set:
#: name-heavy weighting (3.0/1.5/0.8, no intents field) scored hit@1 0.20,
#: because a tool whose NAME contains a query word beat one whose description
#: actually described the query -- "copy a file" returned the tool literally
#: named `file`. Demoting the name and adding the intents field took hit@1 to
#: the value recorded in tests/test_retrieval_quality.py.
DEFAULT_FIELDS: tuple[Field, ...] = (
    #: Names still matter for exact and near-exact recall, but they are no
    #: longer allowed to dominate: a name is a label, not a description of
    #: intent, and command names are short enough that any single shared token
    #: looks like a strong match.
    #: `b` is low across the board, and that is a corpus-specific correction,
    #: not a tuning whim. Length normalisation exists to stop a long document
    #: matching everything -- but here a "document" is a TOOL, and a tool with
    #: more documentation is more useful, not more diluted. At the textbook
    #: b=0.75, `git commit` ranked 287th for the query `git com` while
    #: `git commit-graph` ranked near the top, purely because git commit has
    #: eight examples and commit-graph has three. The normalisation was
    #: actively ranking the best-documented tools last.
    Field(name="name", weight=1.2, b=0.4),
    #: What the tool IS.
    Field(name="description", weight=2.0, b=0.5),
    #: What people WANT it for -- the example descriptions. This is the single
    #: most valuable retrieval field in the corpus, because it is the only text
    #: phrased the way a user phrases a request: "commit staged files with the
    #: specified message" sits against `git commit -m`. It is weighted highest
    #: for exactly that reason.
    Field(name="intents", weight=2.4, b=0.25),
    #: The command text itself. Lowest weight: it matches flags and tool names
    #: a user rarely types in a natural-language query, but it does catch the
    #: case where someone half-remembers a flag.
    Field(name="commands", weight=0.6, b=0.25),
)

#: Fraction of a query's IDF mass that must be matched for a result to count.
#:
#: This is the score floor, expressed in the only scale that means anything.
#: A raw BM25 threshold cannot work: scores are unbounded and shift with corpus
#: size, so any constant is either always or never exceeded. Coverage asks a
#: question with a stable answer instead -- "how much of what the user actually
#: asked for did this tool match?" -- weighted by IDF so a rare, meaningful
#: word counts for more than a common one.
#:
#: 0.3 is deliberately permissive. Returning nothing is a good outcome only
#: when there is genuinely nothing; the cost of a weak suggestion the user
#: ignores is one keystroke, while a missing suggestion looks like the tool not
#: existing.
MIN_COVERAGE = 0.3

#: At most this many leaves from one binary in the final list.
#:
#: Without it, "git" fills every slot with git_commit, git_push, git_pull... and
#: the user cycling five candidates sees one binary. Diversity is worth more
#: than the marginal relevance of a fourth git subcommand.
MAX_PER_BINARY = 2


@dataclass(frozen=True)
class Candidate:
    """One retrieved tool, with enough detail to explain the ranking."""

    tool: Tool
    score: float
    coverage: float = 0.0
    matched: frozenset[str] = frozenset()
    #: Which signals contributed, for debugging a bad ranking.
    signals: tuple[str, ...] = ()

    @property
    def dangerous(self) -> bool:
        return Capability.DESTRUCTIVE in self.tool.capabilities


def _document_for(tool: Tool) -> dict[str, str]:
    """Flatten one tool into indexable fields.

    The name is expanded to include its invocation form (`git commit` as well
    as `git_commit`) because a user types the latter and the catalog stores the
    former.
    """
    invocation = " ".join((tool.binary, *tool.path))
    return {
        "name": f"{tool.name} {invocation}",
        "description": tool.description,
        # Joined with newlines so tokenisation cannot run two examples' words
        # together into a phantom term.
        "intents": "\n".join(e.description for e in tool.examples if e.description),
        "commands": "\n".join(e.command for e in tool.examples),
    }


@dataclass
class ToolIndex:
    """A searchable index over a set of tools."""

    tools: tuple[Tool, ...]
    fields: tuple[Field, ...] = DEFAULT_FIELDS
    _leaf: Bm25Index = field(init=False)
    _binary_of: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        documents = [_document_for(t) for t in self.tools]
        self._leaf = Bm25Index.build(documents, self.fields)
        self._binary_of = tuple(t.binary for t in self.tools)

    @classmethod
    def from_catalog(
        cls,
        catalog: object,
        *,
        shell: str | None = None,
        os_name: str | None = None,
    ) -> ToolIndex:
        """Build from a Catalog, resolving variants for one (os, shell) target.

        Variant resolution happens HERE rather than at query time: a tool that
        cannot run on this machine should never enter the index, because a
        filter applied after ranking silently shortens the candidate list
        instead of promoting the next real match.
        """
        if shell is not None:
            tools = catalog.for_target(shell, os_name)  # type: ignore[attr-defined]
        else:
            tools = catalog.tools  # type: ignore[attr-defined]
        return cls(tools=tuple(tools))

    @cached_property
    def _by_binary(self) -> Mapping[str, tuple[int, ...]]:
        grouped: dict[str, list[int]] = {}
        for doc_id, binary in enumerate(self._binary_of):
            grouped.setdefault(binary, []).append(doc_id)
        return {b: tuple(ids) for b, ids in grouped.items()}

    def __len__(self) -> int:
        return len(self.tools)

    # -- signals ----------------------------------------------------------

    def _prefix_ranking(self, raw_query: str, terms: Sequence[str]) -> list[int]:
        """Documents whose name or binary starts with what the user is typing.

        Scored by how much of the name the prefix covers, so `pyth` prefers
        `python` over `pythonw`, and a whole-name match beats any prefix.
        """
        if not terms:
            return []
        # The LAST token is the one being typed; earlier tokens are context.
        # `git com` -> the user is completing "com" within git.
        tail = terms[-1]
        joined = " ".join(raw_query.split()).lower()

        scored: list[tuple[float, int]] = []
        for doc_id, tool in enumerate(self.tools):
            invocation = " ".join((tool.binary, *tool.path)).lower()
            name = tool.name.lower()
            best = 0.0
            if invocation == joined or name == joined:
                best = 3.0
            elif invocation.startswith(joined) or name.startswith(joined):
                best = 2.0 + len(joined) / max(len(invocation), 1)
            elif tool.binary.lower() == tail or name == tail:
                best = 1.5
            elif invocation.startswith(tail) or name.startswith(tail):
                best = 1.0 + len(tail) / max(len(invocation), 1)
            elif any(seg.lower().startswith(tail) for seg in tool.path):
                best = 0.8
            if best > 0.0:
                scored.append((best, doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc_id for _, doc_id in scored]

    def _whole_query_prefix_hits(self, raw_query: str) -> frozenset[int]:
        """Documents whose invocation starts with the ENTIRE typed query."""
        joined = " ".join(raw_query.split()).lower()
        if not joined:
            return frozenset()
        out = set()
        for doc_id, tool in enumerate(self.tools):
            invocation = " ".join((tool.binary, *tool.path)).lower()
            if invocation.startswith(joined) or tool.name.lower().startswith(joined):
                out.add(doc_id)
        return frozenset(out)

    def _prominence(self, tool: Tool) -> float:
        """Prefer the general-purpose tool over a specialised subcommand.

        A bare binary is what someone means by default; `git ls-files` is a
        specialist reading of "list files" and `ls` is the ordinary one. Depth
        is a genuine signal of specialisation, and without it a deep
        subcommand whose NAME happens to contain the query words outranks the
        canonical tool whose description actually matches.

        Kept mild. It is a tie-breaker between comparable matches, not a
        licence to bury subcommands -- `docker container ls` must still win
        "list running containers" over any bare binary.
        """
        return 1.0 / (1.0 + 0.30 * len(tool.path))

    def _binary_boost(self, terms: Sequence[str]) -> dict[int, float]:
        """Stage 1, applied as a boost: how well each BINARY matches the query.

        Aggregating at the binary level sees evidence a single leaf cannot. If
        several `docker` subcommands weakly match "containers", that is strong
        evidence for docker as a whole, and the boost lifts all of them.
        """
        if not terms:
            return {}
        per_binary: dict[str, float] = {}
        for scored in self._leaf.score(terms):
            binary = self._binary_of[scored.doc]
            per_binary[binary] = per_binary.get(binary, 0.0) + scored.score
        if not per_binary:
            return {}
        top = max(per_binary.values()) or 1.0
        return {
            doc_id: per_binary.get(self._binary_of[doc_id], 0.0) / top
            for doc_id in range(len(self.tools))
        }

    def _coverage(self, terms: Sequence[str], matched: frozenset[str]) -> float:
        """Share of the query's IDF mass that this document matched."""
        total = 0.0
        hit = 0.0
        for term in dict.fromkeys(terms):
            weight = self._leaf.idf(term)
            if weight <= 0.0:
                # A term the corpus has never seen is not evidence against any
                # tool -- it is usually a value the user typed, like a filename.
                # Counting it in the denominator would make every query with an
                # argument fall under the floor.
                continue
            total += weight
            if term in matched:
                hit += weight
        if total <= 0.0:
            return 1.0
        return hit / total

    # -- searching --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        context: ContextFacts | None = None,
        min_coverage: float = MIN_COVERAGE,
        max_per_binary: int = MAX_PER_BINARY,
        vectors: object | None = None,
    ) -> list[Candidate]:
        """Rank tools for `query`. Returns [] when nothing genuinely matches."""
        terms = tokenize_query(query)
        if not terms or not self.tools:
            return []
        # NO COMPOUND EXPANSION. Tried and measured, twice.
        #
        # Users type closed-up command names as separate words -- "check out",
        # "make a directory" -- so concatenating adjacent query tokens to reach
        # `checkout` and `mkdir` looks like an obvious win. It is not. Merged
        # into the query it dropped hit@5 from 0.733 to 0.700, because
        # `listfiles` matches `dpkg --listfiles` and displaced `ls` for "list
        # files". Fused in as a separate low-weight signal it was worse still:
        # hit@1 0.467 -> 0.433. Neither version actually fixed the two cases it
        # was written for, so it earned nothing and added noise.
        #
        # This gap is real and stays open: it is a semantic gap, and bridging
        # `make a directory` to `mkdir` is what the embedding half of the hybrid
        # is for. Guessing at it lexically makes ranking worse.

        lexical = self._leaf.score(terms)
        prefix = self._prefix_ranking(query, terms)
        if not lexical and not prefix and vectors is None:
            return []

        matched_by_doc = {s.doc: s.matched for s in lexical}
        lexical_order = [s.doc for s in lexical]

        # The prefix ranker is weighted above the lexical one. A name starting
        # with what the user is typing is near-certain evidence of intent,
        # while a BM25 rank in a corpus of thousands is a much weaker signal --
        # at equal weight, lexical noise pushed `git_commit` off the list
        # entirely for the query `git com`.
        # The vector half, when there is one. It RERANKS the lexical
        # candidates rather than searching, so a semantic near-miss can never
        # bury a tool the user's own words clearly matched. Fused by ORDER, so
        # its narrow cosine band cannot be mis-scaled against unbounded BM25
        # scores -- and its absence changes nothing, because RRF over one fewer
        # list is still RRF.
        semantic: list[int] = []
        if vectors is not None and getattr(vectors, "available", bool)():
            pool = [doc for doc, _ in fuse_ranked_ids([lexical_order, prefix])]
            # The returned ids are validated, not trusted. `vectors` is a
            # pluggable component behind a Protocol, and a backend that returns
            # a stale or out-of-range id would otherwise index into the wrong
            # tool -- or, as an early version did, raise IndexError from inside
            # a keystroke handler.
            semantic = [
                doc
                for doc in vectors.rerank(query, pool)  # type: ignore[attr-defined]
                if isinstance(doc, int) and 0 <= doc < len(self.tools)
            ]

        rankings = [lexical_order, prefix]
        weights = [1.0, 2.5]
        if semantic:
            rankings.append(semantic)
            # Equal to the lexical half rather than above it. The two make
            # different mistakes -- semantic-only ranked `ls` second for "list
            # all running containers" -- and neither has earned precedence on
            # this corpus.
            weights.append(1.0)
        fused = fuse_ranked_ids(rankings, weights=weights)
        boost = self._binary_boost(terms)
        # Full-query prefix matches, e.g. `git com` -> `git commit`. Tracked
        # separately from tail-only matches because only these may bypass the
        # coverage floor: for `git com`, `comma` and `comm` also match the tail
        # token while ignoring the `git` the user already typed.
        whole = self._whole_query_prefix_hits(query)

        installed = context.installed if context else frozenset()

        ranked: list[tuple[int, float, int]] = []
        for doc_id, rrf in fused:
            weighted = (
                rrf
                * (1.0 + 0.25 * boost.get(doc_id, 0.0))
                * self._prominence(self.tools[doc_id])
            )
            # Whole-query prefix matches form a PRIORITY TIER above everything
            # else, because that is what pressing Tab means. If what the user
            # typed is a prefix of a real command, that command is the answer,
            # and no amount of lexical similarity elsewhere should displace it.
            #
            # Without this tier, relevance and completion competed on one
            # scale and completion lost: `git com` returned `git commit-graph`
            # and `git commit-tree` while `git commit` -- the obvious answer --
            # fell off the list entirely.
            tier = 0 if doc_id in whole else 1
            ranked.append((tier, weighted, doc_id))
        # Inside the completion tier, the closest match wins: order by how much
        # of the candidate the typed text covers, so `git com` prefers
        # `git commit` over the longer `git commit-graph`.
        lengths = {
            doc_id: len(" ".join((self.tools[doc_id].binary, *self.tools[doc_id].path)))
            for _, _, doc_id in ranked
        }
        ranked.sort(
            key=lambda item: (
                item[0],
                lengths[item[2]] if item[0] == 0 else 0,
                -item[1],
                item[2],
            )
        )

        out: list[Candidate] = []
        per_binary: dict[str, int] = {}
        for _tier, score, doc_id in ranked:
            tool = self.tools[doc_id]

            # Hard gates first. These are not scores: a tool that is not
            # installed cannot be run, so ranking it lower is the wrong
            # answer -- it must not appear at all.
            if installed and tool.binary not in installed:
                continue

            matched = matched_by_doc.get(doc_id, frozenset())
            coverage = self._coverage(terms, matched)
            is_prefix_hit = doc_id in whole
            # Only a WHOLE-query prefix match bypasses the coverage floor.
            # `git com` covers almost none of the query's IDF mass -- `com` is
            # barely a corpus term -- yet `git commit` is unambiguously what
            # the user wants. A tail-only match does not earn the bypass:
            # `comma` and `comm` start with `com` too, and admitting them means
            # the list is filled by tools that ignore half the query.
            if coverage < min_coverage and not is_prefix_hit:
                continue

            count = per_binary.get(tool.binary, 0)
            if count >= max_per_binary:
                continue
            per_binary[tool.binary] = count + 1

            signals: list[str] = []
            if matched:
                signals.append("lexical")
            if is_prefix_hit:
                signals.append("prefix")
            if boost.get(doc_id, 0.0) > 0.0:
                signals.append("binary")

            out.append(
                Candidate(
                    tool=tool,
                    score=score,
                    coverage=coverage,
                    matched=matched,
                    signals=tuple(signals),
                )
            )
            if len(out) >= limit:
                break
        return out

    def explain(self, query: str, *, limit: int = 5) -> list[str]:
        """Human-readable ranking trace. For debugging a bad result."""
        return [
            f"{c.tool.name}  score={c.score:.4f} coverage={c.coverage:.2f} "
            f"signals={'+'.join(c.signals) or 'none'} matched={sorted(c.matched)}"
            for c in self.search(query, limit=limit)
        ]


def build_index(
    tools: Iterable[Tool], fields: Sequence[Field] | None = None
) -> ToolIndex:
    """Convenience constructor."""
    return ToolIndex(
        tools=tuple(tools), fields=tuple(fields) if fields else DEFAULT_FIELDS
    )
