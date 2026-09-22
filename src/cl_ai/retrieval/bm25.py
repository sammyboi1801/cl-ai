"""Lexical half of the hybrid. Fixes semantic-only errors -- measured: `ls` ranked
second for "list all running containers".

Okapi BM25 over weighted fields. Two implementation choices here are not
stylistic:

NON-NEGATIVE IDF
The textbook IDF, log((N - df + 0.5) / (df + 0.5)), goes NEGATIVE for any term
appearing in more than half the documents. In this corpus that is guaranteed:
"file", "directory" and "the" are in a large fraction of command descriptions.
A negative weight means matching a common word actively *lowers* a tool's
score, so a tool whose description says "file" ranks below one that never
mentions files -- for the query "file". The +1 variant used below,
log(1 + (N - df + 0.5) / (df + 0.5)), is always positive and removes the whole
failure class.

FIELD WEIGHTS, NOT FIELD CONCATENATION
A tool's name, description and examples are indexed as separate fields with
separate length normalisation. Concatenating them would let a tool with fifty
examples swamp one with a precise name, because BM25 normalises by document
length and examples dominate the token count.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .text import tokenize

__all__ = ["Bm25Index", "Bm25Params", "Field", "Scored"]


@dataclass(frozen=True)
class Bm25Params:
    """Standard Okapi parameters.

    k1 controls how fast term-frequency saturates; b how strongly length is
    normalised. The defaults are the usual 1.2/0.75. b is lowered for the name
    field by the index, because a name is two or three tokens and normalising
    it as if it were prose punishes multi-word names like `git commit`.
    """

    k1: float = 1.2
    b: float = 0.75


@dataclass(frozen=True)
class Field:
    """One indexed field: its text weight and its length-normalisation."""

    name: str
    weight: float = 1.0
    b: float = 0.75


@dataclass(frozen=True)
class Scored:
    """A document with its score and the query terms it actually matched.

    `matched` is returned rather than discarded because the score floor is
    computed from which terms matched, not from the score -- see Retriever.
    """

    doc: int
    score: float
    matched: frozenset[str] = frozenset()


@dataclass
class Bm25Index:
    """An inverted index over a fixed set of documents.

    Built once, queried many times. Documents are addressed by integer id --
    the caller owns the mapping back to tools, so this class stays a pure
    text-ranking component with nothing catalog-specific in it.
    """

    fields: tuple[Field, ...]
    #: term -> field -> doc -> term frequency
    _postings: dict[str, dict[str, dict[int, int]]] = field(default_factory=dict)
    _doc_len: dict[str, dict[int, int]] = field(default_factory=dict)
    _avg_len: dict[str, float] = field(default_factory=dict)
    _doc_count: int = 0
    params: Bm25Params = field(default_factory=Bm25Params)
    _idf: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        documents: Sequence[Mapping[str, str]],
        fields: Sequence[Field],
        params: Bm25Params | None = None,
    ) -> Bm25Index:
        """Index `documents`, each a mapping of field name -> text."""
        index = cls(fields=tuple(fields), params=params or Bm25Params())
        postings: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        doc_len: dict[str, dict[int, int]] = {f.name: {} for f in fields}

        for doc_id, document in enumerate(documents):
            for spec in fields:
                tokens = tokenize(document.get(spec.name, ""))
                doc_len[spec.name][doc_id] = len(tokens)
                if not tokens:
                    continue
                counts: dict[str, int] = defaultdict(int)
                for token in tokens:
                    counts[token] += 1
                for token, count in counts.items():
                    postings[token][spec.name][doc_id] = count

        index._postings = {t: dict(f) for t, f in postings.items()}
        index._doc_len = doc_len
        index._doc_count = len(documents)
        index._avg_len = {
            spec.name: (
                sum(doc_len[spec.name].values()) / len(documents) if documents else 0.0
            )
            for spec in fields
        }
        index._idf = index._compute_idf()
        return index

    # -- internals --------------------------------------------------------

    def _compute_idf(self) -> dict[str, float]:
        """Document frequency is counted ACROSS fields, not per field.

        A term's rarity is a property of the corpus, not of where it happens to
        appear. Computing IDF per field would make a word that is common in
        descriptions look rare in names, and rank a coincidental name match
        above a real description match.
        """
        total = self._doc_count
        out: dict[str, float] = {}
        for term, by_field in self._postings.items():
            docs: set[int] = set()
            for docmap in by_field.values():
                docs.update(docmap)
            df = len(docs)
            # The +1 variant: always positive. See the module docstring.
            out[term] = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        return out

    def idf(self, term: str) -> float:
        """IDF of a term; 0.0 for a term the corpus has never seen.

        Zero rather than the maximum: an unknown term carries no evidence, and
        treating it as maximally informative would let a typo dominate.
        """
        return self._idf.get(term, 0.0)

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def vocabulary(self) -> int:
        return len(self._postings)

    # -- querying ---------------------------------------------------------

    def score(self, terms: Iterable[str], *, limit: int | None = None) -> list[Scored]:
        """Score every document that matches at least one term.

        Only matching documents are considered, so cost scales with the length
        of the posting lists rather than the size of the corpus.
        """
        query = list(terms)
        if not query or not self._doc_count:
            return []

        totals: dict[int, float] = defaultdict(float)
        matched: dict[int, set[str]] = defaultdict(set)
        k1 = self.params.k1

        for term in query:
            by_field = self._postings.get(term)
            if not by_field:
                continue
            weight = self.idf(term)
            if weight <= 0.0:
                continue
            for spec in self.fields:
                docmap = by_field.get(spec.name)
                if not docmap:
                    continue
                avg = self._avg_len[spec.name] or 1.0
                for doc_id, freq in docmap.items():
                    length = self._doc_len[spec.name][doc_id]
                    norm = 1.0 - spec.b + spec.b * (length / avg)
                    saturated = (freq * (k1 + 1.0)) / (freq + k1 * norm)
                    totals[doc_id] += weight * saturated * spec.weight
                    matched[doc_id].add(term)

        results = [
            Scored(doc=doc_id, score=score, matched=frozenset(matched[doc_id]))
            for doc_id, score in totals.items()
        ]
        # Descending score, then ascending doc id. The doc-id tie-break is what
        # makes two runs over the same index return the same order; without it
        # ordering follows dict iteration and the user sees candidates shuffle.
        results.sort(key=lambda s: (-s.score, s.doc))
        if limit is not None:
            return results[:limit]
        return results
