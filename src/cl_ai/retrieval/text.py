"""Tokenisation shared by the index and the query.

One module, because an index and a query tokenised by different rules silently
fail to match. That bug does not raise; it just returns nothing, which is
indistinguishable from "no such tool".

The corpus here is command documentation, which makes ordinary text-search
assumptions wrong in specific ways:

* Command names carry structure. `git-commit`, `git_commit` and `git commit`
  are the same thing, and `Get-ChildItem` is two words a user may type
  separately. So separators split, and CamelCase splits -- while the joined
  form is kept as well, because a user typing `childitem` must also match.
* Digits are part of names, not noise. `7z`, `base64`, `python3`, `md5sum`
  all lose their identity if digits are stripped.
* Stopwords must NEVER be removed from documents. `who`, `which`, `at`, `do`,
  `test`, `for`, `in` and `true` are all real commands. A stoplist applied to
  the corpus would make them unfindable -- the exact failure that looks like
  the tool not existing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = [
    "QUERY_STOPWORDS",
    "normalize_command_text",
    "tokenize",
    "tokenize_query",
    "unique",
]

# Split on anything that is not a letter or a digit. Unicode letters are kept:
# a description may legitimately be non-ASCII, and dropping those characters
# would silently truncate the token rather than leave it alone.
_SEPARATORS = re.compile(r"[^0-9A-Za-zÀ-￿]+")

# A lower-to-upper transition, or an acronym followed by a word: `ChildItem`,
# `HTTPServer`. Used to split identifiers, not prose.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

#: Removed from QUERIES only, never from documents.
#:
#: These are the filler words people type when they phrase a request as a
#: question -- "how do i list files". Left in, they hurt twice: they add terms
#: that no tool matches, which drags down the coverage floor, and a rare filler
#: word carries high IDF, so an accidental match outranks a real one.
#:
#: Deliberately small. Every word here is a word a user cannot search for, so
#: the list holds only words that are useless *as commands* and common *as
#: filler*. `all`, `list`, `show`, `find` and `new` are excluded on purpose:
#: they are what the user actually means.
QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "how", "do", "does", "did", "i", "me", "my", "we",
    "to", "of", "on", "please", "can", "could", "would", "should",
    "want", "wanna", "need", "am", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "there",
    "what", "whats", "s", "t",
})


def _split_identifier(token: str) -> list[str]:
    """`ChildItem` -> ["childitem", "child", "item"].

    The joined form comes first and the parts follow, so both `childitem` and
    `child` find the tool. Emitting only the parts would break an exact-name
    search; emitting only the joined form would break a natural-language one.
    """
    pieces = [p for p in _CAMEL.split(token) if p]
    if len(pieces) <= 1:
        return [token.lower()]
    return [token.lower(), *(p.lower() for p in pieces)]


def tokenize(text: str) -> list[str]:
    """Tokenise a document. Never removes stopwords.

    Duplicates are preserved: BM25 needs term frequencies, and collapsing them
    here would silently turn it into a set-membership score.
    """
    out: list[str] = []
    for raw in _SEPARATORS.split(text):
        if not raw:
            continue
        out.extend(_split_identifier(raw))
    return out


def tokenize_query(text: str) -> list[str]:
    """Tokenise what the user typed, dropping filler.

    Filler is dropped only when something survives. A query of nothing but
    stopwords is far more likely to be someone searching for a command that
    happens to be a common word -- `which`, `do`, `test` -- than a query with
    no content, so in that case the words are kept verbatim.
    """
    tokens = tokenize(text)
    kept = [t for t in tokens if t not in QUERY_STOPWORDS]
    return kept or tokens


def normalize_command_text(text: str) -> str:
    """Collapse whitespace in a command string for display and dedup.

    Used so two examples that differ only in spacing are recognised as one
    suggestion rather than filling two slots in a five-slot list.
    """
    return " ".join(text.split())


def unique(tokens: Iterable[str]) -> list[str]:
    """Deduplicate while preserving order.

    Order matters because it decides tie-breaks downstream, and a set would
    make those vary between runs.
    """
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out
