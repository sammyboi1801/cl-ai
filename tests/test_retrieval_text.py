"""Tests for tokenisation.

The failure mode this guards against is silent: a document and a query
tokenised by different rules simply never match, which is indistinguishable
from the tool not existing. So the properties here are mostly about the two
paths agreeing.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cl_ai.retrieval.text import (
    QUERY_STOPWORDS,
    normalize_command_text,
    tokenize,
    tokenize_query,
    unique,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("git commit", ["git", "commit"]),
        ("git-commit", ["git", "commit"]),
        ("git_commit", ["git", "commit"]),
        ("GIT COMMIT", ["git", "commit"]),
        ("  spaced   out  ", ["spaced", "out"]),
        ("", []),
        ("---", []),
    ],
)
def test_separators_and_case(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_all_command_name_spellings_agree() -> None:
    """`git-commit`, `git_commit` and `git commit` must be one thing.

    The catalog stores one spelling and users type another; if these diverged,
    the tool would be unfindable by the name printed in its own docs.
    """
    forms = ["git commit", "git-commit", "git_commit", "Git Commit"]
    assert len({tuple(tokenize(f)) for f in forms}) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ChildItem", ["childitem", "child", "item"]),
        ("Get-ChildItem", ["get", "childitem", "child", "item"]),
        ("HTTPServer", ["httpserver", "http", "server"]),
        ("lowercase", ["lowercase"]),
        ("ALLCAPS", ["allcaps"]),
    ],
)
def test_camelcase_splits_but_keeps_the_joined_form(
    text: str, expected: list[str]
) -> None:
    """Both `childitem` and `child` have to find Get-ChildItem.

    Emitting only the parts breaks an exact-name search; emitting only the
    joined form breaks a natural-language one.
    """
    assert tokenize(text) == expected


@pytest.mark.parametrize("name", ["7z", "base64", "python3", "md5sum", "bzip2", "s3cmd"])
def test_digits_are_part_of_names(name: str) -> None:
    """Stripping digits destroys the identity of these tools."""
    assert name.lower() in tokenize(name)


def test_term_frequencies_are_preserved() -> None:
    """BM25 needs counts; deduplicating here would silently make it a set."""
    assert tokenize("git git git") == ["git", "git", "git"]


def test_non_ascii_is_kept_not_dropped() -> None:
    """Dropping the characters would truncate the token rather than skip it."""
    assert tokenize("café über") == ["café", "über"]


# --------------------------------------------------------------------------
# The stoplist applies to queries only
# --------------------------------------------------------------------------

@pytest.mark.parametrize("command", ["who", "which", "at", "do", "test", "for", "in", "true"])
def test_stopwords_are_never_removed_from_documents(command: str) -> None:
    """These are all real commands.

    A stoplist applied to the corpus would make them unfindable -- the exact
    failure that looks like the tool not existing.
    """
    assert tokenize(command) == [command]


def test_query_filler_is_dropped() -> None:
    assert tokenize_query("how do i list files") == ["list", "files"]


def test_intent_words_are_not_filler() -> None:
    """`list`, `show`, `find`, `all`, `new` are what the user actually means."""
    for word in ("list", "show", "find", "all", "new", "delete", "copy"):
        assert word not in QUERY_STOPWORDS
        assert tokenize_query(word) == [word]


def test_a_query_of_pure_filler_keeps_its_words() -> None:
    """Someone typing `do` is far more likely searching for the command `do`.

    Returning [] would mean "no query", and the user would get nothing back
    for a tool that exists.
    """
    assert tokenize_query("do") == ["do"]
    assert tokenize_query("what") == ["what"]
    assert tokenize_query("the a an") == ["the", "a", "an"]


def test_query_tokenisation_matches_document_tokenisation_on_content() -> None:
    """Whatever survives filtering must be tokenised identically."""
    query = tokenize_query("git-commit")
    assert query == tokenize("git-commit")


@given(st.text(max_size=80))
@settings(max_examples=400, deadline=None)
def test_tokenize_never_raises_and_yields_clean_tokens(text: str) -> None:
    tokens = tokenize(text)
    for token in tokens:
        assert token
        assert token == token.lower() or not token.isascii()
        assert " " not in token


@given(st.text(max_size=80))
@settings(max_examples=400, deadline=None)
def test_query_tokens_are_a_subsequence_of_document_tokens(text: str) -> None:
    """tokenize_query may only drop tokens, never invent or reorder them.

    If it could, a query would search for terms no document was indexed under.
    """
    doc = tokenize(text)
    query = tokenize_query(text)
    assert set(query) <= set(doc)
    position = 0
    for token in query:
        position = doc.index(token, position) + 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ls   -la", "ls -la"),
        ("  git  commit  ", "git commit"),
        ("a\tb\nc", "a b c"),
        ("", ""),
    ],
)
def test_normalize_command_text(text: str, expected: str) -> None:
    assert normalize_command_text(text) == expected


def test_unique_preserves_order() -> None:
    """Order decides tie-breaks downstream; a set would make them vary."""
    assert unique(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
