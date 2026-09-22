"""Choosing which example lands in the prompt buffer.

The failure this guards is specific and nastier than a bad tool: `git_commit`
is the right tool for eleven intents, and returning "amend the last commit"
when the user asked to sign one is confident, plausible and wrong. Ranking
the tool correctly and then the example badly looks like success from every
angle except the user's.
"""

from __future__ import annotations

import pytest

from cl_ai.ir import Example, Tool
from cl_ai.retrieval.examples import (
    best_example,
    command_for,
    is_destructive,
    matched_terms,
    rank_examples,
)


def _tool(*pairs: tuple[str, str], name: str = "git_commit", **kwargs: object) -> Tool:
    return Tool(
        name=name,
        binary=kwargs.pop("binary", "git"),  # type: ignore[arg-type]
        path=kwargs.pop("path", ("commit",)),  # type: ignore[arg-type]
        description=kwargs.pop("description", "Commit files to the repository."),  # type: ignore[arg-type]
        examples=tuple(Example(description=d, command=c) for d, c in pairs),
    )


#: A real `git commit` page, abridged. Kept verbatim rather than invented,
#: because the whole difficulty is that these siblings are near-identical.
GIT_COMMIT = _tool(
    ("Open an editor to write a message and commit staged files", "git commit"),
    (
        "Commit staged files to the repository with the specified message",
        'git commit --message "message"',
    ),
    (
        "Commit staged files with a message read from a file",
        "git commit --file path/to/commit_message_file",
    ),
    (
        "Auto stage all modified and deleted files and commit with a message",
        'git commit --all --message "message"',
    ),
    (
        "Amend the last commit, changing its message",
        'git commit --amend --message "message"',
    ),
    (
        "Create a commit, even if there are no staged files",
        'git commit --message "message" --allow-empty',
    ),
)


# -- it picks the right sibling -------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("commit with a message", '--message "message"'),
        ("amend the last commit", "--amend"),
        ("read the commit message from a file", "--file"),
        ("stage everything and commit", "--all"),
        ("commit even with nothing staged", "--allow-empty"),
    ],
)
def test_the_distinguishing_flag_is_chosen(query: str, expected: str) -> None:
    chosen = best_example(GIT_COMMIT, query)
    assert chosen is not None
    assert expected in chosen.command, f"{query!r} -> {chosen.command!r}"


def test_a_term_common_to_every_sibling_decides_nothing() -> None:
    """Why IDF is computed WITHIN the tool, not across the corpus.

    Every `git commit` example contains "commit", so it carries no information
    here even though it is highly informative corpus-wide. If it counted, it
    would add the same weight everywhere and the ranking would collapse to
    the position tie-break.
    """
    plain = rank_examples(GIT_COMMIT, "commit")
    amend = rank_examples(GIT_COMMIT, "commit amend")
    assert plain[0][1].command == "git commit", "bare query -> the common usage"
    assert "--amend" in amend[0][1].command, "one distinguishing word must decide"


def test_a_query_matching_nothing_returns_the_first_example() -> None:
    """tldr orders a page by how common the usage is, so example zero is the
    maintainers' answer to "what does someone usually want here"."""
    chosen = best_example(GIT_COMMIT, "xyzzy plugh frobnicate")
    assert chosen is not None
    assert chosen.command == "git commit"


def test_a_word_in_the_flag_still_counts() -> None:
    """A user's word is often in the syntax rather than the prose."""
    tool = _tool(
        ("Do the usual thing", "tool --run"),
        ("Do something else", "tool --verbose"),
    )
    chosen = best_example(tool, "verbose")
    assert chosen is not None and "--verbose" in chosen.command


def test_the_description_outranks_the_command_for_the_same_term() -> None:
    """Prose is written as intent; syntax is written as syntax."""
    tool = _tool(
        ("Something unrelated entirely", "tool --archive"),
        ("Archive the output", "tool --zzz"),
    )
    chosen = best_example(tool, "archive")
    assert chosen is not None and chosen.command == "tool --zzz"


# -- degenerate inputs ----------------------------------------------------


def test_a_tool_with_no_examples_has_no_best() -> None:
    assert best_example(_tool(), "anything") is None
    assert rank_examples(_tool(), "anything") == []


def test_an_empty_query_still_returns_the_common_usage() -> None:
    """The widget can send an empty buffer; it must not crash or return None
    for a tool that plainly has an answer."""
    for query in ("", "   ", "\t\n"):
        chosen = best_example(GIT_COMMIT, query)
        assert chosen is not None and chosen.command == "git commit"


def test_a_query_of_pure_stopwords_returns_the_common_usage() -> None:
    chosen = best_example(GIT_COMMIT, "how do i the a")
    assert chosen is not None


def test_ranking_is_total_and_stable() -> None:
    """A user cycling candidates must not see them reshuffle."""
    first = [e.command for _, e in rank_examples(GIT_COMMIT, "commit a message")]
    second = [e.command for _, e in rank_examples(GIT_COMMIT, "commit a message")]
    assert first == second
    assert len(first) == len(GIT_COMMIT.examples), "every example must be ranked"


def test_ties_break_toward_the_earlier_example() -> None:
    tool = _tool(
        ("Identical wording here", "tool --first"),
        ("Identical wording here", "tool --second"),
    )
    chosen = best_example(tool, "identical wording")
    assert chosen is not None and "--first" in chosen.command


def test_position_can_never_outweigh_a_real_term_match() -> None:
    """The tie-break is 1e-6; a single IDF hit is order 0.5. Asserted because
    a tie-break that can overturn evidence is just a bug with a rationale."""
    many = _tool(*[(f"Usage number {i}", f"tool --n{i}") for i in range(500)])
    chosen = best_example(many, "usage number 499")
    assert chosen is not None and chosen.command == "tool --n499"


# -- command_for ----------------------------------------------------------


def test_command_for_returns_a_runnable_line() -> None:
    assert command_for(GIT_COMMIT, "amend the last commit") == (
        'git commit --amend --message "message"'
    )


def test_command_for_falls_back_to_the_bare_invocation() -> None:
    """Honest partial answer. `git bisect` with no arguments is still what the
    user meant, and beats substituting a neighbour that happens to have
    examples -- the failure mode `Tool.schematized` exists to name.
    """
    bare = Tool(name="git_bisect", description="", binary="git", path=("bisect",))
    assert command_for(bare, "find the bad commit") == "git bisect"


def test_the_fallback_is_typed_syntax_not_the_identifier() -> None:
    """`git_bisect` is not a command. Pasting it into a shell fails."""
    bare = Tool(name="git_bisect", description="", binary="git", path=("bisect",))
    assert "_" not in command_for(bare, "anything")


def test_a_subcommand_containing_an_underscore_survives() -> None:
    """Why invocation is derived from binary+path and not by un-mangling the
    name: splitting `name` on underscores corrupts this."""
    tool = Tool(
        name="dolt_sql_server",
        description="Start a MySQL-compatible server.",
        binary="dolt",
        path=("sql-server",),
    )
    assert command_for(tool, "start the server") == "dolt sql-server"


# -- explanation ----------------------------------------------------------


def test_matched_terms_reports_what_was_actually_matched() -> None:
    terms = matched_terms(GIT_COMMIT, "amend the last commit")
    assert "amend" in terms
    assert "the" not in terms, "stopwords are not evidence"


def test_matched_terms_is_empty_when_nothing_matched() -> None:
    assert list(matched_terms(GIT_COMMIT, "xyzzy plugh")) == []


def test_matched_terms_is_empty_for_a_tool_with_no_examples() -> None:
    assert list(matched_terms(_tool(), "anything")) == []


# -- against the real corpus ----------------------------------------------


def test_every_tool_in_the_corpus_yields_something_typeable(
    corpus_tools: list[Tool],
) -> None:
    """Corpus-wide invariant: there is always a non-empty line to insert.

    Deliberately NOT a placeholder check. The first version of this asserted
    `"}}" not in command` and failed on `koji call --kwargs '{"opts":{"scratch":
    True}}'` -- legitimate nested JSON, not a leaked template. That is the
    third time a brace invariant here has caught valid syntax, and the lesson
    has stuck: placeholder expansion is the extractor's job and is tested
    there, on the RawTool that still has the template to compare against. By
    the time a Tool reaches this module the evidence is gone, so a check here
    could only ever be a guess about punctuation.

    What this module CAN promise is that it never invents text: the result is
    an example's command verbatim, or the invocation. So that is what is
    asserted.
    """
    for tool in corpus_tools:
        command = command_for(tool, "do the thing")
        assert command.strip(), tool.name
        assert command == command.strip(), (tool.name, command)
        known = {e.command for e in tool.examples} | {tool.invocation}
        assert command in known, (tool.name, command)


def test_the_chosen_example_starts_with_a_plausible_binary(
    corpus_tools: list[Tool],
) -> None:
    """Loose on purpose: some examples legitimately start with `sudo`, a pipe
    or a different tool entirely (`tldr curl`). The check is that the line is
    shaped like a command, not that it names this exact tool."""
    for tool in corpus_tools:
        head = command_for(tool, "do the thing").split()[0]
        assert head, tool.name
        assert " " not in head


IPCONFIG = _tool(
    ("List all network adapters", "ipconfig"),
    ("Show a detailed list of network adapters", "ipconfig /all"),
    ("Renew the IP addresses for a network adapter", "ipconfig /renew adapter"),
    ("Free up the IP addresses for a network adapter", "ipconfig /release adapter"),
    name="ipconfig",
    binary="ipconfig",
    path=(),
    description="Display and manage the network configuration of Windows.",
)


@pytest.mark.xfail(
    reason="a matched noun is not a matched intent; needs an eval set to fix",
    strict=True,
)
def test_a_read_query_should_not_select_a_write_example() -> None:
    """Known wrong, pinned so it stays visible. See the module docstring.

    "IP" occurs only in the two examples that CHANGE the address, so the read
    intent scores them top. The correct answer shares no word with the query.
    Strict, so that whoever fixes it is told to delete this marker rather than
    leaving a passing xfail behind.
    """
    chosen = best_example(IPCONFIG, "what is my ip")
    assert chosen is not None
    assert "/renew" not in chosen.command and "/release" not in chosen.command


def test_the_right_tool_is_still_returned_for_that_query() -> None:
    """Bounding the damage above: the failure is example choice, not tool
    choice, and the line inserted is still a real `ipconfig` invocation."""
    assert command_for(IPCONFIG, "what is my ip").startswith("ipconfig")


# -- destructive examples need to be asked for -----------------------------


GIT_BARE = _tool(
    ("Create an empty Git repository", "git init"),
    ("Stage all changes for a commit", "git add --all"),
    ("Commit changes to version history", "git commit --message message_text"),
    (
        "Reset everything the way it was in the latest commit",
        "git reset --hard; git clean --force",
    ),
    name="git",
    binary="git",
    path=(),
    description="Distributed version control system.",
)


def test_a_destructive_example_does_not_win_on_an_incidental_word() -> None:
    """The measured failure this rule exists for.

    "Reset everything the way it was in the latest commit" matches both
    "everything" and "commit", and "everything" is rare among git's examples
    so it carries high IDF. The real commit example matches only "commit".
    Before this rule, asking to commit put `git reset --hard; git clean
    --force` in the buffer -- one Enter from discarding the work the user
    was trying to save.
    """
    chosen = best_example(GIT_BARE, "commit everything with a message please")
    assert chosen is not None
    assert chosen.command == "git commit --message message_text"


def test_a_destructive_example_still_wins_when_it_is_asked_for() -> None:
    """A penalty, not a ban. Suppressing the command someone explicitly
    asked for would be its own kind of wrong."""
    chosen = best_example(GIT_BARE, "reset everything to the last commit")
    assert chosen is not None
    assert "--hard" in chosen.command


@pytest.mark.parametrize(
    "query",
    [
        "discard all my local changes",
        "force remove these files",
        "wipe the working tree",
        "throw it all away and start over",
    ],
)
def test_various_phrasings_of_destructive_intent_are_recognised(query: str) -> None:
    from cl_ai.retrieval.examples import _wants_destruction
    from cl_ai.retrieval.text import tokenize_query

    assert _wants_destruction(tokenize_query(query)), query


@pytest.mark.parametrize(
    "query",
    ["commit my work", "show the log", "list the branches", "stage a file"],
)
def test_ordinary_queries_are_not_read_as_destructive(query: str) -> None:
    """A false positive here would hold back the right answer for a safe
    query, so the intent list must not be loose."""
    from cl_ai.retrieval.examples import _wants_destruction
    from cl_ai.retrieval.text import tokenize_query

    assert not _wants_destruction(tokenize_query(query)), query


@pytest.mark.parametrize(
    "command",
    [
        "git reset --hard; git clean --force",
        "rm -rf path/to/directory",
        "rm path/to/file",
        "docker system prune --all --volumes",
        "taskkill /im process_name",
        "shred --remove path/to/file",
        "dd if=file.iso of=/dev/usb",
        "truncate --size 0 path/to/file",
        "git branch --delete branch_name",
    ],
)
def test_destructive_commands_are_recognised(command: str) -> None:
    assert is_destructive(Example(description="d", command=command)), command


@pytest.mark.parametrize(
    "command",
    [
        "git commit --message \"message\"",
        "ls -la",
        "docker container ls",
        "git log --oneline",
        "cat path/to/file",
        "mkdir --parents path/to/dir",
    ],
)
def test_safe_commands_are_not_flagged(command: str) -> None:
    """A warning on everything is a warning on nothing."""
    assert not is_destructive(Example(description="s", command=command)), command


def test_a_destructive_verb_hidden_behind_a_separator_is_found() -> None:
    """`git clean` is the second half of a compound line. Splitting only on
    whitespace would miss it, and that exact line is the one that started
    all of this."""
    hidden = Example(description="d", command="git status; git clean -f")
    assert is_destructive(hidden)


def test_marking_is_judged_per_example_not_per_tool() -> None:
    """`git` is correctly NOT a destructive tool -- its examples span the
    whole of git. That is why the tool-level capability could never catch
    this, and why the check has to live here."""
    safe, destructive = GIT_BARE.examples[2], GIT_BARE.examples[3]
    assert not is_destructive(safe)
    assert is_destructive(destructive)
