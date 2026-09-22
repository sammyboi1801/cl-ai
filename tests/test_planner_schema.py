"""Converting a catalog Tool into the schema Needle reads.

Needle is a SPAN EXTRACTOR: per Cactus's own guidance, "a call contains only
values evidenced by the request", and each argument is copied from a span of
the user's text. So the per-argument description is not documentation -- it
is the instruction saying which span to look for, and getting it wrong means
the model has no way to find the value.

Cactus name the effective forms ("City, ST", "e.g. T-1042", "the place after
'from'") and the ineffective ones (vague category language, instructions
aimed at the model). Every test here is about staying on the right side of
that line.
"""

from __future__ import annotations

import re

import pytest

from cl_ai.ir import Example, Param, ParamKind, Tool
from cl_ai.planner.schema import MAX_DECLARED_TOOLS, schema_for, schemas_for

IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


GIT_COMMIT = Tool(
    name="git_commit",
    description="Commit files to the repository",
    binary="git",
    path=("commit",),
    params=(
        Param(
            name="all", type="boolean", kind=ParamKind.OPTION, flag="--all",
            description="Auto stage all modified and deleted files and commit",
        ),
        Param(
            name="message", type="string", kind=ParamKind.OPTION,
            flag="--message", description="Commit staged files with the message",
        ),
        Param(
            name="file", type="string", kind=ParamKind.OPTION, flag="--file",
            description="Commit staged files with a message read from a file",
        ),
        Param(name="file", type="array", kind=ParamKind.POSITIONAL,
              description="Commit only specific files"),
    ),
    examples=(
        Example(description="Commit with a message",
                command='git commit --message "message"'),
        Example(description="Message from a file",
                command="git commit --file path/to/commit_message_file"),
    ),
)


# -- the overall shape ----------------------------------------------------


def test_the_schema_has_the_three_fields_needle_reads() -> None:
    schema = schema_for(GIT_COMMIT)
    assert set(schema) == {"name", "description", "parameters"}
    assert schema["parameters"]["type"] == "object"
    assert isinstance(schema["parameters"]["properties"], dict)


def test_the_name_is_the_leaf_not_the_binary() -> None:
    """One tool per action. A bare `git` carrying eight unrelated examples is
    the catch-all Cactus warn against -- it pushes the decision into free-text
    arguments the model has to invent."""
    assert schema_for(GIT_COMMIT)["name"] == "git_commit"


def test_nothing_is_marked_required() -> None:
    """A required field with no span suppresses the WHOLE call, while an
    optional one is simply omitted. Partial arguments plus the example's
    placeholder for the rest beats no suggestion at all."""
    assert "required" not in schema_for(GIT_COMMIT)["parameters"]


# -- the per-argument description is the whole game -----------------------


def test_a_value_argument_carries_a_concrete_example() -> None:
    """The "e.g. T-1042" form. The literal is recovered from the example
    COMMAND, where the extractor already substituted the placeholder, so the
    hint is grounded in the page rather than invented."""
    described = schema_for(GIT_COMMIT)["parameters"]["properties"]["file"]
    assert "e.g. path/to/commit_message_file" in described["description"]


def test_a_description_never_just_repeats_the_task() -> None:
    """The defect this module exists to fix: every one of 34,840 extracted
    params carried the EXAMPLE's description verbatim, so `message` was
    documented as "Commit staged files to the repository with the specified
    message" -- the task, not the span."""
    properties = schema_for(GIT_COMMIT)["parameters"]["properties"]
    assert properties["message"]["description"] != (
        "Commit staged files with the message"
    )
    assert "message" in properties["message"]["description"]


def test_an_argument_with_no_usable_literal_names_its_flag() -> None:
    """Cactus's effective form is positional -- "the place after 'from'".
    For a CLI the equivalent landmark is the flag."""
    properties = schema_for(GIT_COMMIT)["parameters"]["properties"]
    assert "--message" in properties["message"]["description"]


def test_a_sample_that_merely_repeats_the_name_is_dropped() -> None:
    """tldr often substitutes a placeholder with its own name, so the
    recovered literal is the slug again: "message, e.g. message" tells a
    model nothing."""
    tool = Tool(
        name="t", description="d", binary="t",
        params=(Param(name="message", type="string", kind=ParamKind.OPTION,
                      flag="-m", description="x"),),
        examples=(Example(description="d", command="t -m message"),),
    )
    described = schema_for(tool)["parameters"]["properties"]["message"]
    assert "e.g. message" not in described["description"]


def test_a_boolean_keeps_the_prose_that_says_when_to_set_it() -> None:
    """A flag has no value to point at, so WHEN is the useful information."""
    described = schema_for(GIT_COMMIT)["parameters"]["properties"]["all"]
    assert "Auto stage" in described["description"]


def test_tldr_mnemonic_markup_is_stripped() -> None:
    """`[c]reate a g[z]ipped archive` is for a human reading the page."""
    tool = Tool(
        name="tar", description="Archiving utility.", binary="tar",
        params=(Param(name="wildcards", type="boolean", kind=ParamKind.OPTION,
                      flag="--wildcards",
                      description="E[x]tract files from a `tar` archive"),),
    )
    text = schema_for(tool)["parameters"]["properties"]["wildcards"]["description"]
    assert "[x]" not in text and "`" not in text
    assert "Extract files from a tar archive" in text


def test_an_enum_is_declared_as_a_constraint() -> None:
    """Constraints belong in the grammar, not in prose: an invalid value
    should be unrepresentable rather than discouraged."""
    tool = Tool(
        name="t", description="d", binary="t",
        params=(Param(name="mode", type="string", kind=ParamKind.OPTION,
                      flag="--mode", enum=("soft", "hard"), description="x"),),
    )
    described = schema_for(tool)["parameters"]["properties"]["mode"]
    assert described["enum"] == ["soft", "hard"]
    assert "soft" in described["description"]


def test_descriptions_are_bounded() -> None:
    """Schemas share context with the system prompt and history, and
    needle_init fails outright if the static prefix does not fit."""
    tool = Tool(
        name="t", description="x" * 900, binary="t",
        params=(Param(name="a", type="boolean", kind=ParamKind.OPTION,
                      flag="-a", description="y" * 900),),
    )
    schema = schema_for(tool)
    assert len(schema["description"]) < 200
    assert len(schema["parameters"]["properties"]["a"]["description"]) < 200


# -- names that a consumer can actually address ---------------------------


def test_a_numeric_flag_becomes_a_legal_boolean_switch() -> None:
    """`kill -9 pid` sends signal 9; it does not pass a value to `-9`. The
    extractor types these as options WITH values, which produced 1,213
    properties named "1", "9" and "100" across the corpus.
    """
    tool = Tool(
        name="kill", description="Send a signal.", binary="kill",
        params=(Param(name="9", type="string", kind=ParamKind.OPTION, flag="-9",
                      description="Immediately terminate a program"),),
    )
    properties = schema_for(tool)["parameters"]["properties"]
    assert "9" not in properties
    assert properties["opt_9"]["type"] == "boolean"


def test_every_property_name_is_an_identifier() -> None:
    tool = Tool(
        name="t", description="d", binary="t",
        params=(
            Param(name="9", type="string", kind=ParamKind.OPTION, flag="-9",
                  description="x"),
            Param(name="100", type="string", kind=ParamKind.OPTION, flag="-100",
                  description="x"),
        ),
    )
    for name in schema_for(tool)["parameters"]["properties"]:
        assert IDENTIFIER.match(name), name


def test_colliding_names_are_both_kept_and_distinguished() -> None:
    """`git commit --file MSG` and `git commit FILE...` are different things
    with one slug. In a properties dict the second silently clobbers the
    first, which is how an argument disappears without any error."""
    properties = schema_for(GIT_COMMIT)["parameters"]["properties"]
    assert properties["file"]["type"] == "string"
    assert properties["files"]["type"] == "array"


def test_subcommands_are_not_arguments() -> None:
    """`git commit` is the tool. Declaring subcommand="commit" as an argument
    invites the model to fill it, and the renderer would emit it twice."""
    tool = Tool(
        name="pacman_sync", description="Sync packages.", binary="pacman",
        path=("--sync",),
        params=(
            Param(name="sync", type="string", kind=ParamKind.SUBCOMMAND,
                  flag="--sync", enum=("sync",), description="x"),
            Param(name="package", type="string", kind=ParamKind.POSITIONAL,
                  description="x"),
        ),
    )
    properties = schema_for(tool)["parameters"]["properties"]
    assert "sync" not in properties
    assert "package" in properties


def test_an_array_declares_its_items() -> None:
    assert schema_for(GIT_COMMIT)["parameters"]["properties"]["files"]["items"] == {
        "type": "string"
    }


def test_a_tool_with_no_params_still_produces_a_valid_schema() -> None:
    """12.6% of the corpus. An empty properties object is honest -- there is
    nothing to fill -- and must not be a crash."""
    bare = Tool(name="ls", description="List directory contents.", binary="ls")
    schema = schema_for(bare)
    assert schema["parameters"]["properties"] == {}
    assert schema["description"]


def test_a_tool_with_no_description_falls_back_to_its_invocation() -> None:
    tool = Tool(name="git_bisect", description="", binary="git", path=("bisect",))
    assert schema_for(tool)["description"] == "git bisect"


# -- the declared set -----------------------------------------------------


def test_the_declared_set_is_capped() -> None:
    """Above five tools Needle runs its own contrastive retrieval and keeps
    the top five. We cap from OUR ranking instead, because two stacked
    rankers make a bad answer impossible to attribute to either."""
    tools = [
        Tool(name=f"t{i}", description=f"Tool {i}", binary=f"t{i}")
        for i in range(12)
    ]
    assert len(schemas_for(tools)) == MAX_DECLARED_TOOLS


def test_duplicate_tools_are_declared_once() -> None:
    """Two schemas with one name is ambiguous: a call naming it cannot be
    attributed, and the grammar is built over the union."""
    tool = Tool(name="ls", description="List.", binary="ls")
    assert len(schemas_for([tool, tool, tool])) == 1


def test_an_empty_candidate_set_yields_no_schemas() -> None:
    assert schemas_for([]) == []


# -- against the real corpus ----------------------------------------------


def test_every_tool_in_the_corpus_converts_cleanly(corpus_tools) -> None:
    """Corpus-wide invariant. A schema Needle rejects takes out the whole
    turn, not one argument, so this has to hold for everything."""
    for tool in corpus_tools:
        schema = schema_for(tool)
        assert schema["name"], tool.name
        assert schema["description"], tool.name
        for name, described in schema["parameters"]["properties"].items():
            assert IDENTIFIER.match(name), (tool.name, name)
            assert described["type"] in {
                "string", "integer", "number", "boolean", "array",
            }, (tool.name, name, described["type"])
            assert described.get("description"), (tool.name, name)
            # No tldr mnemonic markup survives in the PROSE half. The
            # "e.g. ..." half is a literal copied from a real command and is
            # left exactly as written: `comby` documents
            # `assert_eq!(:[a],` where `:[a]` is comby's own template
            # syntax, and `tar` has `path/to/f[.gz|.bz2]`. Stripping there
            # would corrupt the very hint the description exists to give.
            #
            # This is the fourth bracket invariant in this project to catch
            # valid syntax rather than a bug. The rule that keeps surviving:
            # only assert over the span you actually control.
            prose = described["description"].split(", e.g. ", 1)[0]
            assert not re.search(r"\[[a-zA-Z]\]", prose), (
                tool.name, name, described["description"],
            )


@pytest.mark.parametrize("field", ["name", "description"])
def test_no_schema_field_is_unbounded(corpus_tools, field: str) -> None:
    for tool in corpus_tools:
        assert len(schema_for(tool)[field]) < 250, tool.name
