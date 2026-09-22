"""Tests for the tier-4 (tldr) extractor.

Two layers, deliberately:

1. Unit tests over the placeholder grammar, driven by cases found by scanning
   the real 7,367-page corpus rather than invented. Every adversarial case
   below is a literal string from a real page.
2. Corpus tests over committed fixture pages -- authentic copies of the pages
   that exercise each edge case. They are committed because the full corpus
   lives outside the installable package and so is absent on CI; a test that
   skips there would report a pass while checking nothing.

The full corpus is additionally checked when CL_AI_TLDR_ROOT points at it,
which is how the invariants get exercised at scale locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cl_ai.catalog.extract.base import (
    OS_TO_SHELLS,
    ParamKind,
    RawExample,
    RawParam,
    RawTool,
    Source,
    infer_capabilities,
    shells_for_os,
)
from cl_ai.catalog.extract.tldr import (
    ENGLISH_DIRS,
    MAX_PAGE_BYTES,
    PlaceholderError,
    TldrSource,
    capabilities_for,
    default_root,
    infer_type,
    is_flag_token,
    parse_page,
    parse_placeholders,
    platforms_for,
    slot_identifier,
)
from cl_ai.ir import Capability, SourceTier

FIXTURES = Path(__file__).parent / "fixtures" / "tldr"


@pytest.fixture(scope="module")
def fixture_tools() -> dict[str, RawTool]:
    source = TldrSource(FIXTURES)
    assert source.available(), f"fixture corpus missing at {FIXTURES}"
    tools = list(source.harvest())
    assert tools, "fixture corpus produced no tools"
    return {t.qualified_name: t for t in tools}


# --------------------------------------------------------------------------
# Placeholder grammar
# --------------------------------------------------------------------------

def test_simple_placeholder() -> None:
    (ph,) = parse_placeholders("cat {{path/to/file}}")
    assert ph.content == "path/to/file"
    assert ph.alternatives == ()
    assert not ph.is_option


def test_option_alternation_splits_short_and_long() -> None:
    phs = parse_placeholders('git commit {{[-m|--message]}} "{{message}}"')
    assert len(phs) == 2
    assert phs[0].alternatives == ("-m", "--message")
    assert phs[0].is_option
    assert phs[1].content == "message"
    assert not phs[1].is_option


def test_alternation_order_is_not_assumed() -> None:
    """A long-first alternation must classify the same as short-first."""
    (ph,) = parse_placeholders("tool {{[--output|-o]}}")
    assert set(ph.alternatives) == {"--output", "-o"}
    params = _params("tool {{[--output|-o]}}")
    assert params[0].long == "--output"
    assert params[0].short == "-o"


def test_nested_brace_expansion_is_one_placeholder() -> None:
    """From pages/common/$.md -- a shell expansion inside a slot.

    A regex stopping at the first `}}` would cut this in half and emit a
    truncated command.
    """
    (ph,) = parse_placeholders("echo ${{{array_name[@]}}}")
    assert ph.content == "{array_name[@]}"


def test_trailing_backslash_in_placeholder() -> None:
    """From pages/dos/mount.md -- `{{A:\\}}` is a drive letter, not an escape."""
    (ph,) = parse_placeholders("MOUNT A {{A:\\}} -t floppy")
    assert ph.content == "A:\\"


def test_doubled_escape_is_a_literal_brace_not_a_slot() -> None:
    assert parse_placeholders(r"echo \{\{not a slot\}\}") == ()


def test_inconsistent_escaping_raises_rather_than_guesses() -> None:
    """From pages/common/aws-dynamodb.md.

    The opening brace is unescaped while its closing partner is escaped, so no
    single rule resolves it. Refusing is the point: a guess here would put a
    malformed command in the user's buffer.
    """
    with pytest.raises(PlaceholderError):
        parse_placeholders(
            'aws dynamodb put-item --item \'{{{"AttributeName": {"S": "value"\\}\\}}}\''
        )


def test_unterminated_placeholder_raises() -> None:
    with pytest.raises(PlaceholderError):
        parse_placeholders("cat {{path/to/file")


@pytest.mark.parametrize("text", ["", "ls", "ls -la", "echo hello world"])
def test_texts_without_placeholders(text: str) -> None:
    assert parse_placeholders(text) == ()


def test_adjacent_placeholders_are_separate() -> None:
    phs = parse_placeholders("{{a}}{{b}}")
    assert [p.content for p in phs] == ["a", "b"]
    assert phs[0].end == phs[1].start


def test_spans_slice_back_to_their_own_text() -> None:
    template = 'git commit {{[-m|--message]}} "{{message}}" {{path/to/file}}'
    for ph in parse_placeholders(template):
        span = template[ph.start : ph.end]
        assert span.startswith("{{") and span.endswith("}}")
        assert ph.content in span


# --------------------------------------------------------------------------
# Identifiers and types
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("path/to/file", "file"),
        ("path/to/directory", "directory"),
        ("path/to/commit_message_file", "commit_message_file"),
        ("path/to/file1 path/to/file2 ...", "file"),
        ("[-m|--message]", "message"),
        ("[-o|--output]", "output"),
        ("--amend", "amend"),
        ("-m", "m"),
        ("branch_name", "branch_name"),
        ("example.com", "example_com"),
        ("A:\\", "a"),
    ],
)
def test_slot_identifier(content: str, expected: str) -> None:
    assert slot_identifier(content) == expected


@given(st.text(max_size=60))
@settings(max_examples=300, deadline=None)
def test_slot_identifier_is_always_usable(content: str) -> None:
    """Never empty, never leading/trailing separators.

    An empty parameter name would make RawParam raise deep inside a catalog
    build over thousands of pages, where the traceback says nothing about
    which page caused it.
    """
    ident = slot_identifier(content)
    assert ident
    assert not ident.startswith("_")
    assert not ident.endswith("_")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("100", "integer"),
        ("-5", "integer"),
        ("3.5", "number"),
        ("path/to/file", "string"),
        ("path/to/file1 path/to/file2 ...", "array"),
        ("message", "string"),
    ],
)
def test_infer_type(content: str, expected: str) -> None:
    assert infer_type(content) == expected


# --------------------------------------------------------------------------
# Placeholder property: no corruption, ever
# --------------------------------------------------------------------------

@given(
    st.text(alphabet=st.sampled_from("{}\\|[]-abc. "), max_size=40)
)
@settings(max_examples=800, deadline=None)
def test_placeholders_never_corrupt(text: str) -> None:
    """For any input: either refuse, or return clean non-overlapping spans.

    This is the property that matters. The parser is allowed to reject input;
    it is not allowed to return spans that overlap, run backwards, or point at
    text that is not actually a placeholder -- any of which would silently
    produce a mangled command rather than an error.
    """
    try:
        phs = parse_placeholders(text)
    except PlaceholderError:
        return
    last_end = -1
    for ph in phs:
        assert 0 <= ph.start < ph.end <= len(text)
        assert ph.start >= last_end
        last_end = ph.end
        span = text[ph.start : ph.end]
        assert span.startswith("{{")
        assert span.endswith("}}")
        assert text[ph.start + 2 : ph.end - 2] == ph.content


# --------------------------------------------------------------------------
# Page parsing
# --------------------------------------------------------------------------

def _params(template: str) -> list[RawParam]:
    page = f"# tool\n\n> Desc.\n\n- Do the thing:\n\n`{template}`\n"
    tool = parse_page(page, source="t", os_target="common")
    assert tool is not None
    return list(tool.params)


def test_git_commit_page(fixture_tools: dict[str, RawTool]) -> None:
    tool = fixture_tools["git_commit"]
    assert tool.binary == "git"
    assert tool.path == ("commit",)
    assert tool.invocation == "git commit"
    assert tool.tier is SourceTier.TLDR
    assert "Commit files" in tool.description
    assert tool.homepage == "https://git-scm.com/docs/git-commit"
    assert tool.examples

    by_long = {p.long: p for p in tool.params if p.long}
    assert by_long["--message"].short == "-m"
    assert by_long["--message"].value_name == "message"
    # A literal flag taking no value must still be captured.
    assert "--amend" in by_long
    assert by_long["--amend"].value_name is None


def test_more_information_line_is_not_part_of_the_description(
    fixture_tools: dict[str, RawTool],
) -> None:
    tool = fixture_tools["git_commit"]
    assert "More information" not in tool.description
    assert "git-scm.com" not in tool.description


def test_examples_are_literalised(fixture_tools: dict[str, RawTool]) -> None:
    tool = fixture_tools["git_commit"]
    literals = [e.literal for e in tool.examples]
    assert 'git commit --message "message"' in literals
    for literal in literals:
        assert "{{" not in literal
        assert "}}" not in literal


def test_hyphenated_binary_is_not_split(fixture_tools: dict[str, RawTool]) -> None:
    """`apt-get` is one binary. The filename hyphen is not a separator."""
    tool = fixture_tools["apt_get"]
    assert tool.binary == "apt-get"
    assert tool.path == ()


def test_subcommand_may_contain_a_hyphen(fixture_tools: dict[str, RawTool]) -> None:
    """`pm install-commit` -- the title says where the split is; nothing else can."""
    tool = fixture_tools["pm_install_commit"]
    assert tool.binary == "pm"
    assert tool.path == ("install-commit",)


def test_flag_spelled_subcommands_stay_distinct(
    fixture_tools: dict[str, RawTool],
) -> None:
    """`pacman --sync` and `pacman --query` are different tools.

    Dropping the flag collapsed pacman's entire surface into one bare entry.
    """
    assert "pacman_sync" in fixture_tools
    assert "pacman_query" in fixture_tools
    assert "pacman" in fixture_tools
    sync = fixture_tools["pacman_sync"]
    assert sync.binary == "pacman"
    assert sync.path == ("--sync",)
    # The path token is kept verbatim so the rendered invocation is correct.
    assert sync.invocation == "pacman --sync"


def test_dotted_binary_with_flag_subcommand(fixture_tools: dict[str, RawTool]) -> None:
    tool = fixture_tools["acme_sh_dns"]
    assert tool.binary == "acme.sh"
    assert tool.path == ("--dns",)


def test_unparseable_examples_are_dropped_with_a_warning(
    fixture_tools: dict[str, RawTool],
) -> None:
    tool = fixture_tools["aws_dynamodb"]
    assert tool.warnings, "inconsistent escaping should be reported"
    assert all("unbalanced" in w for w in tool.warnings)
    # The page still yields its parseable examples rather than being discarded.
    for example in tool.examples:
        assert "{{" not in example.literal


def test_dos_page_with_backslash_placeholder(fixture_tools: dict[str, RawTool]) -> None:
    """`{{A:\\}}` is a drive letter ending in a backslash, not an escape."""
    # DOS pages title their commands in uppercase (`# MOUNT`), and that case is
    # preserved: DOS `MOUNT` and Linux `mount` are unrelated tools, so folding
    # them to one identifier would merge two different tools into one entry.
    tool = fixture_tools["MOUNT"]
    assert tool.examples
    assert not tool.warnings
    assert any("A:\\" in e.literal for e in tool.examples)


@pytest.mark.parametrize(
    ("page", "reason"),
    [
        ("", "empty"),
        ("   \n\n  \n", "blank only"),
        ("no title here\n\n> Desc.\n", "missing title"),
        ("# \n\n> Desc.\n", "empty title"),
    ],
)
def test_unusable_pages_return_none(page: str, reason: str) -> None:
    assert parse_page(page, source="t", os_target="common") is None, reason


def test_page_without_examples_still_yields_identity() -> None:
    tool = parse_page("# foo\n\n> Does a thing.\n", source="t", os_target="common")
    assert tool is not None
    assert tool.binary == "foo"
    assert tool.examples == ()
    assert tool.description == "Does a thing"


def test_option_value_is_not_also_a_positional() -> None:
    """`{{[-o|--output]}} {{path/to/file}}` is one option, not option+positional."""
    params = _params("tool {{[-o|--output]}} {{path/to/file}}")
    kinds = {(p.kind, p.name) for p in params}
    assert (ParamKind.OPTION, "output") in kinds
    assert (ParamKind.POSITIONAL, "file") not in kinds


def test_bare_positional_is_kept() -> None:
    params = _params("cat {{path/to/file}}")
    assert [(p.kind, p.name) for p in params] == [(ParamKind.POSITIONAL, "file")]


def test_repeatable_positional_detected() -> None:
    params = _params("cat {{path/to/file1 path/to/file2 ...}}")
    assert params[0].repeatable


def test_literal_flags_are_extracted() -> None:
    params = _params("git commit --amend --no-edit")
    longs = {p.long for p in params}
    assert longs == {"--amend", "--no-edit"}


def test_negative_numbers_are_not_flags() -> None:
    params = _params("tool -5 {{path/to/file}}")
    assert all(p.long != "-5" and p.short != "-5" for p in params)


def test_flags_inside_placeholders_are_not_double_counted() -> None:
    params = _params("git commit {{[-m|--message]}}")
    assert len([p for p in params if p.long == "--message"]) == 1


def test_subcommand_alternation_is_not_an_option() -> None:
    """`{{[images|image ls]}}` names two spellings of a subcommand.

    Classifying every alternation as an option produced params with
    kind=OPTION and no flag at all, which a renderer cannot emit. 386 distinct
    alternations in the corpus are of this shape.
    """
    params = _params("docker {{[images|image ls]}} --format json")
    subs = [p for p in params if p.kind is ParamKind.SUBCOMMAND]
    assert len(subs) == 1
    assert subs[0].choices == ("images", "image ls")
    assert subs[0].short is None
    assert subs[0].long is None


def test_flag_alternation_records_all_spellings() -> None:
    params = _params("git commit {{[-m|--message]}} {{message}}")
    param = next(p for p in params if p.long == "--message")
    assert param.choices == ("-m", "--message")


def test_dos_slash_options_are_recognised() -> None:
    """`{{[/L|/list]}}` is an option in cmd's convention, not a subcommand."""
    params = _params("tool {{[/L|/list]}}")
    assert len(params) == 1
    assert params[0].kind is ParamKind.OPTION
    assert params[0].short == "/L"
    assert params[0].long == "/list"


def test_unbracketed_flag_alternation() -> None:
    """`{{-s|-3}}` on android/pm-list.md omits the brackets.

    Previously this produced the nonsense flag `-s|-3`.
    """
    params = _params("pm list packages {{-s|-3}}")
    assert params[0].kind is ParamKind.OPTION
    assert params[0].short == "-s"
    assert params[0].choices == ("-s", "-3")


def test_unbracketed_value_alternation_is_not_a_flag() -> None:
    """`{{yes|no}}` is a value set for a positional, not option spellings."""
    params = _params("tool {{yes|no}}")
    assert all(p.kind is not ParamKind.OPTION for p in params)


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("-v", True),
        ("--verbose", True),
        ("/L", True),
        ("/list", True),
        ("-s|-3", False),
        ("/etc/passwd", False),
        ("-", False),
        ("--", False),
        ("path/to/file", False),
        ("", False),
    ],
)
def test_is_flag_token(token: str, expected: bool) -> None:
    assert is_flag_token(token) is expected


def test_escaped_go_template_survives_literalisation() -> None:
    """docker's `--format "\\{\\{.ID\\}\\}"` must render as `{{.ID}}`.

    The braces belong to the command, not to tldr. Unescaping them is correct
    output even though the result contains `{{`.
    """
    template = r'docker image ls --format "\{\{.ID\}\}"'
    tool = parse_page(
        f"# docker image ls\n\n> D.\n\n- Fmt:\n\n`{template}`\n",
        source="t",
        os_target="common",
    )
    assert tool is not None
    assert tool.examples[0].literal == 'docker image ls --format "{{.ID}}"'
    assert not tool.warnings


def test_escaped_json_braces_survive_literalisation() -> None:
    """From pages/common/aws-ses.md -- `\\}\\}` closes real JSON."""
    template = r'aws ses send-email --message "Body={Text={Data={{body}}\}\}"'
    tool = parse_page(
        f"# aws ses\n\n> D.\n\n- Send:\n\n`{template}`\n",
        source="t",
        os_target="common",
    )
    assert tool is not None
    assert tool.examples[0].literal.endswith('Data=body}}"')


def test_duplicate_params_across_examples_are_merged() -> None:
    page = (
        "# tool\n\n> Desc.\n\n"
        "- One:\n\n`tool {{[-v|--verbose]}}`\n\n"
        "- Two:\n\n`tool {{[-v|--verbose]}} {{path/to/file}}`\n"
    )
    tool = parse_page(page, source="t", os_target="common")
    assert tool is not None
    assert len([p for p in tool.params if p.long == "--verbose"]) == 1


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("binary", "path", "expected"),
    [
        ("git", ("commit",), "git_commit"),
        ("apt-get", (), "apt_get"),
        ("pacman", ("--sync",), "pacman_sync"),
        ("pacman", ("-S",), "pacman_S"),
        ("acme.sh", ("--dns",), "acme_sh_dns"),
        ("pm", ("install-commit",), "pm_install_commit"),
    ],
)
def test_qualified_name(binary: str, path: tuple[str, ...], expected: str) -> None:
    tool = RawTool(
        binary=binary, path=path, description="d", tier=SourceTier.TLDR, source="s"
    )
    assert tool.qualified_name == expected


@pytest.mark.parametrize(
    ("binary", "expected"),
    [
        (".", "dot"),
        (":", "colon"),
        ("|", "pipe"),
        ("[", "lbracket"),
        ("[[", "lbracket_lbracket"),
        ("((", "lparen_lparen"),
        ("?", "question"),
        ("!", "bang"),
    ],
)
def test_punctuation_only_names_get_spelled_identifiers(
    binary: str, expected: str
) -> None:
    """The corpus documents 20 such builtins; `.` used to become bare `_`."""
    tool = RawTool(binary, (), "d", SourceTier.TLDR, "s")
    assert tool.qualified_name == expected


def test_punctuation_names_stay_distinct() -> None:
    """Two different builtins must never share a catalog key."""
    names = {
        RawTool(b, (), "d", SourceTier.TLDR, "s").qualified_name
        for b in (".", ":", "[", "]", "[[", "]]", "|", "<", ">", "<>", "{", "}")
    }
    assert len(names) == 12


@given(st.text(min_size=1, max_size=20))
@settings(max_examples=400, deadline=None)
def test_qualified_name_is_always_a_usable_key(binary: str) -> None:
    tool = RawTool(binary, (), "d", SourceTier.TLDR, "s")
    name = tool.qualified_name
    assert name
    assert not name.startswith("_")
    assert not name.endswith("_")


def test_qualified_name_preserves_case() -> None:
    """`pacman -Q` and `pacman -q` are different commands."""
    upper = RawTool("pacman", ("-Q",), "d", SourceTier.TLDR, "s").qualified_name
    lower = RawTool("pacman", ("-q",), "d", SourceTier.TLDR, "s").qualified_name
    assert upper != lower


def test_raw_tool_rejects_empty_binary() -> None:
    with pytest.raises(ValueError):
        RawTool(binary="", path=(), description="d", tier=SourceTier.TLDR, source="s")


def test_raw_param_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        RawParam(name="", kind=ParamKind.OPTION)


def test_raw_example_rejects_empty_template() -> None:
    with pytest.raises(ValueError):
        RawExample(description="d", template="", literal="")


def test_raw_param_display_prefers_long() -> None:
    param = RawParam(name="message", kind=ParamKind.OPTION, short="-m", long="--message")
    assert param.display == "--message"
    assert RawParam(name="m", kind=ParamKind.OPTION, short="-m").display == "-m"


def test_warnings_do_not_affect_equality() -> None:
    """Two findings differing only in parse warnings are the same finding."""
    base = {
        "binary": "x",
        "path": (),
        "description": "d",
        "tier": SourceTier.TLDR,
        "source": "s",
    }
    assert RawTool(**base, warnings=("a",)) == RawTool(**base, warnings=())


# --------------------------------------------------------------------------
# OS -> shell mapping
# --------------------------------------------------------------------------

def test_common_maps_to_every_shell() -> None:
    assert "bash" in shells_for_os(["common"])
    assert "powershell" in shells_for_os(["common"])


def test_windows_does_not_map_to_posix_shells() -> None:
    shells = shells_for_os(["windows"])
    assert "bash" not in shells
    assert "powershell" in shells


def test_appliance_pages_map_to_no_shell(fixture_tools: dict[str, RawTool]) -> None:
    """cisco-ios commands run on a router, not on the user's machine."""
    assert shells_for_os(["cisco-ios"]) == frozenset()
    cisco = [t for t in fixture_tools.values() if "cisco-ios" in t.os_targets]
    assert cisco, "fixture should include a cisco-ios page"
    for tool in cisco:
        assert platforms_for(tool) == frozenset()


def test_unknown_os_contributes_nothing() -> None:
    """Not everything -- guessing 'all shells' surfaces unrunnable commands."""
    assert shells_for_os(["plan9", ""]) == frozenset()


def test_os_mapping_is_case_and_space_insensitive() -> None:
    assert shells_for_os([" Linux "]) == shells_for_os(["linux"])


def test_os_targets_union() -> None:
    assert shells_for_os(["windows", "linux"]) == (
        OS_TO_SHELLS["windows"] | OS_TO_SHELLS["linux"]
    )


# --------------------------------------------------------------------------
# Capability inference
# --------------------------------------------------------------------------

def test_rm_is_destructive() -> None:
    caps = infer_capabilities("rm", ())
    assert Capability.DESTRUCTIVE in caps
    assert Capability.WRITES in caps


def test_ls_is_not_destructive() -> None:
    assert Capability.DESTRUCTIVE not in infer_capabilities("ls", ())


@pytest.mark.parametrize("binary", ["rmarkdown", "formatter", "ddrescue", "killer"])
def test_destructive_hints_respect_word_boundaries(binary: str) -> None:
    """Substring matching would flag these and train users to ignore warnings."""
    assert Capability.DESTRUCTIVE not in infer_capabilities(binary, ())


def test_network_driver_needs_a_network_subcommand() -> None:
    """`git push` reaches the network; `git commit` does not."""
    assert Capability.NEEDS_NETWORK in infer_capabilities("git", ("push",))
    assert Capability.NEEDS_NETWORK not in infer_capabilities("git", ("commit",))
    assert Capability.NEEDS_NETWORK not in infer_capabilities("docker", ("ps",))


def test_dedicated_network_tool_needs_no_subcommand() -> None:
    assert Capability.NEEDS_NETWORK in infer_capabilities("curl", ())


@pytest.mark.parametrize("binary", ["rm", "rmdir", "shred", "dd", "mkfs", "kill", "del"])
def test_genuinely_destructive_tools_are_flagged(binary: str) -> None:
    assert Capability.DESTRUCTIVE in infer_capabilities(binary, ())


@pytest.mark.parametrize(
    ("binary", "path"),
    [("docker", ()), ("ps", ()), ("git", ()), ("docker", ("ps",)), ("git", ("log",))],
)
def test_a_destructive_subcommand_does_not_poison_its_parent(
    binary: str, path: tuple[str, ...]
) -> None:
    """Observed live: "list running containers" marked `docker` and `ps` as
    destructive, on a query that destroys nothing.

    A bare binary's examples span its whole surface, so `docker rm` and a `ps`
    example piped into `kill` flagged the parents. A marker shown on harmless
    commands is worse than none at all, because it teaches the user to dismiss
    the one that matters -- so DESTRUCTIVE is judged on identity and prose,
    never on examples.
    """
    caps = infer_capabilities(
        binary,
        path,
        examples=[f"{binary} rm something", f"{binary} ps | kill"],
        description="manage things",
    )
    assert Capability.DESTRUCTIVE not in caps


def test_destructive_still_fires_on_a_destructive_subcommand() -> None:
    assert Capability.DESTRUCTIVE in infer_capabilities("docker", ("rmi",))
    assert Capability.DESTRUCTIVE in infer_capabilities("git", ("rm",))


def test_destructive_can_come_from_the_description() -> None:
    caps = infer_capabilities("wipefs", (), description="erase a filesystem signature")
    assert Capability.DESTRUCTIVE in caps


def test_elevation_detected_from_examples() -> None:
    caps = infer_capabilities("apt", ("install",), examples=["sudo apt install vim"])
    assert Capability.ELEVATED in caps


def test_everything_reads() -> None:
    """So an empty set means 'not analysed', never 'inert'."""
    assert Capability.READS in infer_capabilities("anything", ())


def test_capabilities_for_uses_the_tool(fixture_tools: dict[str, RawTool]) -> None:
    assert Capability.DESTRUCTIVE in capabilities_for(fixture_tools["rm"])
    assert Capability.DESTRUCTIVE not in capabilities_for(fixture_tools["ls"])


# --------------------------------------------------------------------------
# TldrSource
# --------------------------------------------------------------------------

def test_source_satisfies_the_protocol() -> None:
    assert isinstance(TldrSource(FIXTURES), Source)
    assert TldrSource(FIXTURES).tier is SourceTier.TLDR


def test_unavailable_when_root_is_missing(tmp_path: Path) -> None:
    source = TldrSource(tmp_path / "nope")
    assert not source.available()
    assert list(source.harvest()) == []


def test_unavailable_when_root_is_none() -> None:
    source = TldrSource.__new__(TldrSource)
    source._root = None  # type: ignore[attr-defined]
    assert not source.available()
    assert list(source.harvest()) == []


def test_root_may_be_the_pages_directory_itself() -> None:
    source = TldrSource(FIXTURES / "pages")
    assert source.available()
    assert any(t.binary == "git" for t in source.harvest())


def test_pages_and_pages_en_are_not_both_harvested(tmp_path: Path) -> None:
    """A checkout can hold both, byte-identical. Unioning doubled every tool."""
    page = "# foo\n\n> Does a thing.\n\n- Do it:\n\n`foo {{path/to/file}}`\n"
    for name in ENGLISH_DIRS:
        target = tmp_path / name / "common"
        target.mkdir(parents=True)
        (target / "foo.md").write_text(page, encoding="utf-8")

    source = TldrSource(tmp_path)
    tools = list(source.harvest())
    assert len(tools) == 1, [t.source for t in tools]


def test_binaries_filter() -> None:
    tools = list(TldrSource(FIXTURES).harvest(binaries=["git"]))
    assert tools
    assert {t.binary for t in tools} == {"git"}


def test_binaries_filter_is_case_insensitive() -> None:
    assert list(TldrSource(FIXTURES).harvest(binaries=["GIT"]))


def test_empty_binaries_filter_yields_nothing() -> None:
    """An empty allow-list means 'none', which must differ from None."""
    assert list(TldrSource(FIXTURES).harvest(binaries=[])) == []


def test_oversized_page_is_skipped(tmp_path: Path) -> None:
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "ok.md").write_text("# ok\n\n> Fine.\n", encoding="utf-8")
    (pages / "huge.md").write_text(
        "# huge\n\n> " + "x" * (MAX_PAGE_BYTES + 10), encoding="utf-8"
    )
    names = {t.binary for t in TldrSource(tmp_path).harvest()}
    assert names == {"ok"}


def test_undecodable_page_is_skipped_not_fatal(tmp_path: Path) -> None:
    """One bad file is one missing tool, not a failed catalog build."""
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "ok.md").write_text("# ok\n\n> Fine.\n", encoding="utf-8")
    (pages / "bad.md").write_bytes(b"# bad\n\n> \xff\xfe invalid utf-8\n")
    names = {t.binary for t in TldrSource(tmp_path).harvest()}
    assert names == {"ok"}


def test_non_markdown_files_are_ignored(tmp_path: Path) -> None:
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "ok.md").write_text("# ok\n\n> Fine.\n", encoding="utf-8")
    (pages / "LICENSE").write_text("not a page", encoding="utf-8")
    (pages / "index.json").write_text("{}", encoding="utf-8")
    assert len(list(TldrSource(tmp_path).harvest())) == 1


# --------------------------------------------------------------------------
# Encoding and line endings
#
# The corpus is a path the USER controls -- their own tldr client cache, or a
# git clone under their own core.autocrlf setting. None of our .gitattributes
# rules apply to it, so the parser has to cope with whatever they have.
# --------------------------------------------------------------------------

_PAGE = "# foo bar\n\n> Does a thing.\n\n- Do it:\n\n`foo bar {{[-m|--message]}}`\n"


@pytest.mark.parametrize(
    ("ending", "label"),
    [("\n", "LF"), ("\r\n", "CRLF"), ("\r", "CR")],
)
def test_line_endings(ending: str, label: str) -> None:
    tool = parse_page(_PAGE.replace("\n", ending), source="s", os_target="common")
    assert tool is not None, label
    assert tool.binary == "foo"
    assert tool.path == ("bar",)
    assert tool.examples
    for example in tool.examples:
        assert "\r" not in example.template
        assert "\r" not in example.literal


def test_utf8_bom_is_stripped() -> None:
    """A BOM used to hide the `# ` title, dropping the page with no warning.

    Windows editors and PowerShell redirects add one routinely, so a user's
    corpus can carry it while ours never does.
    """
    tool = parse_page("﻿" + _PAGE, source="s", os_target="common")
    assert tool is not None
    assert tool.binary == "foo"


def test_bom_on_disk_is_stripped(tmp_path: Path) -> None:
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "foo.md").write_bytes(b"\xef\xbb\xbf" + _PAGE.encode("utf-8"))
    tools = list(TldrSource(tmp_path).harvest())
    assert [t.binary for t in tools] == ["foo"]


def test_crlf_on_disk(tmp_path: Path) -> None:
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "foo.md").write_bytes(_PAGE.replace("\n", "\r\n").encode("utf-8"))
    tools = list(TldrSource(tmp_path).harvest())
    assert [t.binary for t in tools] == ["foo"]
    assert all("\r" not in e.literal for t in tools for e in t.examples)


def test_non_ascii_title_yields_a_usable_key() -> None:
    tool = parse_page("# 你好\n\n> Desc.\n", source="s", os_target="common")
    assert tool is not None
    assert tool.qualified_name
    assert tool.qualified_name.isascii()


# --------------------------------------------------------------------------
# Filesystem hostility
# --------------------------------------------------------------------------

def test_directory_named_like_a_page_is_skipped(tmp_path: Path) -> None:
    """Reading it raises OSError; one bad entry is not a failed build."""
    pages = tmp_path / "pages" / "common"
    pages.mkdir(parents=True)
    (pages / "ok.md").write_text(_PAGE, encoding="utf-8")
    (pages / "weird.md").mkdir()
    assert [t.binary for t in TldrSource(tmp_path).harvest()] == ["foo"]


def test_stray_file_among_os_directories(tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    (pages / "common").mkdir(parents=True)
    (pages / "common" / "ok.md").write_text(_PAGE, encoding="utf-8")
    (pages / "index.json").write_text("{}", encoding="utf-8")
    assert len(list(TldrSource(tmp_path).harvest())) == 1


def test_root_that_is_a_file(tmp_path: Path) -> None:
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")
    source = TldrSource(target)
    assert not source.available()
    assert list(source.harvest()) == []


def test_default_root_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Called during discovery; a crash here takes the daemon down."""
    monkeypatch.delenv("CL_AI_TLDR_ROOT", raising=False)

    def no_home() -> Path:
        raise RuntimeError("no home directory")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    assert default_root() is None


def test_default_root_honours_the_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CL_AI_TLDR_ROOT", str(tmp_path))
    assert default_root() == tmp_path
    monkeypatch.setenv("CL_AI_TLDR_ROOT", str(tmp_path / "missing"))
    assert default_root() is None


def test_pathological_placeholder_count() -> None:
    """A page is prose, but nothing stops one line holding thousands of slots."""
    template = "x " + "{{a}} " * 2000
    tool = parse_page(
        f"# x\n\n> D.\n\n- Go:\n\n`{template}`\n", source="s", os_target="common"
    )
    assert tool is not None
    assert len(tool.examples) == 1


def test_harvest_is_lazy() -> None:
    """The corpus is thousands of pages; an indexer must not wait for all."""
    import types

    result = TldrSource(FIXTURES).harvest()
    assert isinstance(result, types.GeneratorType)
    assert next(iter(result)) is not None


def test_source_is_recorded_per_tool(fixture_tools: dict[str, RawTool]) -> None:
    tool = fixture_tools["git_commit"]
    assert tool.source == "tldr/common/git-commit.md"
    provenance = tool.provenance()
    assert provenance.tier is SourceTier.TLDR
    assert provenance.source == tool.source


def test_os_target_comes_from_the_directory(fixture_tools: dict[str, RawTool]) -> None:
    assert fixture_tools["git_commit"].os_targets == frozenset({"common"})
    assert fixture_tools["pacman_sync"].os_targets == frozenset({"linux"})
    assert fixture_tools["MOUNT"].os_targets == frozenset({"dos"})


# --------------------------------------------------------------------------
# Corpus-wide invariants
# --------------------------------------------------------------------------

def _assert_invariants(tools: list[RawTool]) -> None:
    for tool in tools:
        assert tool.binary
        assert tool.qualified_name
        assert not tool.qualified_name.startswith("_")
        for param in tool.params:
            assert param.name
            assert param.kind in ParamKind
            # Flag-shaped, allowing the DOS/cmd `/list` convention alongside
            # the POSIX dash forms.
            if param.short is not None:
                assert is_flag_token(param.short), (tool.source, param.short)
            if param.long is not None:
                assert is_flag_token(param.long), (tool.source, param.long)
            if param.kind is ParamKind.SUBCOMMAND:
                assert param.choices, (tool.source, param.name)
        for example in tool.examples:
            assert example.template
            # The invariant that matters most: no half-parsed placeholder may
            # ever reach a rendered command.
            #
            # Checked only for templates with no escaped braces. Escaping is
            # how a page writes braces that belong to the COMMAND rather than
            # to tldr, and both forms occur for real: aws-ses.md escapes
            # nested JSON as `\}\}`, and docker-image-ls.md escapes a Go
            # template as `\{\{.ID\}\}`, whose correct output genuinely
            # contains `{{`. Where nothing is escaped, a surviving brace pair
            # can only mean an unsubstituted slot.
            if "\\{" not in example.template and "\\}" not in example.template:
                assert "{{" not in example.literal, (tool.source, example.literal)
                assert "}}" not in example.literal, (tool.source, example.literal)
            # Escapes must always be resolved, never carried into a command.
            assert "\\{" not in example.literal, (tool.source, example.literal)
            assert "\\}" not in example.literal, (tool.source, example.literal)


def test_fixture_corpus_invariants(fixture_tools: dict[str, RawTool]) -> None:
    _assert_invariants(list(fixture_tools.values()))


@pytest.mark.skipif(
    not os.environ.get("CL_AI_TLDR_ROOT"),
    reason="set CL_AI_TLDR_ROOT to run the full-corpus check",
)
def test_full_corpus_invariants() -> None:
    root = Path(os.environ["CL_AI_TLDR_ROOT"])
    source = TldrSource(root)
    assert source.available(), f"CL_AI_TLDR_ROOT={root} is not a tldr corpus"
    tools = list(source.harvest())
    assert len(tools) > 1000, f"only {len(tools)} tools; corpus looks truncated"
    _assert_invariants(tools)

    # Parse failures are expected but must stay rare; a jump means the grammar
    # changed upstream and this extractor needs revisiting.
    warned = sum(len(t.warnings) for t in tools)
    assert warned < len(tools) * 0.01, f"{warned} parse warnings across {len(tools)}"
