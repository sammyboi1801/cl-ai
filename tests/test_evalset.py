"""The scorer itself.

Unusually important. Every conclusion drawn from the evaluation set is only
as good as this arithmetic, and a scorer that quietly counts the wrong
denominator produces confident numbers that mean nothing -- which is worse
than having no numbers, because you act on them.

So the metrics are tested against hand-computed values, and the cases below
are deliberately small enough to verify by eye.
"""

from __future__ import annotations

import pytest

from tests.evalset import Case, Outcome, load_cases, score


def _case(**kwargs: object) -> Case:
    base: dict[str, object] = {"query": "q", "tools": frozenset({"right"})}
    base.update(kwargs)
    return Case(**base)  # type: ignore[arg-type]


def _outcome(case: Case, *names: str, commands: tuple[str, ...] = (), **kw) -> Outcome:
    return Outcome(
        case=case,
        names=names,
        commands=commands or tuple(f"{n} --flag" for n in names),
        **kw,
    )


# -- loading is strict ----------------------------------------------------


def test_a_minimal_dict_loads() -> None:
    (case,) = load_cases([{"query": "list files", "tools": ["ls"]}])
    assert case.query == "list files"
    assert case.tools == frozenset({"ls"})
    assert case.style == "natural"


def test_an_unknown_key_is_rejected() -> None:
    """A typo in a key name would silently turn an assertion into a default.

    These cases are ground truth; a mis-parsed one makes the measurement a
    guess that still prints three decimal places.
    """
    with pytest.raises(ValueError, match="unknown keys"):
        load_cases([{"query": "x", "tools": ["ls"], "expects": ["-l"]}])


def test_an_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty query"):
        load_cases([{"query": "   ", "tools": ["ls"]}])


def test_a_bare_string_is_accepted_where_a_list_is_meant() -> None:
    """Case authors write `"expect": "-l"` by hand. Treating a string as an
    iterable of characters would silently require five substrings."""
    (case,) = load_cases([{"query": "x", "tools": "ls", "expect": "-l"}])
    assert case.tools == frozenset({"ls"})
    assert case.expect == ("-l",)


def test_the_error_names_the_offending_case() -> None:
    with pytest.raises(ValueError, match="case 2"):
        load_cases(
            [
                {"query": "a", "tools": ["ls"]},
                {"query": "b", "tools": ["ls"]},
                {"query": "c", "nonsense": 1},
            ]
        )


# -- one outcome at a time ------------------------------------------------


def test_rank_finds_the_first_acceptable_tool() -> None:
    case = _case(tools=frozenset({"right", "alsoright"}))
    assert _outcome(case, "wrong", "alsoright", "right").rank == 1


def test_rank_is_none_when_nothing_acceptable_came_back() -> None:
    assert _outcome(_case(), "wrong", "other").rank is None


def test_an_empty_expectation_is_satisfied_by_an_empty_answer() -> None:
    """A no-match case is a PASS when nothing is returned. This is the whole
    point of the score floor and has to be scored as success, not absence."""
    case = _case(tools=frozenset())
    assert case.expects_nothing
    assert _outcome(case).tool_ok is True


def test_an_empty_expectation_is_failed_by_any_answer() -> None:
    assert _outcome(_case(tools=frozenset()), "something").tool_ok is False


def test_tool_ok_requires_the_top_slot_not_merely_presence() -> None:
    assert _outcome(_case(), "wrong", "right").tool_ok is False
    assert _outcome(_case(), "right", "wrong").tool_ok is True


def test_a_forbidden_tool_in_the_top_slot_is_reported() -> None:
    case = _case(tools=frozenset({"findstr"}), forbid_tools=frozenset({"grep"}))
    assert _outcome(case, "grep", "findstr").forbidden_hit == "grep"
    assert _outcome(case, "findstr").forbidden_hit is None


def test_a_forbidden_tool_lower_down_is_not_a_hit() -> None:
    """Only the top slot is what lands in the buffer."""
    case = _case(tools=frozenset({"findstr"}), forbid_tools=frozenset({"grep"}))
    assert _outcome(case, "findstr", "grep").forbidden_hit is None


# -- example selection is scored separately -------------------------------


def test_example_is_unjudged_when_no_expectations_were_written() -> None:
    """None, not True. Counting unwritten expectations as passes would
    inflate the example score with every case that does not test it."""
    assert _outcome(_case(), "right").example_ok is None


def test_example_is_unjudged_when_the_tool_was_wrong() -> None:
    """Otherwise one miss is counted twice, as a tool failure and an example
    failure, and the two numbers stop being independent."""
    case = _case(expect=("--soft",))
    assert _outcome(case, "wrong").example_ok is None


def test_example_passes_when_the_expected_substring_is_present() -> None:
    case = _case(expect=("--soft",))
    out = _outcome(case, "right", commands=("git reset --soft HEAD~1",))
    assert out.example_ok is True


def test_example_fails_when_the_expected_substring_is_missing() -> None:
    case = _case(expect=("--soft",))
    out = _outcome(case, "right", commands=("git reset --hard",))
    assert out.example_ok is False


def test_example_fails_on_a_forbidden_substring() -> None:
    """The ipconfig /renew failure: right tool, actively wrong example."""
    case = _case(forbid=("/renew",))
    out = _outcome(case, "right", commands=("ipconfig /renew adapter",))
    assert out.example_ok is False


def test_every_expected_substring_must_be_present() -> None:
    case = _case(expect=("--all", "--message"))
    assert _outcome(
        case, "right", commands=('git commit --message "m"',)
    ).example_ok is False
    assert _outcome(
        case, "right", commands=('git commit --all --message "m"',)
    ).example_ok is True


def test_the_example_is_read_from_the_matching_rank_not_the_top() -> None:
    """When the acceptable tool is second, its command is the one to judge."""
    case = _case(expect=("--soft",))
    out = _outcome(
        case, "wrong", "right", commands=("nonsense", "git reset --soft HEAD~1")
    )
    assert out.example_ok is True


# -- destructive marking --------------------------------------------------


def test_danger_is_unjudged_for_a_non_destructive_case() -> None:
    assert _outcome(_case(), "right").danger_ok is None


def test_danger_passes_when_the_answer_was_marked() -> None:
    case = _case(dangerous=True)
    out = _outcome(case, "right", dangerous_flags=(True,))
    assert out.danger_ok is True


def test_danger_fails_when_a_destructive_answer_was_not_marked() -> None:
    """The direction that costs a user their files."""
    case = _case(dangerous=True)
    assert _outcome(case, "right", dangerous_flags=(False,)).danger_ok is False


# -- aggregation ----------------------------------------------------------


def test_no_match_cases_are_excluded_from_the_tool_metrics() -> None:
    """They are scored by their own metric. Counting a correct empty answer
    as a tool@1 hit would let a system that returns nothing at all look like
    it had solved retrieval."""
    report = score(
        [
            _outcome(_case(), "right"),
            _outcome(_case(tools=frozenset())),
            _outcome(_case(tools=frozenset())),
        ]
    )
    assert report.tool_at(1) == 1.0, "denominator must be the 1 real case"
    rate, count = report.no_match_accuracy()
    assert (rate, count) == (1.0, 2)


def test_tool_at_k_counts_by_position() -> None:
    report = score(
        [
            _outcome(_case(), "right"),
            _outcome(_case(), "a", "right"),
            _outcome(_case(), "a", "b", "c", "right"),
            _outcome(_case(), "a", "b", "c", "d"),
        ]
    )
    assert report.tool_at(1) == 0.25
    assert report.tool_at(3) == 0.5
    assert report.tool_at(5) == 0.75


def test_mrr_is_the_mean_reciprocal_rank() -> None:
    report = score(
        [
            _outcome(_case(), "right"),        # 1/1
            _outcome(_case(), "a", "right"),   # 1/2
            _outcome(_case(), "a", "b"),       # 0
        ]
    )
    assert report.mrr() == pytest.approx((1.0 + 0.5) / 3)


def test_metrics_are_zero_rather_than_dividing_by_zero() -> None:
    empty = score([])
    assert empty.tool_at(1) == 0.0
    assert empty.mrr() == 0.0
    assert empty.example_accuracy() == (0.0, 0)
    assert empty.no_match_accuracy() == (0.0, 0)
    assert empty.danger_recall() == (0.0, 0)
    assert empty.forbidden_rate() == (0.0, 0)
    assert empty.latency() == (0.0, 0.0)


def test_example_accuracy_counts_only_judgeable_cases() -> None:
    report = score(
        [
            _outcome(_case(expect=("-l",)), "right", commands=("ls -l",)),
            _outcome(_case(expect=("-l",)), "right", commands=("ls -a",)),
            _outcome(_case(), "right"),  # unjudgeable, must not count
        ]
    )
    assert report.example_accuracy() == (0.5, 2)


def test_forbidden_rate_counts_only_cases_that_declared_one() -> None:
    forbidding = _case(forbid_tools=frozenset({"grep"}))
    report = score(
        [
            _outcome(forbidding, "grep"),
            _outcome(forbidding, "right"),
            _outcome(_case(), "right"),
        ]
    )
    assert report.forbidden_rate() == (0.5, 2)


def test_grouping_reports_a_rate_and_a_count_per_bucket() -> None:
    report = score(
        [
            _outcome(_case(category="git"), "right"),
            _outcome(_case(category="git"), "wrong"),
            _outcome(_case(category="net"), "right"),
        ]
    )
    assert report.by("category") == {"git": (0.5, 2), "net": (1.0, 1)}


def test_failures_include_every_kind_of_wrong() -> None:
    good = _outcome(_case(), "right")
    miss = _outcome(_case(), "wrong")
    bad_example = _outcome(
        _case(expect=("-l",)), "right", commands=("ls -a",)
    )
    neighbour = _outcome(
        _case(tools=frozenset({"findstr"}), forbid_tools=frozenset({"grep"})),
        "grep",
    )
    report = score([good, miss, bad_example, neighbour])
    assert good not in report.failures()
    assert {id(o) for o in report.failures()} == {
        id(miss), id(bad_example), id(neighbour)
    }


def test_latency_reports_median_and_worst() -> None:
    report = score(
        [_outcome(_case(), "right", elapsed_ms=ms) for ms in (5.0, 1.0, 100.0)]
    )
    assert report.latency() == (5.0, 100.0)
