"""Tests for the IR -- the contract every other component depends on.

This file exists because ir.py had no tests at all while being the most
load-bearing module in the codebase. Its invariants are not decoration: a
mutable "frozen" dataclass or an unhashable Step fails at runtime in the
assembler, far from the cause.
"""

from __future__ import annotations

import dataclasses

import pytest

from cl_ai.ir import (
    Capabilities,
    Capability,
    ContextFacts,
    Join,
    Param,
    ParamKind,
    Plan,
    PlanRequest,
    Provenance,
    SourceTier,
    Step,
    StreamKind,
    Tool,
)

# --------------------------------------------------------------------------
# Param
# --------------------------------------------------------------------------

def test_param_requires_a_name() -> None:
    with pytest.raises(ValueError):
        Param(name="", type="string")


def test_option_must_carry_a_spelling() -> None:
    """An option with no flag cannot be rendered at all.

    Failing here, at catalog build time, means there is a source path to blame.
    Failing in the renderer means a silently missing argument.
    """
    with pytest.raises(ValueError, match="neither flag nor short"):
        Param(name="message", type="string", kind=ParamKind.OPTION)


def test_option_accepts_short_only() -> None:
    param = Param(name="m", type="string", kind=ParamKind.OPTION, short="-m")
    assert param.short == "-m"
    assert param.flag is None


def test_positional_needs_no_flag() -> None:
    param = Param(name="file", type="string", kind=ParamKind.POSITIONAL)
    assert param.flag is None


def test_param_default_kind_is_positional() -> None:
    assert Param(name="x", type="string").kind is ParamKind.POSITIONAL


def test_flag_is_stored_not_derived() -> None:
    """`--gpg-sign` slugs to `gpg_sign`; the hyphen cannot be recovered."""
    param = Param(
        name="gpg_sign", type="string", kind=ParamKind.OPTION, flag="--gpg-sign"
    )
    assert param.flag == "--gpg-sign"
    assert param.flag != f"--{param.name}"


def test_param_is_frozen() -> None:
    param = Param(name="x", type="string")
    with pytest.raises(dataclasses.FrozenInstanceError):
        param.name = "y"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("tier", "trusted"),
    [
        (SourceTier.NATIVE, True),
        (SourceTier.MACHINE_HELP, True),
        (SourceTier.COMPLETIONS, True),
        (SourceTier.MANDOCS, False),
        (SourceTier.TLDR, False),
        (SourceTier.HELPTEXT, False),
    ],
)
def test_enum_trust_follows_the_tier_ladder(tier: SourceTier, trusted: bool) -> None:
    """Completion scripts state valid values; prose only shows some of them.

    Treating a prose-derived enum as complete would make a planner reject
    arguments that are perfectly valid.
    """
    provenance = Provenance(tier=tier, source="s")
    assert provenance.enum_untrustworthy(("a", "b")) is (not trusted)


def test_empty_enum_is_never_untrustworthy() -> None:
    provenance = Provenance(tier=SourceTier.HELPTEXT, source="s")
    assert provenance.enum_untrustworthy(()) is False


def test_tier_ordering_is_meaningful() -> None:
    """Lower is more trustworthy; the merge depends on this."""
    assert SourceTier.NATIVE < SourceTier.TLDR < SourceTier.HANDWRITTEN


# --------------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------------

def test_schematized_distinguishes_unknown_from_empty() -> None:
    """The point of this flag: "exists but I have no schema" is a real state.

    It is what lets the system say so instead of substituting a neighbouring
    tool -- the failure that turns a gap into a confidently wrong command.
    """
    bare = Tool(name="git", description="d", binary="git")
    assert not bare.schematized
    with_examples = Tool(name="git", description="d", binary="git", examples=("git",))
    assert with_examples.schematized
    with_params = Tool(
        name="git",
        description="d",
        binary="git",
        params=(Param(name="x", type="string"),),
    )
    assert with_params.schematized


def test_tool_is_hashable() -> None:
    """Tools land in sets during retrieval deduplication."""
    tool = Tool(name="git", description="d", binary="git")
    assert len({tool, tool}) == 1


# --------------------------------------------------------------------------
# Step and Plan
# --------------------------------------------------------------------------

def test_step_arguments_are_read_only() -> None:
    """frozen=True protects the binding, not the dict behind it."""
    step = Step(tool="git_commit", arguments={"message": "hi"})
    with pytest.raises(TypeError):
        step.arguments["message"] = "bye"  # type: ignore[index]


def test_step_arguments_are_copied_from_the_caller() -> None:
    """Mutating the caller's dict afterwards must not alter the Step."""
    source = {"message": "hi"}
    step = Step(tool="git_commit", arguments=source)
    source["message"] = "changed"
    assert step.arguments["message"] == "hi"


def test_step_is_hashable_with_arguments() -> None:
    """Needed to deduplicate candidates; a dict field would break this."""
    step = Step(tool="git_commit", arguments={"message": "hi"})
    assert len({step, step}) == 1


def test_equal_steps_hash_equal() -> None:
    a = Step(tool="t", arguments={"k": "v"})
    b = Step(tool="t", arguments={"k": "v"})
    assert a == b
    assert hash(a) == hash(b)


def test_steps_differing_only_in_arguments_are_not_equal() -> None:
    a = Step(tool="t", arguments={"k": "v"})
    b = Step(tool="t", arguments={"k": "other"})
    assert a != b


def test_plan_ignores_raw_for_equality() -> None:
    """Two plans differing only in provider noise are one suggestion."""
    step = Step(tool="t")
    assert Plan(steps=(step,), raw={"a": 1}) == Plan(steps=(step,), raw={"b": 2})


def test_plan_is_empty() -> None:
    assert Plan().is_empty
    assert Plan(refused=True, steps=(Step(tool="t"),)).is_empty
    assert not Plan(steps=(Step(tool="t"),)).is_empty


def test_join_and_stream_kinds_exist() -> None:
    """Pipelines need both a join and a stream type to render faithfully."""
    assert Join.PIPE.value == "pipe"
    assert StreamKind.OBJECTS.value == "objects"
    step = Step(tool="ls", emits=StreamKind.PATHS, join_to_next=Join.PIPE)
    assert step.join_to_next is Join.PIPE


# --------------------------------------------------------------------------
# Requests and capabilities
# --------------------------------------------------------------------------

def test_context_facts_defaults_are_inert() -> None:
    facts = ContextFacts()
    assert facts.installed == frozenset()
    assert facts.recent_commands == ()


def test_plan_request_defaults() -> None:
    request = PlanRequest(query="list files", tools=())
    assert request.max_steps == 3
    assert isinstance(request.context, ContextFacts)


def test_backend_capabilities_default_to_the_weakest_backend() -> None:
    """Core programs against these defaults; upgrades are opportunistic."""
    caps = Capabilities()
    assert not caps.grammar_constrained
    assert not caps.multi_call
    assert not caps.confidence


def test_capability_values_are_stable_strings() -> None:
    """They are serialised into the catalog, so the wire values are a contract."""
    assert Capability.DESTRUCTIVE.value == "destructive"
    assert Capability.NEEDS_NETWORK.value == "needs_network"
