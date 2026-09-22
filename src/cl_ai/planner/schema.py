"""Turn a catalog `Tool` into the schema Needle actually reads.

THE CONTRACT, FROM CACTUS'S OWN GUIDANCE
Needle is a SPAN EXTRACTOR, not a generator: "a call contains only values
evidenced by the request; the model derives each argument from a span of the
text". An optional field with no span is omitted rather than invented, and a
required field with no span suppresses the whole call.

Everything below follows from that. The per-argument description is not prose
for a human -- it is the instruction telling the model what span to look for,
and Cactus name the effective forms explicitly: "City, ST", "e.g. T-1042",
"ISO date", "the place after 'from'". They name the ineffective form too:
vague category language, or instructions addressed to the model.

WHAT WE WERE ABOUT TO SEND, AND WHY IT WOULD HAVE FAILED
Measured over the corpus, every one of 34,840 extracted parameters carried
the EXAMPLE's description verbatim:

    git_commit.message  ->  "Commit staged files to the repository with ..."
    tar.directory       ->  "[c]reate a g[z]ipped (compressed) archive ..."

That is the documented anti-pattern exactly: it describes the task, not the
span, and every property on a tool says roughly the same sentence as the
tool itself. A model told that `message` means "commit staged files" has
been given no way to find the message.

WHERE THE GOOD DESCRIPTION COMES FROM
The example COMMANDS still hold the concrete value the page author chose,
because the extractor substitutes placeholders rather than dropping them:

    git commit --file path/to/commit_message_file
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^

So the format hint is recoverable by reading the value that follows the flag
in a real example -- which yields precisely the "e.g. ..." form Cactus
recommend, grounded in the page rather than invented here.

Booleans get different treatment. A flag has no value to point at, so what
the model needs is when to set it, and there the example's own description
is the right text -- cleaned of tldr's `[c]reate` mnemonic markup.

WHAT IS DELIBERATELY LEFT OUT
  * SUBCOMMAND params. They are part of the invocation, not its arguments;
    `git commit` is the tool, not `git` with subcommand="commit".
  * `required`. Nothing is marked required, because a required field with no
    span suppresses the entire call, and we would rather receive a partial
    set of arguments and fall back to the example's placeholder for the rest
    than receive nothing. This is the weakest-backend rule again.
"""

from __future__ import annotations

import re
from typing import Any

from cl_ai.ir import Param, ParamKind, Tool

__all__ = ["MAX_DECLARED_TOOLS", "schema_for", "schemas_for"]

#: Hard ceiling, measured against needle3.cact. Not a style choice.
#:
#: Cactus's two documentation pages disagree about this. The Python docs say
#: that above five tools "every schema is embedded once at init by a built-in
#: contrastive head" and only the top five enter context. The porting guide
#: says of this exact checkpoint: "this release does not ship one, so on
#: needle3.cact every declared tool goes into the prefix and retrieve_tools
#: is absent".
#:
#: Tested, and the porting guide is right. `retrieve_tools` does not exist on
#: the agent, and declaring more tools simply makes the prefix bigger:
#:
#:      tools     init      answer
#:          1    0.44s      correct
#:          5    1.36s      correct
#:         20    6.90s      no call at all
#:         50   22.51s      no call at all
#:        150       --      needle_init failed (code -1)
#:
#: So there is no tool retrieval to delegate to, the cost is linear in the
#: catalogue, and quality collapses past five. Retrieval stays ours.
MAX_DECLARED_TOOLS = 5

#: tldr marks mnemonic letters as `[c]reate`, `g[z]ipped`. Useful to a human
#: reading the page and noise to a model reading a schema.
_MNEMONIC = re.compile(r"\[([a-zA-Z])\]")

#: Tokens in an example command that are flags rather than values.
_FLAGLIKE = re.compile(r"^-{1,2}[A-Za-z0-9]")

#: A property name JSON Schema and every consumer can handle.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: A flag that is only digits -- `-9`, `-1`, `-100`. These are switches:
#: `kill -9 pid` sends signal 9, it does not pass a value to `-9`. The
#: extractor types them as options WITH values, which produced 1,213
#: properties literally named "1", "9" and "100" across the corpus.
_NUMERIC_FLAG = re.compile(r"^-{1,2}\d+$")

_MAX_DESCRIPTION = 120


def _clean(text: str) -> str:
    """Strip tldr markup and collapse whitespace."""
    text = _MNEMONIC.sub(r"\1", text)
    text = text.replace("`", "")
    text = " ".join(text.split())
    if len(text) > _MAX_DESCRIPTION:
        text = text[:_MAX_DESCRIPTION].rsplit(" ", 1)[0] + "..."
    return text


def _tokens(command: str) -> list[str]:
    """Split a command line well enough to find flag/value pairs.

    Not a shell parser and does not need to be: it is reading examples the
    extractor already produced, to recover the literal a placeholder was
    substituted with.
    """
    return [t for t in command.replace("=", " ").split() if t]


def _value_after_flag(tool: Tool, flag: str) -> str | None:
    """The concrete value an example passes to `flag`."""
    for example in tool.examples:
        tokens = _tokens(example.command)
        for index, token in enumerate(tokens[:-1]):
            if token == flag:
                candidate = tokens[index + 1]
                if not _FLAGLIKE.match(candidate):
                    return candidate.strip("\"'")
    return None


def _value_for_positional(tool: Tool, name: str) -> str | None:
    """A literal from an example that plausibly fills this positional.

    Matched on the slug: `path_to_directory` was substituted from a token
    containing `directory`, so the token is findable.

    Two exclusions, both learned from the output. Tokens belonging to the
    invocation itself are skipped -- matching `source_tar` against the bare
    word `tar` produced "source tar, e.g. tar", pointing the model at the
    binary. And a token that merely repeats the slug is no format hint at
    all, so the value must carry some structure (a path, an extension, a
    wildcard) or be meaningfully longer than the word it matched.
    """
    invocation = {tool.binary.lower(), *(p.lower() for p in tool.path)}
    stem = name.rsplit("_", 1)[-1].lower()
    if not stem:
        return None
    for example in tool.examples:
        for token in _tokens(example.command):
            if _FLAGLIKE.match(token):
                continue
            bare = token.strip("\"'")
            lowered = bare.lower()
            if lowered in invocation or stem not in lowered:
                continue
            if any(ch in bare for ch in "/.*[") or len(lowered) > len(stem) + 2:
                return bare
    return None


def _property_name(param: Param) -> str:
    """A legal identifier for this parameter.

    A name beginning with a digit is not addressable in most consumers and
    reads as nonsense to a model: `kill` declared properties called "1", "2"
    and "9". Prefixed rather than dropped, because the flag is real and a
    user may well ask for it.
    """
    name = param.name
    if _IDENTIFIER.match(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    return f"opt_{cleaned}" if cleaned else "opt"


def _is_switch(param: Param) -> bool:
    """Whether this takes no value, despite being extracted as if it does."""
    return bool(param.flag and _NUMERIC_FLAG.match(param.flag))


def _describe(tool: Tool, param: Param) -> str:
    """The per-argument description: what span to look for.

    See the module docstring. This is the single most important field in the
    schema and the one our extraction got wrong.
    """
    if param.type == "boolean" or _is_switch(param):
        # No value to point at, so the model needs to know WHEN to set it.
        # The example's own description is the right text here.
        return _clean(param.description) or f"set to use {param.flag or param.name}"

    if param.enum:
        return "one of: " + ", ".join(param.enum)

    sample = (
        _value_after_flag(tool, param.flag)
        if param.flag
        else _value_for_positional(tool, param.name)
    )
    readable = param.name.replace("_", " ")
    # "message, e.g. message" tells the model nothing: tldr substituted the
    # placeholder with its own name, so the sample is the slug again. Only
    # keep a sample that adds information.
    if sample and sample.lower().strip("_-") not in {
        param.name.lower(), readable.lower(), param.name.replace("_", "").lower()
    }:
        return f"{readable}, e.g. {sample}"
    if param.flag:
        # Name the flag the span belongs to. Cactus's effective form is
        # positional -- "the place after 'from'" -- and for a CLI the
        # equivalent landmark is the flag itself.
        return f"{readable}, the value for {param.flag}"
    return readable


def _properties(tool: Tool) -> tuple[dict[str, Any], list[str]]:
    properties: dict[str, Any] = {}
    order: list[str] = []
    for param in tool.params:
        # Subcommands are identity, not arguments.
        if param.kind is ParamKind.SUBCOMMAND:
            continue
        # A tool can carry two params with one name -- `git_commit` has both
        # a `--file` option and a `file` positional. In a properties dict the
        # second would silently clobber the first, so the option wins (it is
        # unambiguous to render) and the positional is renamed.
        name = _property_name(param)
        if name in properties:
            if param.kind is ParamKind.OPTION:
                properties.pop(name)
                order.remove(name)
            else:
                # An array positional colliding with a scalar option is the
                # common shape -- `git commit --file MSG` versus
                # `git commit FILE...` -- and the plural reads as what it is.
                name = f"{name}s" if param.type == "array" else f"{name}_arg"
                if name in properties:
                    continue
        declared = "boolean" if _is_switch(param) else param.type
        schema: dict[str, Any] = {"type": declared}
        if declared == "array":
            schema["items"] = {"type": "string"}
        description = _describe(tool, param)
        if description:
            schema["description"] = description
        if param.enum:
            schema["enum"] = list(param.enum)
        properties[name] = schema
        order.append(name)
    return properties, order


def schema_for(tool: Tool) -> dict[str, Any]:
    """One Needle function schema for one tool.

    The tool NAME is the invocation with spaces replaced -- `git_commit`,
    not `git`. Cactus's "one tool per action" rule is why our leaves are
    fine-grained in the first place: a bare `git` carrying eight unrelated
    examples is exactly the `control_home(device, action, value)` catch-all
    they warn against, because it pushes the decision into free text.
    """
    properties, _ = _properties(tool)
    description = _clean(tool.description) or tool.invocation
    return {
        "name": tool.name,
        "description": description,
        "parameters": {"type": "object", "properties": properties},
    }


def schemas_for(tools: object) -> list[dict[str, Any]]:
    """Schemas for a retrieved candidate set, capped at what fits in context.

    The cap is not an optimisation. This checkpoint has no tool retrieval --
    every declared schema goes into the prefix -- so the sixth tool costs
    context and buys nothing, and the twentieth stops it answering at all.
    See MAX_DECLARED_TOOLS for the measurements.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for tool in tools:  # type: ignore[attr-defined]
        if tool.name in seen:
            continue
        seen.add(tool.name)
        out.append(schema_for(tool))
        if len(out) >= MAX_DECLARED_TOOLS:
            break
    return out
