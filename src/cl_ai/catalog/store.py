"""Artifact IO + cache keys.

Cache key is binary IDENTITY (path + size + mtime + content hash), never
`--version` output -- version detection would require execution, and identity
also catches a rebuilt binary at an unchanged version number.

Must also distinguish variants: macOS BSD `ls` and GNU `ls` are the same name
with different schemas.

WHY THE CATALOG IS PERSISTED AT ALL
-----------------------------------
Building it costs seconds; the daemon answers a Tab keypress in milliseconds.
Those are different budgets, so the build is an artifact and the daemon loads
it. That makes the on-disk format part of the contract, which is why it is
versioned and why loading is total: a corrupt or stale file must degrade to
"no catalog" rather than raise inside a keystroke handler.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cl_ai.ir import (
    Capability,
    Param,
    ParamKind,
    Provenance,
    SourceTier,
    Tool,
)

from .discovery import Discovered
from .normalize import Catalog, Family, Variant

__all__ = [
    "SCHEMA_VERSION",
    "CatalogFile",
    "cache_key",
    "default_path",
    "dump",
    "identity_of",
    "load",
    "save",
]

#: Bumped whenever the serialised shape changes. A reader that sees a version
#: it does not know refuses the file instead of guessing at the fields, because
#: a half-understood catalog produces wrong commands rather than no commands.
SCHEMA_VERSION = 1

_HASH_BYTES = 1 << 16


# --------------------------------------------------------------------------
# Identity and cache keys
# --------------------------------------------------------------------------

def identity_of(path: str | os.PathLike[str], *, read_bytes: int = _HASH_BYTES) -> str:
    """A cheap, execution-free fingerprint of one binary.

    Size and mtime alone are not enough -- a rebuild can preserve both -- and
    hashing a whole binary is too slow across a thousand PATH entries. Hashing
    a bounded prefix alongside the size catches the realistic cases: a
    different build, a different architecture, a replaced shim.

    Never runs the binary. Asking `--version` would mean executing arbitrary
    programs found on PATH during a routine cache check, which is exactly the
    thing this project refuses to do implicitly.
    """
    digest = hashlib.sha256()
    try:
        stat = os.stat(path)
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read(read_bytes))
    except OSError:
        # Unreadable is itself a stable fact about this path: report it as an
        # identity rather than raising, so one odd PATH entry cannot fail a
        # whole cache validation.
        return "unreadable"
    return digest.hexdigest()[:32]


def cache_key(entries: Iterable[Discovered], *, extra: str = "") -> str:
    """A key over everything that should invalidate a built catalog.

    Deliberately includes the SCHEMA_VERSION and the tier set, not just the
    binaries: a catalog can go stale because the code that built it changed,
    not only because the machine did.
    """
    digest = hashlib.sha256()
    digest.update(f"schema={SCHEMA_VERSION}\n".encode())
    if extra:
        digest.update(f"extra={extra}\n".encode())
    # Sorted, so two runs over the same machine agree. PATH order is already
    # captured by the entries themselves.
    for entry in sorted(entries, key=lambda e: (e.name, e.path or "")):
        digest.update(
            f"{entry.name}\0{entry.path or ''}\0{entry.size}\0{entry.mtime_ns}\0"
            f"{entry.kind.value}\n".encode()
        )
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def _provenance_to_json(provenance: Provenance | None) -> dict[str, Any] | None:
    if provenance is None:
        return None
    return {
        "tier": int(provenance.tier),
        "source": provenance.source,
        "confidence": provenance.confidence,
        "conflicts": list(provenance.conflicts),
    }


def _provenance_from_json(data: Any) -> Provenance | None:
    if not isinstance(data, dict):
        return None
    try:
        tier = SourceTier(int(data["tier"]))
    except (KeyError, TypeError, ValueError):
        return None
    conflicts = data.get("conflicts") or []
    return Provenance(
        tier=tier,
        source=str(data.get("source", "")),
        confidence=float(data.get("confidence", 1.0)),
        conflicts=tuple(str(c) for c in conflicts),
    )


def _param_to_json(param: Param) -> dict[str, Any]:
    return {
        "name": param.name,
        "type": param.type,
        "kind": param.kind.value,
        "flag": param.flag,
        "description": param.description,
        "enum": list(param.enum),
        "short": param.short,
        "required": param.required,
        "repeatable": param.repeatable,
        "provenance": _provenance_to_json(param.provenance),
    }


def _param_from_json(data: Mapping[str, Any]) -> Param:
    return Param(
        name=str(data["name"]),
        type=str(data.get("type", "string")),
        kind=ParamKind(data.get("kind", ParamKind.POSITIONAL.value)),
        flag=data.get("flag"),
        description=str(data.get("description", "")),
        enum=tuple(str(v) for v in data.get("enum") or ()),
        short=data.get("short"),
        required=bool(data.get("required", False)),
        repeatable=bool(data.get("repeatable", False)),
        provenance=_provenance_from_json(data.get("provenance")),
    )


def _tool_to_json(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "binary": tool.binary,
        "path": list(tool.path),
        "params": [_param_to_json(p) for p in tool.params],
        # Sorted so the file is byte-stable: sets iterate in arbitrary order,
        # and an artifact that differs between identical builds cannot be
        # diffed, cached by content, or trusted when it changes.
        "capabilities": sorted(c.value for c in tool.capabilities),
        "platforms": sorted(tool.platforms),
        "examples": list(tool.examples),
        "homepage": tool.homepage,
        "provenance": _provenance_to_json(tool.provenance),
    }


def _tool_from_json(data: Mapping[str, Any]) -> Tool:
    return Tool(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        binary=str(data.get("binary", "")),
        path=tuple(str(p) for p in data.get("path") or ()),
        params=tuple(_param_from_json(p) for p in data.get("params") or ()),
        capabilities=frozenset(
            Capability(c) for c in data.get("capabilities") or ()
        ),
        platforms=frozenset(str(p) for p in data.get("platforms") or ()),
        examples=tuple(str(e) for e in data.get("examples") or ()),
        homepage=data.get("homepage"),
        provenance=_provenance_from_json(data.get("provenance")),
    )


@dataclass(frozen=True)
class CatalogFile:
    """A catalog plus the metadata needed to decide whether it is still valid."""

    catalog: Catalog
    key: str = ""
    schema_version: int = SCHEMA_VERSION
    built_with: tuple[str, ...] = ()

    def is_valid_for(self, key: str) -> bool:
        return self.schema_version == SCHEMA_VERSION and bool(key) and self.key == key


def dump(
    catalog: Catalog, *, key: str = "", built_with: Iterable[str] = ()
) -> dict[str, Any]:
    """Serialise to a plain JSON-compatible structure."""
    return {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "built_with": sorted(built_with),
        "variants": [
            {"family": v.family.value, "tool": _tool_to_json(v.tool)}
            for v in catalog.variants
        ],
    }


def _loads(data: Any) -> CatalogFile | None:
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        # A version we do not know is refused rather than partially read. The
        # cost is one rebuild; the cost of guessing is wrong commands.
        return None
    variants: list[Variant] = []
    for item in data.get("variants") or ():
        if not isinstance(item, dict):
            continue
        try:
            family = Family(item.get("family", Family.UNKNOWN.value))
        except ValueError:
            family = Family.UNKNOWN
        tool_data = item.get("tool")
        if not isinstance(tool_data, dict):
            continue
        try:
            variants.append(Variant(family=family, tool=_tool_from_json(tool_data)))
        except (KeyError, TypeError, ValueError):
            # Skip the one bad entry. A single malformed tool must not cost the
            # user every other tool in the file.
            continue
    return CatalogFile(
        catalog=Catalog(variants=tuple(variants)),
        key=str(data.get("key", "")),
        schema_version=SCHEMA_VERSION,
        built_with=tuple(str(b) for b in data.get("built_with") or ()),
    )


def default_path() -> Path:
    """Where the built catalog lives.

    Honours the platform's cache convention rather than dropping a dotfile in
    the user's home directory: on Windows that is LOCALAPPDATA, elsewhere
    XDG_CACHE_HOME. A cache is data the user may delete freely, and putting it
    where their OS expects is what makes that true.
    """
    override = os.environ.get("CL_AI_CATALOG")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
        return Path(base) / "cl-ai" / "catalog.json"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "cl-ai" / "catalog.json"
    try:
        home = Path.home()
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "cl-ai" / "catalog.json"
    return home / ".cache" / "cl-ai" / "catalog.json"


def save(
    catalog: Catalog,
    path: str | os.PathLike[str] | None = None,
    *,
    key: str = "",
    built_with: Iterable[str] = (),
) -> Path:
    """Write the catalog atomically. Returns the path written."""
    target = Path(path) if path is not None else default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dump(catalog, key=key, built_with=built_with),
        ensure_ascii=False,
        indent=None,
        separators=(",", ":"),
        sort_keys=True,
    )

    # Written to a temporary file in the SAME directory and then replaced, so a
    # reader never observes a half-written catalog. os.replace is atomic on
    # both POSIX and Windows; a plain open-and-write is not, and the reader
    # here is a daemon that may be loading this file at any moment.
    #
    # The temp file must share the target's directory: os.replace is only
    # atomic within a filesystem, and a cache directory can easily sit on a
    # different mount from the system temp.
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        # Includes KeyboardInterrupt on purpose: an interrupted save must not
        # leave a stray .tmp file beside the catalog on every Ctrl-C.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return target


def load(path: str | os.PathLike[str] | None = None) -> CatalogFile | None:
    """Read a catalog. Returns None for anything unusable.

    Total by design. This is called on the daemon's startup path, where the
    difference between "no catalog" and "an exception" is the difference
    between degraded completion and a shell with a broken Tab key.
    """
    target = Path(path) if path is not None else default_path()
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None
    return _loads(data)
