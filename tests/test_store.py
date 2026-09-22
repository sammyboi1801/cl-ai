"""Tests for catalog persistence.

The two properties that matter: a load can never raise (it runs on the
daemon's startup path, where an exception costs the user a working Tab key),
and a save can never be observed half-written (the daemon may be reading the
file at any moment).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cl_ai.catalog import store
from cl_ai.catalog.discovery import Discovered, EntryKind
from cl_ai.catalog.extract.base import RawParam, RawTool
from cl_ai.catalog.normalize import Family, normalize
from cl_ai.ir import Capability, ParamKind, Provenance, SourceTier, Tool


def sample_catalog():
    return normalize([
        RawTool(
            binary="git",
            path=("commit",),
            description="Commit files",
            tier=SourceTier.TLDR,
            source="tldr/common/git-commit.md",
            os_targets=frozenset({"common"}),
            homepage="https://git-scm.com",
            params=(
                RawParam(
                    name="message",
                    kind=ParamKind.OPTION,
                    short="-m",
                    long="--message",
                    type="string",
                ),
                RawParam(name="file", kind=ParamKind.POSITIONAL, type="array",
                         repeatable=True),
            ),
        ),
        RawTool(
            binary="dir",
            path=(),
            description="List directory",
            tier=SourceTier.TLDR,
            source="tldr/windows/dir.md",
            os_targets=frozenset({"windows"}),
        ),
    ])


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

def test_round_trip_preserves_the_catalog(tmp_path: Path) -> None:
    catalog = sample_catalog()
    path = store.save(catalog, tmp_path / "c.json", key="k")
    loaded = store.load(path)
    assert loaded is not None
    assert loaded.catalog == catalog


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    catalog = sample_catalog()
    path = store.save(catalog, tmp_path / "c.json", key="k")
    loaded = store.load(path)
    assert loaded is not None
    tool = loaded.catalog.select("git_commit", "bash", "linux")
    assert tool is not None
    assert tool.binary == "git"
    assert tool.path == ("commit",)
    assert tool.homepage == "https://git-scm.com"
    assert Capability.READS in tool.capabilities
    option = next(p for p in tool.params if p.kind is ParamKind.OPTION)
    assert option.flag == "--message"
    assert option.short == "-m"
    assert option.type == "string"
    assert option.provenance is not None
    assert option.provenance.tier is SourceTier.TLDR
    assert option.provenance.source == "tldr/common/git-commit.md"
    positional = next(p for p in tool.params if p.kind is ParamKind.POSITIONAL)
    assert positional.repeatable
    assert positional.type == "array"


def test_variant_families_survive(tmp_path: Path) -> None:
    """Losing the family would re-merge Windows and GNU tools on reload."""
    path = store.save(sample_catalog(), tmp_path / "c.json")
    loaded = store.load(path)
    assert loaded is not None
    families = {v.family for v in loaded.catalog.variants}
    assert Family.WINDOWS in families
    assert Family.COMMON in families


def test_provenance_conflicts_survive(tmp_path: Path) -> None:
    catalog = normalize([
        RawTool(binary="t", path=(), description="a", tier=SourceTier.TLDR, source="a"),
        RawTool(binary="t", path=(), description="b", tier=SourceTier.TLDR, source="b"),
    ])
    path = store.save(catalog, tmp_path / "c.json")
    loaded = store.load(path)
    assert loaded is not None
    tool = loaded.catalog.select("t")
    assert tool is not None
    assert tool.provenance is not None
    assert tool.provenance.conflicts


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_saves_are_byte_identical(tmp_path: Path) -> None:
    """An artifact that differs between identical builds cannot be trusted."""
    catalog = sample_catalog()
    a = store.save(catalog, tmp_path / "a.json", key="k", built_with=["tldr"])
    b = store.save(catalog, tmp_path / "b.json", key="k", built_with=["tldr"])
    assert a.read_bytes() == b.read_bytes()


def test_saved_file_uses_lf_endings(tmp_path: Path) -> None:
    """So the artifact is identical on Windows and POSIX."""
    path = store.save(sample_catalog(), tmp_path / "c.json")
    assert b"\r\n" not in path.read_bytes()


# --------------------------------------------------------------------------
# Validity
# --------------------------------------------------------------------------

def test_key_mismatch_invalidates(tmp_path: Path) -> None:
    path = store.save(sample_catalog(), tmp_path / "c.json", key="k1")
    loaded = store.load(path)
    assert loaded is not None
    assert loaded.is_valid_for("k1")
    assert not loaded.is_valid_for("k2")


def test_empty_key_is_never_valid(tmp_path: Path) -> None:
    """An unkeyed file cannot be proven current, so it must not be trusted."""
    path = store.save(sample_catalog(), tmp_path / "c.json")
    loaded = store.load(path)
    assert loaded is not None
    assert not loaded.is_valid_for("")


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    """Refusing costs one rebuild; guessing costs wrong commands."""
    path = tmp_path / "c.json"
    payload = store.dump(sample_catalog(), key="k")
    payload["schema_version"] = store.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.load(path) is None


# --------------------------------------------------------------------------
# Loading is total
# --------------------------------------------------------------------------

def test_missing_file(tmp_path: Path) -> None:
    assert store.load(tmp_path / "nope.json") is None


def test_directory_instead_of_file(tmp_path: Path) -> None:
    target = tmp_path / "adir"
    target.mkdir()
    assert store.load(target) is None


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[]",
        "null",
        '{"schema_version": 1}',
        '{"schema_version": 1, "variants": "not a list"}',
        '{"schema_version": 1, "variants": [null, 3, "x"]}',
        '{"schema_version": 1, "variants": [{"family": "posix"}]}',
        '{"schema_version": 1, "variants": [{"tool": {}}]}',
        '﻿{"schema_version": 1, "variants": []}',
    ],
)
def test_malformed_files_never_raise(tmp_path: Path, content: str) -> None:
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    result = store.load(path)
    assert result is None or result.catalog is not None


def test_undecodable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_bytes(b"\xff\xfe\x00 not utf-8")
    assert store.load(path) is None


def test_one_bad_variant_does_not_lose_the_others(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    payload = store.dump(sample_catalog(), key="k")
    payload["variants"].append({"family": "posix", "tool": {"no_name": True}})
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.load(path)
    assert loaded is not None
    assert len(loaded.catalog) == 2


def test_unknown_family_degrades_rather_than_failing(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    payload = store.dump(sample_catalog(), key="k")
    payload["variants"][0]["family"] = "martian"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.load(path)
    assert loaded is not None
    assert Family.UNKNOWN in {v.family for v in loaded.catalog.variants}


# --------------------------------------------------------------------------
# Saving is atomic
# --------------------------------------------------------------------------

def test_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    store.save(sample_catalog(), tmp_path / "c.json", key="k")
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "c.json"]
    assert leftovers == []


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.json"
    assert store.save(sample_catalog(), target) == target
    assert target.is_file()


def test_save_overwrites_atomically(tmp_path: Path) -> None:
    target = tmp_path / "c.json"
    store.save(sample_catalog(), target, key="k1")
    first = target.read_bytes()
    store.save(normalize([]), target, key="k2")
    assert target.read_bytes() != first
    loaded = store.load(target)
    assert loaded is not None
    assert len(loaded.catalog) == 0


def test_temp_file_shares_the_target_directory(tmp_path: Path, monkeypatch) -> None:
    """os.replace is only atomic within one filesystem."""
    seen: list[str] = []
    real = store.tempfile.mkstemp

    def spy(*args, **kwargs):
        seen.append(str(kwargs.get("dir")))
        return real(*args, **kwargs)

    monkeypatch.setattr(store.tempfile, "mkstemp", spy)
    target = tmp_path / "sub" / "c.json"
    store.save(sample_catalog(), target)
    assert seen == [str(target.parent)]


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------

def entry(name: str, path: str, size: int = 10, mtime: int = 100) -> Discovered:
    return Discovered(
        name=name, kind=EntryKind.EXECUTABLE, path=path, size=size, mtime_ns=mtime
    )


def test_cache_key_is_stable_and_order_independent() -> None:
    a = [entry("git", "/usr/bin/git"), entry("ls", "/bin/ls")]
    assert store.cache_key(a) == store.cache_key(list(reversed(a)))


@pytest.mark.parametrize(
    "changed",
    [
        entry("git", "/usr/bin/git", size=11),
        entry("git", "/usr/bin/git", mtime=101),
        entry("git", "/usr/local/bin/git"),
        entry("gitk", "/usr/bin/git"),
    ],
)
def test_cache_key_notices_every_identity_change(changed: Discovered) -> None:
    base = [entry("git", "/usr/bin/git")]
    assert store.cache_key(base) != store.cache_key([changed])


def test_cache_key_notices_a_new_binary() -> None:
    base = [entry("git", "/usr/bin/git")]
    assert store.cache_key(base) != store.cache_key([*base, entry("ls", "/bin/ls")])


def test_cache_key_includes_the_tier_set() -> None:
    """A catalog goes stale when the build changes, not only the machine."""
    base = [entry("git", "/usr/bin/git")]
    assert store.cache_key(base, extra="tldr") != store.cache_key(
        base, extra="tldr+mandocs"
    )


def test_cache_key_of_nothing_is_stable() -> None:
    assert store.cache_key([]) == store.cache_key([])


# --------------------------------------------------------------------------
# Binary identity
# --------------------------------------------------------------------------

def test_identity_changes_with_content(tmp_path: Path) -> None:
    target = tmp_path / "bin"
    target.write_bytes(b"aaaa")
    first = store.identity_of(target)
    target.write_bytes(b"bbbb")
    assert store.identity_of(target) != first


def test_identity_of_missing_path_is_a_value_not_an_error(tmp_path: Path) -> None:
    """One odd PATH entry must not fail a whole cache validation."""
    assert store.identity_of(tmp_path / "nope") == "unreadable"


def test_identity_of_a_directory(tmp_path: Path) -> None:
    assert store.identity_of(tmp_path) == "unreadable"


def test_identity_is_bounded(tmp_path: Path) -> None:
    """Hashing a whole binary across a thousand PATH entries is too slow."""
    target = tmp_path / "big"
    target.write_bytes(b"x" * 8)
    short = store.identity_of(target, read_bytes=4)
    assert len(short) == 32


# --------------------------------------------------------------------------
# Default location
# --------------------------------------------------------------------------

def test_default_path_honours_the_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CL_AI_CATALOG", str(tmp_path / "x.json"))
    assert store.default_path() == tmp_path / "x.json"


def test_default_path_is_in_a_cache_location(monkeypatch) -> None:
    """A cache is data the user may delete; it belongs where the OS expects."""
    monkeypatch.delenv("CL_AI_CATALOG", raising=False)
    path = store.default_path()
    assert path.name == "catalog.json"
    assert "cl-ai" in path.parts


def test_default_path_never_raises_without_a_home(monkeypatch) -> None:
    monkeypatch.delenv("CL_AI_CATALOG", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    if os.name != "nt":
        def no_home() -> Path:
            raise RuntimeError("no home")

        monkeypatch.setattr(Path, "home", staticmethod(no_home))
    assert store.default_path().name == "catalog.json"


def test_tool_with_no_params_round_trips(tmp_path: Path) -> None:
    catalog = normalize([
        RawTool(binary="x", path=(), description="d", tier=SourceTier.TLDR, source="s")
    ])
    path = store.save(catalog, tmp_path / "c.json")
    loaded = store.load(path)
    assert loaded is not None
    assert loaded.catalog == catalog


def test_dump_is_json_serialisable() -> None:
    """Guards against a set or enum sneaking into the payload."""
    json.dumps(store.dump(sample_catalog(), key="k"))


def test_tool_without_provenance_round_trips(tmp_path: Path) -> None:
    hand = Tool(name="x", description="d", binary="x", provenance=None)
    payload = {
        "schema_version": store.SCHEMA_VERSION,
        "key": "",
        "built_with": [],
        "variants": [{"family": "common", "tool": store._tool_to_json(hand)}],
    }
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.load(path)
    assert loaded is not None
    assert loaded.catalog.select("x") == hand


def test_provenance_with_bad_tier_is_dropped_not_fatal(tmp_path: Path) -> None:
    payload = store.dump(sample_catalog(), key="k")
    payload["variants"][0]["tool"]["provenance"] = {"tier": 999, "source": "x"}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.load(path)
    assert loaded is not None
    assert len(loaded.catalog) == 2


def test_known_provenance_survives() -> None:
    provenance = Provenance(
        tier=SourceTier.COMPLETIONS, source="s", confidence=0.5, conflicts=("a",)
    )
    restored = store._provenance_from_json(store._provenance_to_json(provenance))
    assert restored == provenance
