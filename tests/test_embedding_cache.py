"""The vector cache, and the wrapper that uses it.

The governing property under test is that this cache is never load-bearing:
every way of corrupting the file has to produce an empty cache, not a wrong
vector. A cache miss costs milliseconds; a cache that returns another model's
vectors produces a confidently wrong ranking and nothing raises.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from cl_ai.embedding.cache import (
    _HEADER,
    _MAGIC,
    CachingEmbedder,
    VectorCache,
    default_vector_path,
    model_key,
)


class FakeEmbedder:
    """Deterministic, counts calls, so cache hits are observable."""

    def __init__(self, dim: int = 4, *, width: int | None = None) -> None:
        self._dim = dim
        self._width = dim if width is None else width
        self.calls: list[list[str]] = []

    def dim(self) -> int:
        return self._dim

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) + i) for i in range(self._width)] for t in texts]

    @property
    def embedded(self) -> int:
        return sum(len(c) for c in self.calls)


# -- round tripping -------------------------------------------------------


def test_a_saved_cache_reads_back_identically(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    cache = VectorCache("m1")
    cache.put("alpha", [1.0, 2.0, 3.0])
    cache.put("beta", [-1.5, 0.0, 0.25])
    assert cache.save(path)

    loaded = VectorCache.load(path, "m1")
    assert loaded.dim == 3
    assert len(loaded) == 2
    assert loaded.get("alpha") == [1.0, 2.0, 3.0]
    assert loaded.get("beta") == [-1.5, 0.0, 0.25]
    assert loaded.get("gamma") is None


def test_saving_clears_the_dirty_flag(tmp_path: Path) -> None:
    cache = VectorCache("m1")
    assert not cache.dirty
    cache.put("a", [1.0])
    assert cache.dirty
    cache.save(tmp_path / "v.bin")
    assert not cache.dirty


def test_an_empty_cache_declines_to_save(tmp_path: Path) -> None:
    """Nothing to write and no dim to record -- a header alone is not a cache."""
    path = tmp_path / "v.bin"
    assert VectorCache("m1").save(path) is False
    assert not path.exists()


def test_unicode_text_keys_survive_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    cache = VectorCache("m1")
    for text in ("naïve", "日本語", "emoji 🚀", "tab\there"):
        cache.put(text, [1.0, 2.0])
    cache.save(path)
    loaded = VectorCache.load(path, "m1")
    for text in ("naïve", "日本語", "emoji 🚀", "tab\there"):
        assert loaded.get(text) == [1.0, 2.0], text


def test_a_non_ascii_model_key_round_trips(tmp_path: Path) -> None:
    """The key is length-prefixed in BYTES, not characters."""
    path = tmp_path / "v.bin"
    cache = VectorCache("modèle-日本")
    cache.put("a", [1.0, 2.0])
    cache.save(path)
    assert VectorCache.load(path, "modèle-日本").get("a") == [1.0, 2.0]
    assert len(VectorCache.load(path, "modele")) == 0


def test_the_file_is_byte_order_independent(tmp_path: Path) -> None:
    """Little-endian is written explicitly, so the bytes are pinned here.

    Without this the format silently depends on the writing machine, and the
    failure only ever shows up on hardware CI does not have.
    """
    path = tmp_path / "v.bin"
    cache = VectorCache("m")
    cache.put("x", [1.0, 2.0])
    cache.save(path)
    raw = path.read_bytes()
    magic, dim, count, keylen = _HEADER.unpack_from(raw, 0)
    assert (magic, dim, count, keylen) == (_MAGIC, 2, 1, 1)
    body = raw[_HEADER.size + keylen :]
    assert body[32:] == struct.pack("<ff", 1.0, 2.0)


# -- refusing to be load-bearing ------------------------------------------


def test_a_missing_file_loads_as_empty(tmp_path: Path) -> None:
    assert len(VectorCache.load(tmp_path / "nope.bin", "m")) == 0


def test_a_directory_in_place_of_the_file_loads_as_empty(tmp_path: Path) -> None:
    target = tmp_path / "v.bin"
    target.mkdir()
    assert len(VectorCache.load(target, "m")) == 0


def test_a_foreign_model_key_is_rejected_wholesale(tmp_path: Path) -> None:
    """The failure this format exists to prevent.

    Two models' vectors load cleanly, score cleanly, and rank nonsense. There
    is no partial-trust answer here, so the whole file is dropped.
    """
    path = tmp_path / "v.bin"
    cache = VectorCache("needle3:a")
    cache.put("alpha", [1.0, 2.0, 3.0])
    cache.save(path)
    assert len(VectorCache.load(path, "needle3:b")) == 0


def test_garbage_loads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(b"not a vector cache at all, not even close")
    assert len(VectorCache.load(path, "m")) == 0


def test_an_empty_file_loads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(b"")
    assert len(VectorCache.load(path, "m")) == 0


@pytest.mark.parametrize("keep", [1, 8, 12, 20, 40, 60])
def test_every_truncation_loads_as_empty(tmp_path: Path, keep: int) -> None:
    """A writer that died mid-file must not yield a partly-valid cache."""
    path = tmp_path / "v.bin"
    cache = VectorCache("m")
    cache.put("alpha", [1.0, 2.0, 3.0])
    cache.put("beta", [4.0, 5.0, 6.0])
    cache.save(path)
    full = path.read_bytes()
    path.write_bytes(full[:keep])
    assert len(VectorCache.load(path, "m")) == 0


def test_trailing_junk_loads_as_empty(tmp_path: Path) -> None:
    """The length check is exact, not a lower bound."""
    path = tmp_path / "v.bin"
    cache = VectorCache("m")
    cache.put("alpha", [1.0, 2.0])
    cache.save(path)
    with path.open("ab") as handle:
        handle.write(b"\x00\x01\x02")
    assert len(VectorCache.load(path, "m")) == 0


def test_a_wrong_magic_loads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    cache = VectorCache("m")
    cache.put("a", [1.0])
    cache.save(path)
    raw = bytearray(path.read_bytes())
    raw[0:8] = b"OTHERFMT"
    path.write_bytes(bytes(raw))
    assert len(VectorCache.load(path, "m")) == 0


def test_an_absurd_dim_is_refused_without_allocating(tmp_path: Path) -> None:
    """A corrupt dim read as billions must be rejected by the bound, not by
    running out of memory first."""
    path = tmp_path / "v.bin"
    path.write_bytes(_HEADER.pack(_MAGIC, 0xFFFFFFF0, 1, 0))
    assert len(VectorCache.load(path, "")) == 0


def test_a_zero_dim_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(_HEADER.pack(_MAGIC, 0, 0, 0))
    assert len(VectorCache.load(path, "")) == 0


def test_an_absurd_row_count_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(_HEADER.pack(_MAGIC, 4, 0xFFFFFFF0, 0))
    assert len(VectorCache.load(path, "")) == 0


def test_a_keylen_past_the_end_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(_HEADER.pack(_MAGIC, 4, 0, 10_000))
    assert len(VectorCache.load(path, "")) == 0


def test_an_undecodable_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    path.write_bytes(_HEADER.pack(_MAGIC, 4, 0, 2) + b"\xff\xfe")
    assert len(VectorCache.load(path, "")) == 0


def test_an_unwritable_destination_returns_false_rather_than_raising(
    tmp_path: Path,
) -> None:
    """Failing to persist a cache must not fail the query that filled it."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    cache = VectorCache("m")
    cache.put("a", [1.0])
    assert cache.save(blocker / "sub" / "v.bin") is False


def test_a_failed_save_leaves_no_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "v.bin"
    cache = VectorCache("m")
    cache.put("a", [1.0])
    cache.save(target)
    assert [p.name for p in tmp_path.iterdir()] == ["v.bin"]


# -- dimension discipline -------------------------------------------------


def test_a_mismatched_vector_is_refused_on_the_way_in() -> None:
    """Checked at put(), because a bad row written to disk reads back valid."""
    cache = VectorCache("m")
    cache.put("a", [1.0, 2.0, 3.0])
    cache.put("b", [1.0, 2.0])
    assert cache.dim == 3
    assert cache.get("b") is None
    assert len(cache) == 1


def test_an_empty_vector_is_ignored() -> None:
    cache = VectorCache("m")
    cache.put("a", [])
    assert len(cache) == 0
    assert cache.dim == 0


def test_the_first_vector_sets_the_dim() -> None:
    cache = VectorCache("m")
    assert cache.dim == 0
    cache.put("a", [1.0] * 7)
    assert cache.dim == 7


# -- the caching wrapper --------------------------------------------------


def test_the_backend_is_called_once_per_unique_text(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    wrapper = CachingEmbedder(inner, key="m", path=tmp_path / "v.bin")
    first = wrapper.encode(["alpha", "beta"])
    second = wrapper.encode(["alpha", "beta"])
    assert first == second
    assert inner.embedded == 2, "second call should have been served from cache"


def test_duplicate_texts_in_one_batch_are_embedded_once(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    wrapper = CachingEmbedder(inner, key="m", path=tmp_path / "v.bin")
    result = wrapper.encode(["same", "other", "same", "same"])
    assert inner.calls == [["same", "other"]]
    assert result[0] == result[2] == result[3]
    assert result[1] != result[0]


def test_order_is_preserved_when_some_texts_hit_and_some_miss(
    tmp_path: Path,
) -> None:
    """The bug this guards: filling misses out of order silently transposes
    two tools' vectors, which ranks perfectly and means nothing."""
    inner = FakeEmbedder()
    wrapper = CachingEmbedder(inner, key="m", path=tmp_path / "v.bin")
    wrapper.encode(["b"])
    mixed = wrapper.encode(["a", "b", "cc"])
    assert mixed == [inner.encode([t])[0] for t in ("a", "b", "cc")]


def test_the_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    first = FakeEmbedder()
    CachingEmbedder(first, key="m", path=path).encode(["alpha", "beta"])
    assert first.embedded == 2

    second = FakeEmbedder()
    wrapper = CachingEmbedder(second, key="m", path=path)
    assert wrapper.encode(["alpha", "beta"]) == [
        [5.0, 6.0, 7.0, 8.0],  # len("alpha") + i
        [4.0, 5.0, 6.0, 7.0],  # len("beta") + i
    ]
    assert second.embedded == 0, "nothing should have been re-embedded"


def test_a_changed_model_key_discards_the_whole_cache(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    CachingEmbedder(FakeEmbedder(), key="old", path=path).encode(["alpha"])
    fresh = FakeEmbedder()
    CachingEmbedder(fresh, key="new", path=path).encode(["alpha"])
    assert fresh.embedded == 1


def test_autosave_off_writes_nothing_until_asked(tmp_path: Path) -> None:
    path = tmp_path / "v.bin"
    wrapper = CachingEmbedder(
        FakeEmbedder(), key="m", path=path, autosave=False
    )
    wrapper.encode(["alpha"])
    assert not path.exists()
    assert wrapper.save()
    assert path.exists()


def test_a_backend_returning_the_wrong_row_count_raises(tmp_path: Path) -> None:
    """Silently accepting this pairs each text with another text's vector."""

    class Broken(FakeEmbedder):
        def encode(self, texts):
            return super().encode(texts)[:-1]

    wrapper = CachingEmbedder(Broken(), key="m", path=tmp_path / "v.bin")
    with pytest.raises(ValueError, match="1 vectors for 2 texts"):
        wrapper.encode(["a", "b"])


def test_an_empty_batch_never_reaches_the_backend(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    wrapper = CachingEmbedder(inner, key="m", path=tmp_path / "v.bin")
    assert wrapper.encode([]) == []
    assert inner.calls == []


def test_dim_is_answered_from_the_cache_once_it_is_populated(
    tmp_path: Path,
) -> None:
    """So a warm start never has to load the model just to report a width."""
    path = tmp_path / "v.bin"
    CachingEmbedder(FakeEmbedder(dim=4), key="m", path=path).encode(["alpha"])

    class Exploding(FakeEmbedder):
        def dim(self) -> int:
            raise AssertionError("model should not have been loaded")

    assert CachingEmbedder(Exploding(), key="m", path=path).dim() == 4


def test_dim_falls_back_to_the_backend_when_cold(tmp_path: Path) -> None:
    wrapper = CachingEmbedder(FakeEmbedder(dim=9), key="m", path=tmp_path / "v.bin")
    assert wrapper.dim() == 9


# -- keys and paths -------------------------------------------------------


def test_model_key_includes_the_weights_size(tmp_path: Path) -> None:
    weights = tmp_path / "needle3.cact"
    weights.write_bytes(b"x" * 100)
    assert model_key("needle3", weights) == "needle3:needle3.cact:100"


def test_model_key_changes_when_the_weights_change(tmp_path: Path) -> None:
    """A local finetune keeps the filename, so size is what distinguishes it."""
    weights = tmp_path / "needle3.cact"
    weights.write_bytes(b"x" * 100)
    before = model_key("needle3", weights)
    weights.write_bytes(b"x" * 200)
    assert model_key("needle3", weights) != before


def test_model_key_survives_missing_weights(tmp_path: Path) -> None:
    assert model_key("needle3", tmp_path / "gone.cact") == "needle3:gone.cact"
    assert model_key("needle3", None) == "needle3"


def test_the_vectors_path_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CL_AI_VECTORS", os.path.join("somewhere", "v.bin"))
    assert default_vector_path("anything") == Path("somewhere") / "v.bin"


def test_different_models_get_different_default_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise two models fight over one file and each discards the other's."""
    monkeypatch.delenv("CL_AI_VECTORS", raising=False)
    assert default_vector_path("a") != default_vector_path("b")


def test_the_default_path_survives_a_homeless_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CL_AI_VECTORS", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(
        Path, "home", staticmethod(lambda: (_ for _ in ()).throw(RuntimeError()))
    )
    assert default_vector_path("m").name.startswith("vectors-")
