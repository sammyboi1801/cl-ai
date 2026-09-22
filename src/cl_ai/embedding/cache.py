"""On-disk vector cache, and the wrapper that makes an Embedder use it.

WHY THIS EXISTS
Needle embeds one text at a time at ~17ms. A 900-tool catalog is therefore
~15 seconds of model time before the first query can be answered, and the
daemon is expected to come up behind a keystroke. Embedding is also perfectly
deterministic for a fixed model, so paying that cost more than once is pure
waste.

Cached per TEXT rather than per corpus. Per-corpus would be one line shorter
and would throw the whole 15 seconds away the moment a single tool changed --
which is exactly what happens when a catalog tier is added or a tldr page is
updated. Per-text means a changed catalog re-embeds only what actually
changed.

A CACHE MUST NEVER BE LOAD-BEARING
Every failure path here returns an empty cache rather than raising: a
truncated file, a foreign model, a dimension change, a permission error. The
cost of a cache miss is latency; the cost of trusting a corrupt one is a
confidently wrong ranking. Those are not comparable, so there is no path
through this module where bad bytes reach a caller.

FORMAT
    magic   b"CLAIVEC1"      8 bytes
    dim     uint32 LE
    count   uint32 LE
    keylen  uint32 LE
    key     `keylen` bytes, utf-8    -- model identity, see model_key()
    entries `count` x (32-byte sha256 of the text + `dim` float32 LE)

Little-endian is written explicitly so a cache file is not silently misread on
a big-endian machine. `array` is native-order, hence the byteswaps.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import tempfile
from array import array
from collections.abc import Sequence
from pathlib import Path

__all__ = ["CachingEmbedder", "VectorCache", "default_vector_path", "model_key"]

_MAGIC = b"CLAIVEC1"
_HEADER = struct.Struct("<8sIII")
_DIGEST_BYTES = 32

#: Refuse to allocate for an implausible header. A corrupt `dim` read as a
#: few billion would otherwise try to reserve the whole address space before
#: the length check below could reject it.
_MAX_DIM = 1 << 16
_MAX_ROWS = 1 << 22


def model_key(name: str, weights: str | os.PathLike[str] | None) -> str:
    """Identity of the model that produced a set of vectors.

    Vectors from two different models are not comparable, and mixing them
    produces a ranking that looks fine and is meaningless. The weights file's
    size is included because a local finetune keeps the same filename; mtime
    is deliberately NOT, since reinstalling identical weights would needlessly
    discard a valid cache.
    """
    if weights is None:
        return name
    path = Path(weights)
    try:
        size = path.stat().st_size
    except OSError:
        return f"{name}:{path.name}"
    return f"{name}:{path.name}:{size}"


def default_vector_path(key: str) -> Path:
    """Where vectors live, following the same convention as the catalog."""
    override = os.environ.get("CL_AI_VECTORS")
    if override:
        return Path(override)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    name = f"vectors-{digest}.bin"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
        return Path(base) / "cl-ai" / name
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "cl-ai" / name
    try:
        home = Path.home()
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "cl-ai" / name
    return home / ".cache" / "cl-ai" / name


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class VectorCache:
    """A text -> vector map backed by one file. Not thread-safe by itself."""

    def __init__(self, key: str, dim: int = 0) -> None:
        self.key = key
        self.dim = dim
        self._rows: dict[bytes, array[float]] = {}
        self._dirty = False

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def get(self, text: str) -> list[float] | None:
        row = self._rows.get(_digest(text))
        return None if row is None else list(row)

    def put(self, text: str, vector: Sequence[float]) -> None:
        """Store one vector. Ignored if it disagrees with the cache's dim.

        A silent mismatch here would be written to disk and read back as a
        valid row, so the check has to happen on the way in rather than on the
        way out.
        """
        if not vector:
            return
        if self.dim == 0:
            self.dim = len(vector)
        elif len(vector) != self.dim:
            return
        self._rows[_digest(text)] = array("f", vector)
        self._dirty = True

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str], key: str) -> VectorCache:
        """Read a cache file. Returns an EMPTY cache on any problem at all."""
        empty = cls(key)
        try:
            raw = Path(path).read_bytes()
        except OSError:
            return empty
        if len(raw) < _HEADER.size:
            return empty
        magic, dim, count, keylen = _HEADER.unpack_from(raw, 0)
        if magic != _MAGIC:
            return empty
        if not 0 < dim <= _MAX_DIM or count > _MAX_ROWS or keylen > len(raw):
            return empty

        start = _HEADER.size + keylen
        if start > len(raw):
            return empty
        try:
            stored_key = raw[_HEADER.size : start].decode("utf-8")
        except UnicodeDecodeError:
            return empty
        # A different model's vectors are worse than none: they load cleanly,
        # score cleanly, and rank nonsense.
        if stored_key != key:
            return empty

        stride = _DIGEST_BYTES + dim * 4
        # Exact, not >=. A trailing partial row means the writer died, and the
        # rows before it are no more trustworthy than the one that is missing.
        if len(raw) - start != count * stride:
            return empty

        cache = cls(key, dim)
        swap = sys.byteorder != "little"
        for i in range(count):
            offset = start + i * stride
            digest = raw[offset : offset + _DIGEST_BYTES]
            values: array[float] = array("f")
            values.frombytes(raw[offset + _DIGEST_BYTES : offset + stride])
            if swap:
                values.byteswap()
            cache._rows[digest] = values
        return cache

    def save(self, path: str | os.PathLike[str]) -> bool:
        """Write atomically. Returns False rather than raising on failure.

        Failing to persist a cache is not a reason to fail a query, so an
        unwritable cache directory degrades to "recompute next time".
        """
        if self.dim == 0:
            return False
        target = Path(path)
        encoded = self.key.encode("utf-8")
        chunks = [_HEADER.pack(_MAGIC, self.dim, len(self._rows), len(encoded)), encoded]
        swap = sys.byteorder != "little"
        for digest, values in self._rows.items():
            chunks.append(digest)
            if swap:
                copy = array("f", values)
                copy.byteswap()
                chunks.append(copy.tobytes())
            else:
                chunks.append(values.tobytes())
        payload = b"".join(chunks)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Same directory as the target: os.replace is only atomic within a
            # filesystem, and a cache dir can sit on a different mount from
            # the system temp. See catalog/store.py, which does this too.
            fd, tmp_name = tempfile.mkstemp(
                dir=target.parent, prefix=target.name, suffix=".tmp"
            )
        except OSError:
            return False
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except OSError:
            _unlink(tmp_name)
            return False
        except BaseException:
            # KeyboardInterrupt and the like still have to clean up after
            # themselves, but they are not "the cache could not be written".
            _unlink(tmp_name)
            raise
        self._dirty = False
        return True


class CachingEmbedder:
    """An `Embedder` that consults a `VectorCache` before its backend.

    Deliberately a wrapper rather than a feature of the adapter: caching is
    identical for every backend, and folding it into one would mean writing it
    again for the next.
    """

    def __init__(
        self,
        inner: object,
        *,
        key: str,
        path: str | os.PathLike[str] | None = None,
        autosave: bool = True,
    ) -> None:
        self._inner = inner
        self._path = Path(path) if path is not None else default_vector_path(key)
        self._cache = VectorCache.load(self._path, key)
        self._autosave = autosave

    @property
    def cache(self) -> VectorCache:
        return self._cache

    @property
    def path(self) -> Path:
        return self._path

    def dim(self) -> int:
        if self._cache.dim:
            return self._cache.dim
        return int(self._inner.dim())  # type: ignore[attr-defined]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Fill from cache, embed only the misses, preserve input order."""
        results: list[list[float] | None] = []
        # Deduplicated: a catalog routinely contains two tools with identical
        # description text, and embedding it twice in one batch is 17ms for
        # nothing.
        missing: dict[str, list[int]] = {}
        for position, text in enumerate(texts):
            hit = self._cache.get(text)
            results.append(hit)
            if hit is None:
                missing.setdefault(text, []).append(position)

        if missing:
            wanted = list(missing)
            fresh = self._inner.encode(wanted)  # type: ignore[attr-defined]
            if len(fresh) != len(wanted):
                raise ValueError(
                    f"embedder returned {len(fresh)} vectors for {len(wanted)} texts"
                )
            for text, vector in zip(wanted, fresh):
                self._cache.put(text, vector)
                for position in missing[text]:
                    results[position] = list(vector)
            if self._autosave and self._cache.dirty:
                self._cache.save(self._path)

        return [r if r is not None else [] for r in results]

    def save(self) -> bool:
        return self._cache.save(self._path)
