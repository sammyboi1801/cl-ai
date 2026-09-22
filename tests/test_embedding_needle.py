"""The Needle embedder adapter.

Split in two. Everything above the integration section runs against a stub
model, because the properties that matter -- centering discipline, degradation
when the backend is absent, never raising at import -- are properties of the
ADAPTER, and pinning them to a 3GB checkpoint would mean CI never checks them.

The integration section runs the real model and is skipped without one. It is
small on purpose: it exists to prove the stub is not a fiction.
"""

from __future__ import annotations

import hashlib
import math
import sys
import types
from pathlib import Path

import pytest

from cl_ai.embedding import build_embedder, needle_embedder
from cl_ai.embedding.needle_embedder import (
    NeedleEmbedder,
    NeedleUnavailable,
    default_weights,
)
from cl_ai.ports import Embedder


class StubAgent:
    """Stands in for needle.Needle, with the real model's ANISOTROPY.

    Shaped deliberately: a large constant offset shared by every text, plus a
    small text-dependent signal. That is what Needle actually does -- measured,
    unrelated texts sit at ~0.93 cosine because the shared component dwarfs
    the meaningful one -- and it is the only stub shape under which the
    centering tests say anything.

    A first attempt returned `10 + len(text) + i`, which is collinear across
    texts: centering turned every vector into +-1 and the test that centering
    helps failed against correct code.
    """

    def __init__(self, weights: str | None = None, dim: int = 4) -> None:
        self.weights = weights
        self.dim = dim
        self.calls: list[str] = []
        self.closed = False

    def embed(self, text: str = "") -> list[float]:
        self.calls.append(text)
        noise = hashlib.sha256(text.encode("utf-8")).digest()
        return [10.0 + noise[i % len(noise)] / 255.0 for i in range(self.dim)]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def stub_needle(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake `needle` module for the duration of one test."""
    module = types.ModuleType("needle")
    module.Needle = StubAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    return module


@pytest.fixture
def no_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no model at all.

    `NeedleEmbedder(None)` means "discover", not "none", so a test that wants
    the no-model path has to suppress discovery -- otherwise it passes on CI
    and fails on a developer machine that happens to have weights.
    """
    monkeypatch.setattr(needle_embedder, "default_weights", lambda: None)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "needle3.cact"
    path.write_bytes(b"not really weights")
    return path


def _embedder(weights: Path) -> NeedleEmbedder:
    return NeedleEmbedder(weights)


# -- the port contract ----------------------------------------------------


def test_the_adapter_satisfies_the_embedder_port() -> None:
    assert isinstance(NeedleEmbedder(None), Embedder)


def test_constructing_never_touches_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction happens on the daemon's startup path. Loading a model
    there would be 0.6s before the first keystroke can be served."""
    monkeypatch.setitem(sys.modules, "needle", None)
    NeedleEmbedder("/nonexistent/weights.cact")


def test_encode_returns_one_vector_per_text(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    vectors = _embedder(weights).encode(["a", "bb", "ccc"])
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)


def test_encode_of_nothing_is_nothing(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    embedder = _embedder(weights)
    assert embedder.encode([]) == []
    assert embedder._agent is None, "an empty batch should not load the model"


def test_every_returned_vector_is_unit_length(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    for vector in _embedder(weights).encode(["a", "bb", "ccc", "dddd"]):
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-6)


def test_dim_is_discovered_without_a_real_query(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    embedder = _embedder(weights)
    assert embedder.dim() == 4
    agent = embedder._agent
    assert isinstance(agent, StubAgent)
    before = len(agent.calls)
    assert embedder.dim() == 4
    assert len(agent.calls) == before, "dim should be cached after the probe"


def test_the_model_is_loaded_once_across_many_batches(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    embedder = _embedder(weights)
    embedder.encode(["a"])
    first = embedder._agent
    embedder.encode(["b"])
    assert embedder._agent is first


# -- centering, and the batch-of-one trap ---------------------------------


def test_a_single_text_does_not_center_to_zero(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """THE trap this adapter is written around.

    Centering against the batch's own mean is the obvious implementation and
    is silently catastrophic: at query time the batch is one text, its own
    mean is itself, and the result is the zero vector -- cosine zero against
    everything, no ranking at all, and nothing raises.
    """
    vector = _embedder(weights).encode(["list files"])[0]
    assert any(v != 0.0 for v in vector)
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-6)


def test_a_fitted_mean_is_used_for_a_batch_of_one(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """The corpus mean, not the batch mean -- that is the entire distinction."""
    embedder = _embedder(weights).fit(["a", "bb", "ccc", "dddd", "eeeee"])
    alone = embedder.encode(["bb"])[0]
    together = embedder.encode(["bb", "zzzz"])[0]
    assert alone == together


def test_without_fit_the_adapter_does_not_center(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """Documented default. VectorReranker centers the corpus itself and does
    not trust the adapter, so this is safe in the standard wiring."""
    embedder = _embedder(weights)
    assert embedder.mean is None
    raw = embedder.encode(["abc"])[0]
    assert all(v > 0 for v in raw), "uncentered stub vectors are all positive"


def test_fit_makes_the_corpus_mean_available(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    embedder = _embedder(weights).fit(["a", "bb", "ccc"])
    mean = embedder.mean
    assert mean is not None and len(mean) == 4


def test_fit_refuses_a_single_text(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """One text's mean is itself; centering by it yields the zero vector."""
    with pytest.raises(ValueError, match="at least two"):
        _embedder(weights).fit(["only one"])
    with pytest.raises(ValueError, match="at least two"):
        _embedder(weights).fit([])


def test_centering_widens_the_spread_between_unrelated_texts(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """The measured justification for centering, in miniature.

    Raw Needle vectors sit at ~0.93 cosine regardless of meaning. Removing the
    corpus mean is what turns that band into a usable ranking, so a change
    that quietly stopped centering has to fail something.
    """
    corpus = ["a", "bb", "ccc", "dddd", "eeeee"]

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(p * q for p, q in zip(x, y))

    plain = _embedder(weights).encode(corpus)
    fitted = _embedder(weights).fit(corpus).encode(corpus)

    raw_spread = max(abs(cosine(plain[0], v)) for v in plain[1:])
    centered_spread = max(abs(cosine(fitted[0], v)) for v in fitted[1:])
    assert centered_spread < raw_spread


def test_a_vector_equal_to_the_mean_becomes_zero_rather_than_raising(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dividing by a zero norm inside a keystroke handler is the bad outcome."""

    class Constant(StubAgent):
        def embed(self, text: str = "") -> list[float]:
            return [1.0, 2.0, 3.0, 4.0]

    module = types.ModuleType("needle")
    module.Needle = Constant  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)

    embedder = NeedleEmbedder(weights).fit(["a", "b"])
    assert embedder.encode(["c"]) == [[0.0, 0.0, 0.0, 0.0]]


# -- degradation ----------------------------------------------------------


def test_available_is_false_with_no_weights(no_weights: None) -> None:
    assert NeedleEmbedder(None).available() is False


def test_available_is_false_when_the_weights_are_missing(tmp_path: Path) -> None:
    assert NeedleEmbedder(tmp_path / "gone.cact").available() is False


def test_available_is_false_when_needle_is_not_importable(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "needle", None)
    assert NeedleEmbedder(weights).available() is False


def test_available_is_true_with_weights_and_a_backend(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    assert NeedleEmbedder(weights).available() is True


def test_encoding_with_no_weights_raises_needle_unavailable(
    no_weights: None,
) -> None:
    with pytest.raises(NeedleUnavailable, match="CL_AI_NEEDLE_WEIGHTS"):
        NeedleEmbedder(None).encode(["anything"])


def test_encoding_with_missing_weights_raises_needle_unavailable(
    tmp_path: Path,
) -> None:
    with pytest.raises(NeedleUnavailable, match="weights not found"):
        NeedleEmbedder(tmp_path / "gone.cact").encode(["anything"])


def test_an_unimportable_backend_raises_needle_unavailable(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "needle", None)
    with pytest.raises(NeedleUnavailable, match="not importable"):
        NeedleEmbedder(weights).encode(["anything"])


def test_a_model_that_fails_to_construct_raises_needle_unavailable(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(**_kwargs: object) -> object:
        raise MemoryError("not enough RAM for the weights")

    module = types.ModuleType("needle")
    module.Needle = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    with pytest.raises(NeedleUnavailable, match="could not load"):
        NeedleEmbedder(weights).encode(["anything"])


def test_a_build_without_embed_raises_rather_than_returning_junk(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoEmbed:
        def __init__(self, **_kwargs: object) -> None:
            pass

    module = types.ModuleType("needle")
    module.Needle = NoEmbed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    with pytest.raises(NeedleUnavailable, match="no embed"):
        NeedleEmbedder(weights).encode(["anything"])


def test_a_failing_embed_raises_needle_unavailable(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Failing(StubAgent):
        def embed(self, text: str = "") -> list[float]:
            raise RuntimeError("inference crashed")

    module = types.ModuleType("needle")
    module.Needle = Failing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    with pytest.raises(NeedleUnavailable, match="embed failed"):
        NeedleEmbedder(weights).encode(["anything"])


def test_an_empty_vector_from_the_model_raises(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Empty(StubAgent):
        def embed(self, text: str = "") -> list[float]:
            return []

    module = types.ModuleType("needle")
    module.Needle = Empty  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    with pytest.raises(NeedleUnavailable, match="empty vector"):
        NeedleEmbedder(weights).encode(["anything"])


def test_a_raising_adapter_costs_the_vector_half_and_nothing_more(
    no_weights: None,
) -> None:
    """The contract with VectorReranker.build, asserted from this side."""
    from cl_ai.ir import Tool
    from cl_ai.retrieval.vectors import VectorReranker

    reranker = VectorReranker(embedder=NeedleEmbedder(None))
    reranker.build((Tool(name="ls", binary="ls", description="list"),))
    assert reranker.available() is False


def test_close_is_idempotent_and_survives_a_never_loaded_model(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    embedder = NeedleEmbedder(weights)
    embedder.close()
    embedder.encode(["a"])
    agent = embedder._agent
    assert isinstance(agent, StubAgent)
    embedder.close()
    assert agent.closed
    embedder.close()


def test_a_close_that_raises_is_swallowed(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadClose(StubAgent):
        def close(self) -> None:
            raise OSError("handle already gone")

    module = types.ModuleType("needle")
    module.Needle = BadClose  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "needle", module)
    embedder = NeedleEmbedder(weights)
    embedder.encode(["a"])
    embedder.close()


def test_concurrent_first_use_loads_exactly_one_model(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    """The daemon serves keystrokes from a thread pool; two arriving during
    startup must not each pay 0.6s to build their own model."""
    import threading

    embedder = NeedleEmbedder(weights)
    seen: list[object] = []
    barrier = threading.Barrier(8)

    def run() -> None:
        barrier.wait()
        embedder.encode(["a"])
        seen.append(embedder._agent)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({id(s) for s in seen}) == 1


# -- weight discovery -----------------------------------------------------


def test_the_weights_path_is_overridable(
    weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CL_AI_NEEDLE_WEIGHTS", str(weights))
    assert default_weights() == weights


def test_an_override_pointing_at_nothing_yields_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly configured and wrong should not silently fall back to some
    other model on the machine."""
    monkeypatch.setenv("CL_AI_NEEDLE_WEIGHTS", str(tmp_path / "gone.cact"))
    assert default_weights() is None


def test_an_override_pointing_at_a_directory_yields_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CL_AI_NEEDLE_WEIGHTS", str(tmp_path))
    assert default_weights() is None


def test_weights_are_found_in_a_checkpoints_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CL_AI_NEEDLE_WEIGHTS", raising=False)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "needle3.cact").write_bytes(b"w")
    monkeypatch.chdir(tmp_path)
    found = default_weights()
    assert found is not None and found.name == "needle3.cact"


def test_weight_discovery_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CL_AI_NEEDLE_WEIGHTS", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(
        Path, "home", staticmethod(lambda: (_ for _ in ()).throw(RuntimeError()))
    )
    default_weights()


# -- the wiring helper ----------------------------------------------------


def test_build_embedder_returns_none_with_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Having no embedder is an ordinary configuration, not an error: the
    hybrid's lexical half stands alone."""
    monkeypatch.setenv("CL_AI_EMBEDDER", "needle")
    monkeypatch.setenv("CL_AI_NEEDLE_WEIGHTS", str(tmp_path / "gone.cact"))
    assert build_embedder() is None


def test_build_embedder_is_opt_in(
    stub_needle: types.ModuleType, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unconfigured means NO embedder, even with a loadable model present.

    The only backend implemented is Needle, and Needle measurably makes
    retrieval worse -- hit@1 0.704 -> 0.444. A default that silently wired it
    in would be a component whose own docstring says not to use it.
    """
    monkeypatch.delenv("CL_AI_EMBEDDER", raising=False)
    assert build_embedder(weights) is None


@pytest.mark.parametrize("value", ["", "  ", "none", "bge", "openai"])
def test_an_unrecognised_backend_yields_none(
    stub_needle: types.ModuleType,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("CL_AI_EMBEDDER", value)
    assert build_embedder(weights) is None


def test_build_embedder_wraps_in_a_cache_when_asked_for(
    stub_needle: types.ModuleType, weights: Path, tmp_path: Path
) -> None:
    from cl_ai.embedding.cache import CachingEmbedder

    built = build_embedder(weights, path=tmp_path / "v.bin", backend="needle")
    assert isinstance(built, CachingEmbedder)
    assert isinstance(built, Embedder)


def test_build_embedder_can_skip_the_cache(
    stub_needle: types.ModuleType, weights: Path
) -> None:
    built = build_embedder(weights, cache=False, backend="needle")
    assert isinstance(built, NeedleEmbedder)


def test_an_explicit_backend_overrides_the_environment(
    stub_needle: types.ModuleType, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CL_AI_EMBEDDER", "")
    assert build_embedder(weights, backend="needle") is not None


def test_the_cache_key_tracks_the_weights_file(
    stub_needle: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swapping in a finetune must not reuse the base model's vectors."""
    monkeypatch.delenv("CL_AI_VECTORS", raising=False)
    weights = tmp_path / "needle3.cact"
    weights.write_bytes(b"x" * 10)
    first = build_embedder(weights, path=tmp_path / "a.bin", backend="needle")
    weights.write_bytes(b"x" * 20)
    second = build_embedder(weights, path=tmp_path / "b.bin", backend="needle")
    assert first is not None and second is not None
    assert first.cache.key != second.cache.key  # type: ignore[union-attr]


# -- integration: the real model ------------------------------------------

_REAL = default_weights()
requires_needle = pytest.mark.skipif(
    _REAL is None, reason="no Needle weights; set CL_AI_NEEDLE_WEIGHTS"
)


@requires_needle
def test_the_real_model_produces_unit_vectors() -> None:
    embedder = NeedleEmbedder(_REAL)
    if not embedder.available():
        pytest.skip("needle is not importable here")
    try:
        vectors = embedder.encode(["list files", "commit changes"])
    except NeedleUnavailable as exc:
        pytest.skip(f"needle unusable: {exc}")
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) > 256
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, abs_tol=1e-5)


@requires_needle
def test_the_four_document_probe_that_misled_me_still_passes() -> None:
    """Kept deliberately, as the counterexample it turned out to be.

    This is the evidence on which I recommended building the vector half:
    `mkdir` leads for "make a directory", which lexical retrieval cannot do at
    all. It is TRUE and it is WORTHLESS. Four candidates means 25% by chance,
    the spread is 0.05, and the example was hand-picked. At corpus scale the
    same model scores hit@1 0.037 -- see the test below.

    It stays green so that anyone who repeats this probe, sees it pass, and
    concludes the model works can find the refutation immediately underneath.
    """
    embedder = NeedleEmbedder(_REAL)
    if not embedder.available():
        pytest.skip("needle is not importable here")
    corpus = [
        "mkdir. create directories",
        "cmake. cross-platform build system generator",
        "curl. transfer data from a url",
        "rm. remove files and directories",
    ]
    try:
        embedder.fit(corpus)
        docs = embedder.encode(corpus)
        query = embedder.encode(["make a directory"])[0]
    except NeedleUnavailable as exc:
        pytest.skip(f"needle unusable: {exc}")

    scores = [sum(q * d for q, d in zip(query, doc)) for doc in docs]
    assert scores[0] == max(scores), dict(zip(corpus, scores))


@requires_needle
def test_the_model_discriminates_only_when_distractors_are_far_apart() -> None:
    """Why the four-document probe above passes and the real corpus does not.

    Fifty documents, but the distractors are deliberately unrelated -- fonts,
    certificates, packets. Needle gets all three right here. That is the
    honest shape of its ability: it separates distant things and cannot
    separate near ones, and a catalog of CLI tools is nothing BUT near ones.
    Every tldr page is one imperative sentence about files, processes or the
    network, so the distinctions retrieval needs are exactly the distinctions
    this model does not make. Measured on the real corpus: hit@1 0.037, with
    `tty`, `exec` and `sleep` topping unrelated queries.

    Written because I first asserted "no better than chance" and this test
    failed 3/3 against correct code. "No signal" was wrong; "no signal where
    it matters" is the finding, and it is a sharper reason not to ship it.
    """
    embedder = NeedleEmbedder(_REAL)
    if not embedder.available():
        pytest.skip("needle is not importable here")

    answers = {
        "make a directory": "mkdir. create directories",
        "download a file from a url": "curl. transfer data from a url",
        "show running processes": "ps. report process status",
    }
    distractors = [
        f"tool{i}. {verb} {noun}"
        for i, (verb, noun) in enumerate(
            (v, n)
            for v in ("render", "convert", "inspect", "serve", "sign", "tune", "pack")
            for n in ("audio", "fonts", "certificates", "packets", "images", "logs")
        )
    ]
    corpus = list(answers.values()) + distractors[:47]
    try:
        embedder.fit(corpus)
        docs = embedder.encode(corpus)
        queries = {q: embedder.encode([q])[0] for q in answers}
    except NeedleUnavailable as exc:
        pytest.skip(f"needle unusable: {exc}")

    for query, wanted in answers.items():
        vector = queries[query]
        best = max(
            range(len(docs)),
            key=lambda j: sum(q * d for q, d in zip(vector, docs[j])),
        )
        assert corpus[best] == wanted, f"{query!r} -> {corpus[best]!r}"
