"""Unit tests for the embedder abstraction (no network / keys required)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from memory.embeddings import Embedder, OpenAIEmbedder, make_embedder


class _FakeEmbedder:
    """Minimal duck-typed embedder."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, text, convert_to_numpy: bool = True):
        vec = np.ones(self.dim, dtype=np.float32) * float(len(text))
        return vec if convert_to_numpy else vec.tolist()


def _install_fake_openai(monkeypatch, dim: int = 1536):
    """Install a fake `openai` module so OpenAIEmbedder needs no key/network."""
    captured: dict = {}

    class _Embeddings:
        def create(self, *, model, input, **kwargs):
            captured["model"] = model
            captured["input"] = input
            captured["kwargs"] = kwargs
            data = [
                types.SimpleNamespace(embedding=[float(i)] * dim) for i, _ in enumerate(input)
            ]
            return types.SimpleNamespace(data=data)

    class _OpenAI:
        def __init__(self, *, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.embeddings = _Embeddings()

    fake = types.ModuleType("openai")
    fake.OpenAI = _OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


def test_make_embedder_passes_through_instances():
    fake = _FakeEmbedder()
    assert make_embedder(fake) is fake
    assert isinstance(fake, Embedder)


def test_make_embedder_routes_openai_specs(monkeypatch):
    _install_fake_openai(monkeypatch)
    for spec in ("text-embedding-3-small", "openai:text-embedding-3-large", "openai/foo"):
        emb = make_embedder(spec)
        assert isinstance(emb, OpenAIEmbedder)
    assert make_embedder("openai:my-model").model == "my-model"
    assert make_embedder("text-embedding-3-small").model == "text-embedding-3-small"


def test_make_embedder_rejects_bad_spec():
    with pytest.raises(TypeError):
        make_embedder(123)  # not a str, no .encode


def test_openai_embedder_encode_single(monkeypatch):
    captured = _install_fake_openai(monkeypatch, dim=1536)
    emb = OpenAIEmbedder(api_key="sk-test")
    vec = emb.encode("hello", convert_to_numpy=True)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1536,)
    # SentenceTransformer-compatible: .tolist() works for ContextManager.embed
    assert isinstance(vec.tolist(), list)
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["hello"]  # single string wrapped in a list


def test_openai_embedder_batch_and_dimensions(monkeypatch):
    captured = _install_fake_openai(monkeypatch, dim=256)
    emb = OpenAIEmbedder(model="text-embedding-3-large", dimensions=256)
    arr = emb.encode(["a", "bb"], convert_to_numpy=True)
    assert arr.shape == (2, 256)
    assert captured["kwargs"].get("dimensions") == 256


def test_context_manager_accepts_duck_typed_embedder():
    from memory.config import MemoryConfig
    from memory.longterm import LongTermStore
    from memory.store import ContextStore
    from ContextManager import ContextManager

    cfg = MemoryConfig()
    store = ContextStore(max_tokens=1000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=4)
    cm = ContextManager(store, lt, embedding_model=_FakeEmbedder(dim=4), config=cfg)
    out = cm.embed("abc")
    assert out == [3.0, 3.0, 3.0, 3.0]  # list[float], len(text)=3 over dim 4
