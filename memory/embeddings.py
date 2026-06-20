"""Embedder abstraction: local SentenceTransformers or OpenAI embeddings.

ContextManager only needs an object exposing
``encode(text, convert_to_numpy=True)`` (the SentenceTransformer surface it
already used). ``OpenAIEmbedder`` mirrors that surface so it drops in unchanged,
and ``make_embedder`` resolves a string spec to the right backend so every
existing ``--embedding-model`` flag transparently supports OpenAI models.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything ContextManager can embed with."""

    def encode(self, text: Any, convert_to_numpy: bool = True) -> Any: ...


class OpenAIEmbedder:
    """OpenAI embeddings with a SentenceTransformer-compatible ``encode``.

    Default model is ``text-embedding-3-small`` (1536-dim, matching
    LongTermStore's default ``embedding_dim``). Embeddings hit api.openai.com
    directly — OpenRouter does not serve an embeddings endpoint — with the key
    from ``OPENAI_API_KEY`` unless one is passed. ``dimensions`` optionally
    truncates the output (text-embedding-3-* support this natively).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._dimensions = dimensions
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def encode(self, text: Any, convert_to_numpy: bool = True) -> Any:
        """Embed a string (or iterable of strings). Returns a 1-D array for a
        single string and a 2-D array for a batch, matching SentenceTransformer.
        """
        single = isinstance(text, str)
        inputs = [text] if single else list(text)
        kwargs: dict[str, Any] = {"model": self.model, "input": inputs}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = self._client.embeddings.create(**kwargs)
        vecs = [d.embedding for d in resp.data]
        if convert_to_numpy:
            arr = np.array(vecs, dtype=np.float32)
            return arr[0] if single else arr
        return vecs[0] if single else vecs


_OPENAI_PREFIXES = ("openai:", "openai/")


def make_embedder(spec: str | Embedder) -> Embedder:
    """Resolve an embedder spec to an embedder instance.

    - ``"openai:<model>"`` / ``"openai/<model>"`` -> OpenAIEmbedder(model=<model>)
    - a bare ``"text-embedding-*"`` name     -> OpenAIEmbedder(model=<name>)
    - any other string                       -> SentenceTransformer(<name>) (local)
    - an object with ``.encode``             -> returned as-is (any embedder instance)

    Strings are resolved first: ``str`` itself has an ``.encode`` method, so the
    duck-typed branch must come after the string handling.
    """
    if isinstance(spec, str):
        low = spec.lower()
        for prefix in _OPENAI_PREFIXES:
            if low.startswith(prefix):
                return OpenAIEmbedder(model=spec[len(prefix):])
        if low.startswith("text-embedding"):
            return OpenAIEmbedder(model=spec)

        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(spec)
    if hasattr(spec, "encode"):
        return spec  # already an embedder instance
    raise TypeError(
        f"embedding spec must be a str or expose .encode, got {type(spec)!r}"
    )
