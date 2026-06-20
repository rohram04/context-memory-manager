"""Embedder abstraction: local SentenceTransformers, OpenAI, or OpenRouter embeddings.

ContextManager only needs an object exposing
``encode(text, convert_to_numpy=True)`` (the SentenceTransformer surface it
already used). ``OpenAIEmbedder`` mirrors that surface so it drops in unchanged,
and ``make_embedder`` resolves a string spec to the right backend so every
existing ``--embedding-model`` flag transparently supports all three backends.

Spec syntax for ``make_embedder``::

    "all-MiniLM-L6-v2"              → local SentenceTransformer
    "text-embedding-3-small"         → OpenAI direct  (OPENAI_API_KEY)
    "openai:text-embedding-3-small"  → OpenAI direct  (OPENAI_API_KEY)
    "openrouter:openai/text-embedding-3-small"  → via OpenRouter (OPENROUTER_API_KEY)
    "openrouter:text-embedding-3-small"         → via OpenRouter (model name as-is)
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Matches the constant in llm/openrouter_tools.py — kept local to avoid a
# circular import from the llm layer into the memory layer.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@runtime_checkable
class Embedder(Protocol):
    """Anything ContextManager can embed with."""

    def encode(self, text: Any, convert_to_numpy: bool = True) -> Any: ...


class OpenAIEmbedder:
    """OpenAI-compatible embeddings with a SentenceTransformer-compatible ``encode``.

    Works with any OpenAI-compatible embeddings endpoint: api.openai.com directly
    (default) or OpenRouter (pass ``base_url=_OPENROUTER_BASE_URL``).

    Default model is ``text-embedding-3-small`` (1536-dim, matching LongTermStore's
    default ``embedding_dim``). ``dimensions`` optionally truncates the output
    (text-embedding-3-* support this natively).
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
_OPENROUTER_PREFIXES = ("openrouter:", "openrouter/")


def make_embedder(spec: str | Embedder) -> Embedder:
    """Resolve an embedder spec to an embedder instance.

    - ``"openrouter:<model>"`` / ``"openrouter/<model>"``
        → OpenAIEmbedder routed through OpenRouter (OPENROUTER_API_KEY)
    - ``"openai:<model>"`` / ``"openai/<model>"``
        → OpenAIEmbedder hitting api.openai.com (OPENAI_API_KEY)
    - a bare ``"text-embedding-*"`` name
        → OpenAIEmbedder hitting api.openai.com (OPENAI_API_KEY)
    - any other string
        → SentenceTransformer(<name>) (local, no key needed)
    - an object with ``.encode``
        → returned as-is (any embedder instance)

    Strings are resolved first: ``str`` itself has an ``.encode`` method, so the
    duck-typed branch must come after the string handling.
    """
    if isinstance(spec, str):
        low = spec.lower()

        for prefix in _OPENROUTER_PREFIXES:
            if low.startswith(prefix):
                model = spec[len(prefix):]
                return OpenAIEmbedder(
                    model=model,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url=_OPENROUTER_BASE_URL,
                )

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
