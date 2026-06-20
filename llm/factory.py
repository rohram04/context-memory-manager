from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

from llm.anthropic_tools import AnthropicBackend
from llm.interface import LLMBackend
from llm.openrouter_tools import OPENROUTER_BASE_URL, OpenRouterBackend

_ENV_LOADED = False


def _load_dotenv() -> None:
    """Load repo-root ``.env`` into os.environ (setdefault — won't override)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)

# OpenRouter model ids — primary (conversation) and util (compress/merge/novelty).
OPENROUTER_SONNET = "openai/gpt-4o"
OPENROUTER_HAIKU = "openai/gpt-4o-mini"
LETTA_OPENROUTER_SONNET = "openrouter/anthropic/claude-sonnet-4.6"

# Native Anthropic model ids (dashed, no provider prefix)
ANTHROPIC_SONNET = "claude-sonnet-4-6"
ANTHROPIC_HAIKU = "claude-haiku-4-5-20251001"


def _openrouter_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY_MANAGER") or os.environ.get("OPENROUTER_API_KEY")


def default_models() -> tuple[str, str]:
    """Return (main_model, util_model) ids for the active routing."""
    if _openrouter_key():
        return OPENROUTER_SONNET, OPENROUTER_HAIKU
    return ANTHROPIC_SONNET, ANTHROPIC_HAIKU


def has_llm_key() -> bool:
    """True if any usable key is configured (OpenRouter or native Anthropic)."""
    return bool(_openrouter_key() or os.environ.get("ANTHROPIC_API_KEY"))


def _resolve_backend_name(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LLM_BACKEND", "").lower()
    if env in ("anthropic", "openrouter"):
        return env
    if _openrouter_key():
        return "openrouter"
    return "anthropic"


def make_llm_backend(
    *,
    backend: str | None = None,
    local: bool = False,
    local_base_url: str = "http://localhost:4000/v1",
    api_key: str | None = None,
) -> LLMBackend:
    """Create an LLM backend.

    backend: 'anthropic' | 'openrouter' | None (auto from env/keys)
    local: route to a local LiteLLM OpenAI-compatible proxy
    """
    _load_dotenv()
    name = _resolve_backend_name(backend)

    if local:
        key = api_key or "local"
        base = local_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        client = OpenAI(api_key=key, base_url=base)
        return OpenRouterBackend(client)

    if name == "openrouter":
        or_key = api_key or _openrouter_key()
        if not or_key:
            raise ValueError("OpenRouter backend requires OPENROUTER_API_KEY")
        client = OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL)
        return OpenRouterBackend(client)

    # anthropic Messages API backend
    import anthropic

    or_key = _openrouter_key()
    if backend == "anthropic" and or_key:
        client = anthropic.Anthropic(
            api_key=or_key,
            base_url="https://openrouter.ai/api",
        )
    else:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    return AnthropicBackend(client)
