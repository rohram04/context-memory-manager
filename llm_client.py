"""Anthropic-SDK client factory — routes through OpenRouter when configured.

The codebase calls Claude via the Anthropic SDK (`client.messages.create`).
OpenRouter exposes an Anthropic-compatible endpoint at https://openrouter.ai/api
(the SDK appends /v1/messages), so we can keep all existing call sites and just
point the client at OpenRouter when an OpenRouter key is present.

Env vars (checked in order for the manager's key):
  OPENROUTER_API_KEY_MANAGER  — key for our memory manager's calls
  OPENROUTER_API_KEY          — shared OpenRouter key (also used by the Letta server)
  ANTHROPIC_API_KEY           — native Anthropic fallback (no OpenRouter routing)
"""

from __future__ import annotations

import os

import anthropic

OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# OpenRouter model ids (Anthropic-compatible skin). Verified available 2026-06.
OPENROUTER_SONNET = "anthropic/claude-sonnet-4.6"
OPENROUTER_HAIKU = "anthropic/claude-haiku-4.5"
# Letta uses provider-prefixed handles.
LETTA_OPENROUTER_SONNET = "openrouter/anthropic/claude-sonnet-4.6"


def _openrouter_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY_MANAGER") or os.environ.get("OPENROUTER_API_KEY")


def make_anthropic_client() -> anthropic.Anthropic:
    """Anthropic client routed through OpenRouter if a key is set, else native."""
    or_key = _openrouter_key()
    if or_key:
        return anthropic.Anthropic(api_key=or_key, base_url=OPENROUTER_BASE_URL)
    return anthropic.Anthropic()  # native Anthropic via ANTHROPIC_API_KEY


def has_llm_key() -> bool:
    """True if any usable key is configured (OpenRouter or native Anthropic)."""
    return bool(_openrouter_key() or os.environ.get("ANTHROPIC_API_KEY"))
