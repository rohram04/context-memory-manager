"""LLM client factory — re-exports from llm.factory for backward compatibility."""

from llm.factory import (
    ANTHROPIC_HAIKU,
    ANTHROPIC_SONNET,
    LETTA_OPENROUTER_SONNET,
    OPENROUTER_HAIKU,
    OPENROUTER_SONNET,
    default_models,
    has_llm_key,
    make_llm_backend,
)
from llm.openrouter_tools import OPENROUTER_BASE_URL

__all__ = [
    "ANTHROPIC_HAIKU",
    "ANTHROPIC_SONNET",
    "LETTA_OPENROUTER_SONNET",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_HAIKU",
    "OPENROUTER_SONNET",
    "default_models",
    "has_llm_key",
    "make_llm_backend",
]
