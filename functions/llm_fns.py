from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import anthropic

if TYPE_CHECKING:
    from ContextManager import ContextManager


def _extract_text(content: list) -> str:
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return ""


def make_compress_fn(
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str], str]:
    """LLM-based summarization for algorithmic mode.

    Returns compressed text only. Fidelity is computed by ContextManager.compress
    after the call using the block's original_embedding.
    Falls back to the original content on API failure.
    """
    def compress(content: str) -> str:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": (
                    "Summarize the following concisely, preserving all key facts, "
                    "entities, and relationships. Output only the summary:\n\n" + content
                )}],
            )
            return _extract_text(response.content) or content
        except Exception:
            return content

    return compress


def make_merge_fn(
    client: anthropic.Anthropic,
    cm: ContextManager,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, str], tuple[str, list[float]]]:
    """LLM-based merge for algorithmic mode.

    Synthesizes two blocks into one coherent block and returns
    (merged_content, merged_embedding). Falls back to concatenation on failure.
    """
    def merge(existing: str, new: str) -> tuple[str, list[float]]:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": (
                    "Merge these two memory blocks into a single coherent block, "
                    "preserving all facts from both. Output only the merged result:\n\n"
                    f"BLOCK A:\n{existing}\n\nBLOCK B:\n{new}"
                )}],
            )
            merged = _extract_text(response.content) or f"{existing}\n{new}"
        except Exception:
            merged = f"{existing}\n{new}"
        return (merged, cm.embed(merged))

    return merge
