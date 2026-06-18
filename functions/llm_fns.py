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


_COMPRESS_TOOLS = [
    {
        "name": "provide_summary",
        "description": (
            "Provide a more concise version of the memory block that preserves all "
            "key facts, entities, and relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "The shortened version of the block content.",
                }
            },
            "required": ["summary"],
        },
    },
    {
        "name": "cannot_compress",
        "description": (
            "Signal that the block is already as compact as possible and cannot be "
            "shortened further without losing essential meaning."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def make_compress_fn(
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str], str]:
    """LLM-based summarization for algorithmic mode, via a structured tool call.

    The model calls exactly one of two tools:
      - provide_summary(summary) -> returns the shortened text
      - cannot_compress()        -> returns "" (signals the caller to evict the block)

    The empty-string signal lets MemoryController.fit_budget evict blocks the model
    judges incompressible instead of re-summarizing them forever (the infinite-loop fix).
    Fidelity is computed by ContextManager.compress after the call using the block's
    original_embedding. Falls back to the original content on API failure (a transient
    error must not be read as "incompressible" — the controller's no-progress guard
    still guarantees termination).
    """
    def compress(content: str) -> str:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                tools=_COMPRESS_TOOLS,
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": (
                    "Summarize the following memory block more concisely than it "
                    "currently is, preserving all key facts, entities, and relationships. "
                    "Call provide_summary with the shortened text. Only call "
                    "cannot_compress if the block is already as compact as possible and "
                    "cannot be shortened further without losing essential meaning.\n\n"
                    + content
                )}],
            )
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    if block.name == "cannot_compress":
                        return ""
                    if block.name == "provide_summary":
                        summary = (block.input or {}).get("summary", "")
                        return summary if summary.strip() else content
            # No tool_use block (unexpected): leave content unchanged so the
            # no-progress guard evicts rather than loops.
            return content
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


def make_union_merge_fn(
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, str], str]:
    """Lossless-union of two versions of the same memory (LT copy + context copy).

    Distinct from make_merge_fn (which fuses two *different* blocks and returns an
    embedding): this reconciles divergent copies of one block and returns text only.
    Must run through the metered client so the reconcile cost is counted under "ours".
    Falls back to lossless concatenation on API failure.
    """
    def union(a: str, b: str) -> str:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": (
                    "You are combining two versions of the same memory. Produce a "
                    "single version that preserves EVERY distinct fact from both, "
                    "deduplicates overlapping information, and drops NOTHING. Make it "
                    "as compact as possible while remaining lossless. Output only the "
                    "combined version:\n\n"
                    f"VERSION A:\n{a}\n\nVERSION B:\n{b}"
                )}],
            )
            return _extract_text(response.content) or f"{a}\n{b}"
        except Exception:
            return f"{a}\n{b}"

    return union
