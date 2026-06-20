from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from llm.interface import LLMBackend
from llm.types import StopReason

if TYPE_CHECKING:
    from ContextManager import ContextManager


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
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str], str]:
    """LLM-based summarization for algorithmic mode, via a structured tool call."""
    def compress(content: str) -> str:
        try:
            response = backend.complete(
                model=model,
                max_tokens=512,
                tools=_COMPRESS_TOOLS,
                tool_choice="required",
                messages=[{"role": "user", "content": (
                    "Summarize the following memory block more concisely than it "
                    "currently is, preserving all key facts, entities, and relationships. "
                    "Call provide_summary with the shortened text. Only call "
                    "cannot_compress if the block is already as compact as possible and "
                    "cannot be shortened further without losing essential meaning.\n\n"
                    + content
                )}],
            )
            for call in response.tool_calls:
                if call.name == "cannot_compress":
                    return ""
                if call.name == "provide_summary":
                    summary = call.arguments.get("summary", "")
                    return summary if summary.strip() else content
            return content
        except Exception:
            return content

    return compress


def make_merge_fn(
    backend: LLMBackend,
    cm: ContextManager,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, str], tuple[str, list[float]]]:
    """LLM-based merge for algorithmic mode."""
    def merge(existing: str, new: str) -> tuple[str, list[float]]:
        try:
            response = backend.complete(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": (
                    "Merge these two memory blocks into a single coherent block, "
                    "preserving all facts from both. Output only the merged result:\n\n"
                    f"BLOCK A:\n{existing}\n\nBLOCK B:\n{new}"
                )}],
            )
            merged = response.text or f"{existing}\n{new}"
        except Exception:
            merged = f"{existing}\n{new}"
        return (merged, cm.embed(merged))

    return merge


def make_union_merge_fn(
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, str], str]:
    """Lossless-union of two versions of the same memory (LT copy + context copy)."""
    def union(a: str, b: str) -> str:
        try:
            response = backend.complete(
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
            return response.text or f"{a}\n{b}"
        except Exception:
            return f"{a}\n{b}"

    return union
