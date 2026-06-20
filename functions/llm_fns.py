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


_AUGMENT_DECISION_TOOLS = [
    {
        "name": "augment_into",
        "description": (
            "The new information is about the SAME specific subject as one of the "
            "existing blocks and should be merged into it. Provide that block's id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {
                    "type": "string",
                    "description": "id of the existing block to merge the new information into.",
                }
            },
            "required": ["block_id"],
        },
    },
    {
        "name": "insert_new",
        "description": (
            "The new information is a DISTINCT topic from every candidate block and "
            "should be stored as a brand-new block."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def make_augment_decision_fn(
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, list[tuple[str, str]]], str | None]:
    """LLM judge for algorithmic-mode augmentation."""
    def decide(incoming: str, candidates: list[tuple[str, str]]) -> str | None:
        if not candidates:
            return None
        valid_ids = {bid for bid, _ in candidates}
        listing = "\n\n".join(f"[{bid}]\n{content}" for bid, content in candidates)
        try:
            response = backend.complete(
                model=model,
                max_tokens=256,
                tools=_AUGMENT_DECISION_TOOLS,
                tool_choice="required",
                messages=[{"role": "user", "content": (
                    "A new piece of information just arrived. Decide whether it should be "
                    "MERGED into one of the existing memory blocks because it is about the "
                    "same specific subject (e.g. a follow-up question or update about the "
                    "same entity/topic), or stored as a NEW block because it is a distinct "
                    "topic. Only merge when they are genuinely about the same subject — "
                    "similar phrasing about different subjects must stay separate.\n\n"
                    f"NEW INFORMATION:\n{incoming}\n\n"
                    f"EXISTING BLOCKS:\n{listing}\n\n"
                    "Call augment_into with the matching block_id, or insert_new."
                )}],
            )
            for call in response.tool_calls:
                if call.name == "augment_into":
                    chosen = call.arguments.get("block_id", "")
                    return chosen if chosen in valid_ids else None
                if call.name == "insert_new":
                    return None
            return None
        except Exception:
            return None

    return decide


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
