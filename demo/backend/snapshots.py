"""Snapshot + LT-delta capture for the demo server.

Everything here reads live state off a ``ContextManager`` and freezes it into
plain JSON-safe dicts. Critical: ``CacheBlock.token_cost`` and
``CacheBlock.decay_score`` (and ``LongTermBlock.decay_score``) are pydantic
``computed_field``s that recompute off the global CLOCK every access — so we read
each value exactly once here and store the scalar. We NEVER hold a reference to a
block or send embeddings to the client.
"""

from __future__ import annotations

from typing import Any

from ContextManager import ContextManager


def _block_view(block: Any) -> dict[str, Any]:
    """Freeze one in-context CacheBlock into a plain dict (no embeddings)."""
    tier = "stub" if block.pointer_to_lt_id else "full"
    return {
        "id": block.id,
        "content": block.content,
        "tier": tier,
        "novelty": block.novelty_score,
        "decay": block.decay_score,            # computed — captured now
        "fidelity": block.fidelity,
        "token_cost": block.token_cost,        # computed — captured now
        "compression_count": block.compression_count,
        "access_count": block.access_count,
        "pointer_to_lt_id": block.pointer_to_lt_id,
    }


def _lt_view(block: Any) -> dict[str, Any]:
    """Freeze one LongTermBlock into a plain dict (no embeddings)."""
    return {
        "id": block.id,
        "content": block.content,
        "novelty": block.novelty_score,
        "fidelity": block.fidelity,
        "compression_count": block.compression_count,
        "access_count": block.access_count,
        "decay": block.decay_score,            # computed — captured now
        "is_reconstructed": block.is_reconstructed,
    }


def build_context_snapshot(
    cm: ContextManager,
    *,
    index: int,
    role: str,
    text: str,
    latency_ms: float | None,
) -> dict[str, Any]:
    """Capture the full context-window state for one message."""
    store = cm._store
    return {
        "index": index,
        "role": role,
        "text": text,
        "budget_used": cm.used_tokens,
        "budget_max": store.max_tokens,
        "budget_pressure": cm.budget_pressure,
        "memory_prompt_text": cm.build_memory_prompt(),  # byte-faithful
        "blocks": [_block_view(b) for b in store.all_blocks()],
        "latency_ms": latency_ms,
    }


def _lt_state(cm: ContextManager) -> dict[str, dict[str, Any]]:
    """Map id -> frozen lt_view for the current LT store contents."""
    return {b.id: _lt_view(b) for b in cm._lt.all_blocks()}


def diff_lt(
    prev: dict[str, dict[str, Any]],
    curr: dict[str, dict[str, Any]],
) -> dict[str, list]:
    """Compute the LT delta between two frozen LT states.

    added    — ids in curr but not prev
    updated  — ids in both whose content/fidelity/compression_count changed
    accessed — ids in both whose access_count increased
    """
    added = [v for cid, v in curr.items() if cid not in prev]
    updated: list[dict[str, Any]] = []
    accessed: list[str] = []
    for cid, cv in curr.items():
        pv = prev.get(cid)
        if pv is None:
            continue
        if (
            cv["content"] != pv["content"]
            or cv["fidelity"] != pv["fidelity"]
            or cv["compression_count"] != pv["compression_count"]
        ):
            updated.append(cv)
        if cv["access_count"] > pv["access_count"]:
            accessed.append(cid)
    return {"added": added, "updated": updated, "accessed": accessed}
