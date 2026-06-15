from __future__ import annotations

import heapq
from typing import ClassVar

import numpy as np

from memory.block import CacheBlock


def _priority(block: CacheBlock) -> float:
    novelty_resistance = 1.0 - block.novelty_score
    recency_resistance = 1.0 - block.decay_score
    compressibility = 0.6 * novelty_resistance + 0.4 * recency_resistance
    return block.token_cost * compressibility


class ContextStore:
    PRESSURE_THRESHOLDS: ClassVar[dict[str, float]] = {
        "critical": 0.90,
        "high": 0.75,
        "medium": 0.50,
        "low": 0.0,
    }

    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens
        self._blocks: dict[str, CacheBlock] = {}
        self._heap: list[tuple[float, str]] = []  # (-priority, block_id)

    def _rebuild(self) -> None:
        self._heap = [(-_priority(b), b.id) for b in self._blocks.values()]
        heapq.heapify(self._heap)

    # ------------------------------------------------------------------
    # Block registry
    # ------------------------------------------------------------------

    def add(self, block: CacheBlock) -> None:
        self._blocks[block.id] = block
        heapq.heappush(self._heap, (-_priority(block), block.id))

    def remove(self, block_id: str) -> CacheBlock | None:
        block = self._blocks.pop(block_id, None)
        if block is not None:
            self._rebuild()
        return block

    def get(self, block_id: str) -> CacheBlock | None:
        return self._blocks.get(block_id)

    def access(self, block_id: str) -> CacheBlock | None:
        """Mark a block as read by the LLM: updates last_accessed and re-sorts heap."""
        block = self._blocks.get(block_id)
        if block is not None:
            block.record_access()
            self._rebuild()
        return block

    def update_priority(self, block_id: str) -> None:
        """Re-sort heap after a block's content has changed (compression or expansion)."""
        if block_id in self._blocks:
            self._rebuild()

    def __contains__(self, block_id: str) -> bool:
        return block_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def all_blocks(self) -> list[CacheBlock]:
        return list(self._blocks.values())

    # ------------------------------------------------------------------
    # Token budget
    # ------------------------------------------------------------------

    @property
    def used_tokens(self) -> int:
        return sum(b.token_cost for b in self._blocks.values())

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def budget_fraction(self) -> float:
        return self.used_tokens / self._max_tokens if self._max_tokens > 0 else 0.0

    @property
    def budget_pressure(self) -> str:
        f = self.budget_fraction
        for level in ("critical", "high", "medium", "low"):
            if f >= self.PRESSURE_THRESHOLDS[level]:
                return level
        return "low"

    # ------------------------------------------------------------------
    # Compression heap
    # ------------------------------------------------------------------

    def next_candidate(self) -> CacheBlock | None:
        while self._heap:
            _, block_id = self._heap[0]
            if block_id in self._blocks:
                return self._blocks[block_id]
            heapq.heappop(self._heap)
        return None

    # ------------------------------------------------------------------
    # Similarity search (augmentation candidate)
    # ------------------------------------------------------------------

    def find_similar(
        self, query_embedding: list[float], threshold: float = 0.85
    ) -> CacheBlock | None:
        if not query_embedding:
            return None
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return None
        best_block: CacheBlock | None = None
        best_sim = -1.0
        for block in self._blocks.values():
            if not block.original_embedding:
                continue
            v = np.array(block.original_embedding, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            if sim > best_sim:
                best_sim = sim
                best_block = block
        return best_block if best_sim >= threshold else None
