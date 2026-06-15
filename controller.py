from __future__ import annotations

from typing import Callable, ClassVar

from ContextManager import ContextManager
from memory.block import CacheBlock


class MemoryController:
    """Algorithmic scheduling layer built on top of ContextManager primitives."""

    PROMOTE_ALPHA: ClassVar[float] = 0.7
    PROMOTE_BETA: ClassVar[float] = 0.3
    PROMOTE_THRESHOLD: ClassVar[float] = 0.6

    def __init__(
        self,
        context_manager: ContextManager,
        compress_fn: Callable[[str], str],
        merge_fn: Callable[[str, str], tuple[str, list[float]]],
    ) -> None:
        self._cm = context_manager
        self._compress_fn = compress_fn
        self._merge_fn = merge_fn

    # ------------------------------------------------------------------
    # Algorithmic scheduling
    # ------------------------------------------------------------------

    def compression_tick(self) -> CacheBlock | None:
        """Pick the highest-priority candidate and compress it."""
        block = self._cm.next_candidate()
        if block is None:
            return None
        compressed = self._compress_fn(block.content)
        return self._cm.compress(block.id, compressed)

    def fit_budget(self) -> None:
        """Compress candidates until context is within token budget."""
        while self._cm.budget_fraction > 1.0:
            compressed_block = self.compression_tick()
            if compressed_block is None:
                break

    def pre_prompt_promote(
        self,
        query_vec: list[float],
        top_k: int = 5,
    ) -> list[CacheBlock]:
        """Similarity-search LT and promote high-scoring blocks into context."""
        promoted = []
        for lt_block, similarity in self._cm._lt.similarity_search(query_vec, top_k):
            score = self.PROMOTE_ALPHA * similarity + self.PROMOTE_BETA * lt_block.decay_score
            if score >= self.PROMOTE_THRESHOLD:
                block = self._cm.promote(lt_block.id)
                if block is not None:
                    promoted.append(block)
        return promoted

    def insert_or_augment(
        self,
        content: str,
        embedding: list[float],
        novelty_score: float,
        similarity_threshold: float = 0.85,
    ) -> CacheBlock:
        """Find a similar block and merge into it, or insert as a new block."""
        existing = self._cm.find_similar(embedding, similarity_threshold)
        if existing is not None:
            merged_content, merged_embedding = self._merge_fn(existing.content, content)
            return self._cm.augment(existing.id, merged_content, merged_embedding)

        block = CacheBlock(
            content=content,
            original_embedding=embedding,
            novelty_score=novelty_score,
        )
        self._cm.insert(block)
        return block

    # ------------------------------------------------------------------
    # Pipeline entry point (algorithmic mode)
    # ------------------------------------------------------------------

    def receive(
        self,
        content: str,
        embedding: list[float],
        novelty_score: float,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> CacheBlock:
        """Promote relevant LT blocks, insert or augment, then fit budget.

        Caller is responsible for computing `embedding` once and passing it in
        (it's also used to score novelty upstream, so we don't re-embed here).
        """
        self.pre_prompt_promote(embedding, top_k)
        block = self.insert_or_augment(content, embedding, novelty_score, similarity_threshold)
        self.fit_budget()
        return block

    def embed(self, content: str) -> list[float]:
        return self._cm.embed(content)

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    def build_memory_prompt(self) -> str:
        return self._cm.build_memory_prompt()
