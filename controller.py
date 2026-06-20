from __future__ import annotations

from typing import Callable

from ContextManager import ContextManager
from memory.block import CacheBlock
from memory.clock import CLOCK
from memory.config import MemoryConfig


class MemoryController:
    """Algorithmic scheduling layer built on top of ContextManager primitives."""

    def __init__(
        self,
        context_manager: ContextManager,
        compress_fn: Callable[[str], str],
        merge_fn: Callable[[str, str], tuple[str, list[float]]],
        config: MemoryConfig | None = None,
        augment_decision_fn: Callable[[str, list[tuple[str, str]]], str | None] | None = None,
    ) -> None:
        self._cm = context_manager
        self._compress_fn = compress_fn
        self._merge_fn = merge_fn
        self._augment_decision_fn = augment_decision_fn
        self._config = config or context_manager.config
        # Configure the shared decay clock for this run: wall-clock by default,
        # or a deterministic per-turn simulated clock when clock_seconds_per_turn > 0.
        CLOCK.reset(self._config.clock_seconds_per_turn)

    # ------------------------------------------------------------------
    # Algorithmic scheduling
    # ------------------------------------------------------------------

    def compression_tick(self) -> CacheBlock | None:
        """Pick the highest-priority candidate and compress it.

        An empty compress_fn result is the model's 'cannot_compress' signal: evict the
        block to LT instead of replacing its content with nothing.
        """
        block = self._cm.next_candidate()
        if block is None:
            return None
        compressed = self._compress_fn(block.content)
        if not compressed.strip():
            return self._cm.evict(block.id)
        return self._cm.compress(block.id, compressed)

    def fit_budget(self) -> None:
        """Compress candidates until context is within token budget.

        Each iteration must make progress (shrink or drop a block) or the loop cannot
        terminate: next_candidate() always returns a block while the store is non-empty,
        and a faithful summary of an already-short block neither shrinks it nor drops its
        fidelity below the removal threshold. So we evict the candidate to LT when either
        the model signals cannot_compress (empty result) OR the compression pass did not
        reduce used_tokens. Evicted blocks stay retrievable via the long-term store.
        """
        while self._cm.budget_fraction > 1.0:
            candidate = self._cm.next_candidate()
            if candidate is None:
                break
            before = self._cm.used_tokens
            compressed = self._compress_fn(candidate.content)
            if compressed.strip():
                self._cm.compress(candidate.id, compressed)
            if candidate.id in self._cm._store and self._cm.used_tokens >= before:
                self._cm.evict(candidate.id)

    def pre_prompt_promote(
        self,
        query_vec: list[float],
        top_k: int | None = None,
    ) -> list[CacheBlock]:
        """Similarity-search LT and promote high-scoring blocks into context."""
        cfg = self._config
        k = cfg.promote_top_k if top_k is None else top_k
        promoted = []
        for lt_block, similarity in self._cm._lt.similarity_search(query_vec, k):
            score = cfg.promote_alpha * similarity + cfg.promote_beta * lt_block.decay_score
            if score >= cfg.promote_threshold:
                block = self._cm.promote(lt_block.id)
                if block is not None:
                    promoted.append(block)
        return promoted

    def insert_or_augment(
        self,
        content: str,
        embedding: list[float],
        novelty_score: float,
        similarity_threshold: float | None = None,
    ) -> CacheBlock:
        """Merge into a similar block, or insert as a new block."""
        cfg = self._config
        if cfg.llm_augment_decision and self._augment_decision_fn is not None:
            candidates = self._cm.find_top_k(
                embedding, cfg.augment_candidate_top_k, cfg.augment_candidate_floor
            )
            if candidates:
                by_id = {block.id: block for block, _ in candidates}
                chosen_id = self._augment_decision_fn(
                    content, [(block.id, block.content) for block, _ in candidates]
                )
                existing = by_id.get(chosen_id) if chosen_id else None
                if existing is not None:
                    return self._augment_into(existing, content)
            return self._insert_new(content, embedding, novelty_score)

        threshold = (
            cfg.augment_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        existing = self._cm.find_similar(embedding, threshold)
        if existing is not None:
            return self._augment_into(existing, content)
        return self._insert_new(content, embedding, novelty_score)

    def _augment_into(self, existing: CacheBlock, content: str) -> CacheBlock:
        merged_content, merged_embedding = self._merge_fn(existing.content, content)
        return self._cm.augment(existing.id, merged_content, merged_embedding)

    def _insert_new(
        self, content: str, embedding: list[float], novelty_score: float
    ) -> CacheBlock:
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
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> CacheBlock:
        """Promote relevant LT blocks, insert or augment, then fit budget.

        Caller is responsible for computing `embedding` once and passing it in
        (it's also used to score novelty upstream, so we don't re-embed here).
        Advances the logical clock by one turn (no-op under wall-clock mode).
        """
        CLOCK.tick()
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
