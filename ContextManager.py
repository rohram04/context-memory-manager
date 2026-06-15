from __future__ import annotations

from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.block import CacheBlock, LongTermBlock
from memory.longterm import LongTermStore
from memory.store import ContextStore


class ContextManager:
    """Primitive memory operations. Exposed to both the algorithmic layer and LLM tools."""

    FIDELITY_REMOVAL_THRESHOLD: ClassVar[float] = 0.50

    def __init__(
        self,
        store: ContextStore,
        lt_store: LongTermStore,
        embedding_model: str | SentenceTransformer = "all-MiniLM-L6-v2",
    ) -> None:
        self._store = store
        self._lt = lt_store
        self._embedder = (
            embedding_model
            if isinstance(embedding_model, SentenceTransformer)
            else SentenceTransformer(embedding_model)
        )

    def embed(self, text: str) -> list[float]:
        """Embed text via the owned sentence-transformer model."""
        return self._embedder.encode(text, convert_to_numpy=True).tolist()

    # ------------------------------------------------------------------
    # Store passthroughs
    # ------------------------------------------------------------------

    @property
    def budget_pressure(self) -> str:
        return self._store.budget_pressure

    @property
    def budget_fraction(self) -> float:
        return self._store.budget_fraction

    def next_candidate(self) -> CacheBlock | None:
        return self._store.next_candidate()

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(self, block: CacheBlock) -> None:
        """Add a block to context. Budget management is the caller's responsibility."""
        self._store.add(block)

    def augment(
        self,
        block_id: str,
        content: str,
        embedding: list[float],
    ) -> CacheBlock | None:
        """Replace block content + embedding. Resets fidelity to 1.0."""
        block = self._store.get(block_id)
        if block is None:
            return None
        block.content = content
        block.original_embedding = embedding
        block.current_embedding = embedding
        block.fidelity = 1.0
        self._store.update_priority(block.id)
        return block

    def find_similar(
        self,
        embedding: list[float],
        threshold: float,
    ) -> CacheBlock | None:
        """Return the most similar block above threshold, or None."""
        return self._store.find_similar(embedding, threshold)

    # ------------------------------------------------------------------
    # Primitives — callable by the algorithmic layer or LLM tools
    # ------------------------------------------------------------------

    def compress(
        self,
        block_id: str,
        compressed_content: str,
    ) -> CacheBlock | None:
        """Compress a block by replacing its content with a pre-computed summary.

        The caller is responsible for producing compressed_content (the algorithmic
        layer calls its compress_fn; the LLM writes it inline via the compress tool).
        Fidelity is computed here via cosine_sim(embed(compressed_content), block.original_embedding).
        On first compression, the current content is copied to long-term store.
        """
        block = self._store.get(block_id)
        if block is None:
            return None

        if block.fidelity < self.FIDELITY_REMOVAL_THRESHOLD:
            self._store.remove(block.id)
            return block

        if block.pointer_to_lt_id is None:
            lt_block = LongTermBlock(
                content=block.content,
                original_embedding=block.current_embedding,
                fidelity=block.fidelity,
                compression_count=block.compression_count,
                novelty_score=block.novelty_score,
                access_count=block.access_count,
                last_accessed=block.last_accessed,
                referenced_by=[block.id],
            )
            self._lt.add(lt_block)
            block.pointer_to_lt_id = lt_block.id

        new_embedding = self.embed(compressed_content)
        new_emb = np.array(new_embedding, dtype=np.float32)
        orig_emb = np.array(block.original_embedding, dtype=np.float32)
        orig_norm = float(np.linalg.norm(orig_emb))
        new_norm = float(np.linalg.norm(new_emb))
        new_fidelity = (
            float(np.dot(new_emb, orig_emb) / (new_norm * orig_norm))
            if orig_norm > 0 and new_norm > 0
            else 0.0
        )

        block.content = compressed_content
        block.current_embedding = new_embedding
        block.fidelity = new_fidelity
        block.compression_count += 1

        if block.fidelity < self.FIDELITY_REMOVAL_THRESHOLD:
            self._store.remove(block.id)
        else:
            self._store.update_priority(block.id)

        return block

    def promote(self, lt_block_id: str) -> CacheBlock | None:
        """Promote a long-term block back into context.

        Updates an existing stub in place, or creates a new block if the block
        was fully evicted. Budget management is the caller's responsibility.
        """
        lt_block = self._lt.get(lt_block_id)
        if lt_block is None:
            return None

        stub = next(
            (b for b in self._store.all_blocks() if b.pointer_to_lt_id == lt_block_id),
            None,
        )

        if stub is not None:
            stub.content = lt_block.content
            stub.fidelity = lt_block.fidelity
            stub.compression_count = lt_block.compression_count
            self._store.update_priority(stub.id)
            self._lt.access(lt_block_id)
            return stub

        block = CacheBlock(
            content=lt_block.content,
            original_embedding=lt_block.original_embedding,
            fidelity=lt_block.fidelity,
            compression_count=lt_block.compression_count,
            pointer_to_lt_id=lt_block_id,
            novelty_score=lt_block.novelty_score,
        )
        self.insert(block)
        self._lt.access(lt_block_id)
        return block

    def evict(self, block_id: str) -> CacheBlock | None:
        """Evict a block to LT and remove it from context.

        If no LT entry exists yet, creates one before removing.
        """
        block = self._store.get(block_id)
        if block is None:
            return None

        if block.pointer_to_lt_id is None:
            lt_block = LongTermBlock(
                content=block.content,
                original_embedding=block.current_embedding,
                fidelity=block.fidelity,
                compression_count=block.compression_count,
                novelty_score=block.novelty_score,
                access_count=block.access_count,
                last_accessed=block.last_accessed,
            )
            self._lt.add(lt_block)
            block.pointer_to_lt_id = lt_block.id

        self._store.remove(block.id)
        return block

    def update_novelty(self, block_id: str, novelty: float) -> CacheBlock | None:
        """Set a block's novelty score, clamped to [0.0, 1.0]."""
        block = self._store.get(block_id)
        if block is None:
            return None
        block.novelty_score = max(0.0, min(1.0, novelty))
        self._store.update_priority(block.id)
        return block

    # ------------------------------------------------------------------
    # System prompt rendering
    # ------------------------------------------------------------------

    def build_memory_prompt(self) -> str:
        store = self._store
        lines = [
            "=== MEMORY STATUS ===",
            f"Token budget: {store.used_tokens} / {store.max_tokens} "
            f"({store.budget_fraction:.0%} full)",
            f"Context blocks: {len(store)} total",
            f"Long-term blocks: {len(self._lt)}",
            f"Budget pressure: {store.budget_pressure}",
            "",
            "[MEMORY BLOCKS]",
        ]
        for block in store.all_blocks():
            tier = "stub" if block.pointer_to_lt_id else "full"
            lines.append(
                f"{block.id} | tier: {tier} | novelty: {block.novelty_score:.2f}"
                f" | access: {block.access_count} | decay: {block.decay_score:.2f}"
            )
            lines.append(f'  "{block.content}"')
        lines.append("===================")
        return "\n".join(lines)
