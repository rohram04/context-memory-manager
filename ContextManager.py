from __future__ import annotations

from typing import Callable

import numpy as np

from memory.block import CacheBlock, LongTermBlock
from memory.config import MemoryConfig
from memory.embeddings import Embedder, make_embedder
from memory.longterm import LongTermStore
from memory.store import ContextStore


def _concat_union(a: str, b: str) -> str:
    """Lossless default union: join two texts, deduplicating exact-duplicate
    lines while dropping nothing. Keeps the free eval path and tests LLM-free.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for text in (a, b):
        for line in text.splitlines():
            key = line.strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            lines.append(line)
    return "\n".join(lines)


class ContextManager:
    """Primitive memory operations. Exposed to both the algorithmic layer and LLM tools."""

    def __init__(
        self,
        store: ContextStore,
        lt_store: LongTermStore,
        embedding_model: str | Embedder = "all-MiniLM-L6-v2",
        config: MemoryConfig | None = None,
        union_merge_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self._store = store
        self._lt = lt_store
        # Default to the store's config so both layers share one source of truth.
        self._config = config or store.config
        # Accept any embedder instance (SentenceTransformer, OpenAIEmbedder, ...)
        # or a string spec resolved by make_embedder (local ST name, or an OpenAI
        # model like "text-embedding-3-small" / "openai:<model>").
        self._embedder = make_embedder(embedding_model)
        # Lossless union used to reconcile a divergent context block back into LT.
        # Default keeps tests / the free eval path LLM-free and never loses info.
        self._union_merge_fn = union_merge_fn or _concat_union

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def embed(self, text: str) -> list[float]:
        """Embed text via the owned sentence-transformer model."""
        return self._embedder.encode(text, convert_to_numpy=True).tolist()

    # ------------------------------------------------------------------
    # Store passthroughs
    # ------------------------------------------------------------------

    @property
    def used_tokens(self) -> int:
        return self._store.used_tokens

    @property
    def max_tokens(self) -> int:
        return self._store.max_tokens

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

    def reconcile_to_lt(self, block: CacheBlock, *, refresh_block: bool) -> None:
        """Lossless-union a divergent (dirty) context block back into its LT copy.

        Invariant: LT holds the highest-fidelity (uncompressed union) copy of any
        pointered block. Called lazily at the moments a block would otherwise lose
        the info gained by augment: compress, evict, and (re-)promote.

        No-op when the block has no LT copy or is already clean. When
        ``refresh_block`` is True (promote path) the union is also written back into
        the in-context block; otherwise only LT is updated (compress/evict paths).
        """
        if block.pointer_to_lt_id is None or not block.dirty:
            return
        lt = self._lt.get(block.pointer_to_lt_id)
        if lt is None:
            block.dirty = False
            return
        union = self._union_merge_fn(lt.content, block.content)
        emb = self.embed(union)
        lt.content = union
        lt.original_embedding = emb
        lt.fidelity = 1.0
        lt.compression_count = 0
        self._lt.update(lt)
        if refresh_block:
            block.content = union
            block.current_embedding = emb
            block.original_embedding = emb
            block.fidelity = 1.0
        block.dirty = False

    def augment(
        self,
        block_id: str,
        content: str,
        embedding: list[float],
    ) -> CacheBlock | None:
        """Replace block content + embedding. Resets fidelity to 1.0.

        Marks the block dirty: the merged content has gained info not yet in LT.
        The expensive lossless reconcile is deferred until the block is compressed,
        evicted, or re-promoted (see reconcile_to_lt).
        """
        block = self._store.get(block_id)
        if block is None:
            return None
        block.content = content
        block.original_embedding = embedding
        block.current_embedding = embedding
        block.fidelity = 1.0
        block.dirty = True
        # A merge is an access: refresh access_count + last_accessed so re-mentioned
        # facts gain recency/frequency weight (record_access also re-sorts the heap).
        block.record_access()
        return block

    def find_similar(
        self,
        embedding: list[float],
        threshold: float,
    ) -> CacheBlock | None:
        """Return the most similar block above threshold, or None."""
        return self._store.find_similar(embedding, threshold)

    def find_top_k(
        self,
        embedding: list[float],
        k: int = 3,
        min_sim: float = 0.0,
    ) -> list[tuple[CacheBlock, float]]:
        """Top-k similar in-context blocks (>= min_sim), highest first."""
        return self._store.find_top_k(embedding, k, min_sim)

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

        if block.fidelity < self._config.fidelity_removal_threshold:
            return self.evict(block_id)

        if block.pointer_to_lt_id is None:
            # First compression pass: copy the current (full, possibly augmented)
            # content to LT before the lossy summary degrades it. This already
            # captures any augment, so the block is now clean against LT.
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
            block.dirty = False
        elif block.dirty:
            # Already pointered and augmented since: reconcile the augmented fact
            # into LT at full fidelity BEFORE this lossy pass degrades it.
            self.reconcile_to_lt(block, refresh_block=False)

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

        if block.fidelity < self._config.fidelity_removal_threshold:
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
            if stub.dirty:
                # Stub gained info since it last synced to LT: union both directions
                # so neither the stub's nor LT's facts are lost, and expand in context.
                self.reconcile_to_lt(stub, refresh_block=True)
            else:
                stub.content = lt_block.content
                stub.fidelity = lt_block.fidelity
                stub.compression_count = lt_block.compression_count
                # Under the invariant the stub now mirrors LT's content; keep
                # current_embedding describing it.
                stub.current_embedding = lt_block.original_embedding
            self._store.access(stub.id)
            self._lt.access(lt_block_id)
            return stub

        block = CacheBlock(
            content=lt_block.content,
            original_embedding=lt_block.original_embedding,
            current_embedding=lt_block.original_embedding,
            fidelity=lt_block.fidelity,
            compression_count=lt_block.compression_count,
            pointer_to_lt_id=lt_block_id,
            novelty_score=lt_block.novelty_score,
        )
        self.insert(block)
        self._store.access(block.id)
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
        elif block.dirty:
            # Pointered but augmented since: union the gained info into LT before
            # removal so eviction never loses it. (Clean pointered: LT already authoritative.)
            self.reconcile_to_lt(block, refresh_block=False)

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
                f" | fidelity: {block.fidelity:.2f} | tokens: {block.token_cost}"
                f" | access: {block.access_count} | decay: {block.decay_score:.2f}"
            )
            lines.append(f'  "{block.content}"')
        lines.append("===================")
        return "\n".join(lines)
