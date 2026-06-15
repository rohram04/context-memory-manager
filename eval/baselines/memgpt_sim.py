"""MemGPT-style FIFO baseline.

The minimum viable comparison point for the eviction-policy claim: keep a flat
list of `CacheBlock`s in insertion order, drop the oldest blocks first whenever
the token budget is exceeded, and answer queries by cosine top-1 over what's
left. No long-term store, no compression, no novelty scoring — those are
precisely the things our system adds on top.

Mirrors how MemGPT's main-context window evicts older summaries when new turns
push the budget over. We reuse `CacheBlock` so token accounting and embedding
shape stay identical to our system's path.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from memory.block import CacheBlock


class FIFOMemory:
    def __init__(self, max_tokens: int, embedder: SentenceTransformer) -> None:
        self.max_tokens = max_tokens
        self.embedder = embedder
        self.blocks: list[CacheBlock] = []

    def _embed(self, text: str) -> list[float]:
        return self.embedder.encode(text, convert_to_numpy=True).tolist()

    def ingest(self, content: str) -> None:
        block = CacheBlock(
            content=content,
            original_embedding=self._embed(content),
            novelty_score=0.5,
        )
        self.blocks.append(block)
        while self.blocks and sum(b.token_cost for b in self.blocks) > self.max_tokens:
            self.blocks.pop(0)

    def query(self, query_text: str) -> CacheBlock | None:
        if not self.blocks:
            return None
        q = np.array(self._embed(query_text), dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return None
        best: CacheBlock | None = None
        best_sim = -1.0
        for b in self.blocks:
            v = np.array(b.original_embedding, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            if sim > best_sim:
                best_sim, best = sim, b
        return best
