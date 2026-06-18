"""Shared machinery for building and driving the algorithmic memory system in evals.

Used by eval/recall_advantage.py (Part B) and eval/locomo_eval.py (Part A).
Supports two cost modes:
  - stub  (free): text-truncation compress + concat merge, embedding novelty, no API.
  - live  (paid): Haiku/Sonnet compress + merge via functions.llm_fns, metered.
Probing supports a free retrieval-token check and a paid LLM-answer probe.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import anthropic
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager  # noqa: E402
from controller import MemoryController  # noqa: E402
from eval.cost import CostMeter  # noqa: E402
from ContextManager import _concat_union  # noqa: E402
from functions.llm_fns import (  # noqa: E402
    _extract_text,
    make_compress_fn,
    make_merge_fn,
    make_union_merge_fn,
)
from memory.config import MemoryConfig  # noqa: E402
from memory.longterm import LongTermStore  # noqa: E402
from memory.novelty import NoveltyMode, build_novelty_fn  # noqa: E402
from memory.store import ContextStore  # noqa: E402

QUERY_SYSTEM_SUFFIX = (
    "\n\nAnswer the user's question using only information from MEMORY STATUS above. "
    "If the answer is not present, say you don't know. Reply with the fact only — no preamble."
)


def embedding_dim(embedder: SentenceTransformer) -> int:
    return int(embedder.get_sentence_embedding_dimension())


def _truncate_compress(content: str) -> str:
    """Free stub compression: halve by character count, 20-char floor."""
    if len(content) <= 20:
        return content
    return content[: max(20, len(content) // 2)]


def _make_concat_merge(cm: ContextManager) -> Callable[[str, str], tuple[str, list[float]]]:
    def merge(existing: str, new: str) -> tuple[str, list[float]]:
        merged = f"{existing} | {new}"
        return merged, cm.embed(merged)
    return merge


class OursSystem:
    """An instance of the algorithmic memory manager wired for evaluation."""

    def __init__(
        self,
        max_tokens: int,
        embedder: SentenceTransformer,
        config: MemoryConfig,
        *,
        cost_mode: str = "stub",          # "stub" (free) | "live" (paid)
        novelty_mode: NoveltyMode = NoveltyMode.EMBEDDING,
        client: anthropic.Anthropic | None = None,
        util_model: str = "claude-sonnet-4-6",
        meter: CostMeter | None = None,
        system_name: str = "ours",
    ) -> None:
        self.config = config
        self.client = client
        self.util_model = util_model
        self.meter = meter
        self.system_name = system_name

        dim = embedding_dim(embedder)
        store = ContextStore(max_tokens=max_tokens, config=config)
        self.lt = LongTermStore("sqlite:///:memory:", embedding_dim=dim)

        if cost_mode == "live":
            assert client is not None, "live cost_mode requires an anthropic client"
            # Meter EVERY ours call (compress, merge, union-reconcile, novelty-llm,
            # probe) so the reported cost is complete and --max-spend can bound the run.
            self.client = meter.wrap_client(client, system_name) if meter else client
            union_merge_fn = make_union_merge_fn(self.client, util_model)
        else:
            union_merge_fn = _concat_union

        self.cm = ContextManager(
            store,
            self.lt,
            embedding_model=embedder,
            config=config,
            union_merge_fn=union_merge_fn,
        )

        if cost_mode == "live":
            compress_fn = make_compress_fn(self.client, util_model)
            merge_fn = make_merge_fn(self.client, self.cm, util_model)
        else:
            compress_fn = _truncate_compress
            merge_fn = _make_concat_merge(self.cm)

        self.controller = MemoryController(self.cm, compress_fn, merge_fn, config)
        self.novelty_fn = build_novelty_fn(novelty_mode, self.cm, self.client, util_model)

    # -- ingest / probe ------------------------------------------------------
    def ingest(self, content: str) -> None:
        emb = self.cm.embed(content)
        novelty = self.novelty_fn(content, emb)
        self.controller.receive(content, emb, novelty)

    def retained_token(self, token: str) -> bool:
        """True if the token survives anywhere in memory (context ∪ long-term).

        Isolates the eviction-policy claim from retrieval ranking: 'did the system
        keep the fact at all', independent of whether top-1 retrieval surfaces it.
        """
        if any(token in b.content for b in self.cm._store.all_blocks()):
            return True
        return any(token in b.content for b in self.lt.all_blocks())

    def context_contains(self, substring: str) -> bool:
        """True if `substring` is present in any CONTEXT block (long-term excluded).

        Used by tuning as the in-context retention signal. Pass a fact's anchor
        (start-of-content, e.g. 'Project Cobalt Anvil'), which survives stub
        truncation — unlike the opaque token, which sits at the end and is dropped
        by character-truncation even when the block itself is still in context.
        """
        return any(substring in b.content for b in self.cm._store.all_blocks())

    def retrieve_top1(self, query: str) -> str | None:
        """Free: highest-similarity block content across context ∪ LT."""
        q_emb = self.cm.embed(query)
        self.controller.pre_prompt_promote(q_emb)
        ctx = self.cm.find_similar(q_emb, threshold=-1.0)
        lt_hits = self.lt.similarity_search(q_emb, top_k=1)
        ctx_sim = self._sim(q_emb, ctx.original_embedding) if ctx else -1.0
        lt_block, lt_sim = (lt_hits[0] if lt_hits else (None, -1.0))
        if ctx_sim >= lt_sim:
            return ctx.content if ctx else None
        return lt_block.content if lt_block else None

    def llm_probe(self, query: str, query_model: str) -> str:
        """Paid: build memory prompt and ask the LLM to answer from it."""
        assert self.client is not None
        q_emb = self.cm.embed(query)
        self.controller.pre_prompt_promote(q_emb)
        memory_prompt = self.cm.build_memory_prompt()
        response = self.client.messages.create(
            model=query_model,
            max_tokens=256,
            system=memory_prompt + QUERY_SYSTEM_SUFFIX,
            messages=[{"role": "user", "content": query}],
        )
        # self.client is metered in live mode, so usage is already recorded.
        return _extract_text(response.content)

    @staticmethod
    def _sim(a: list[float], b: list[float]) -> float:
        import numpy as np
        if not a or not b:
            return -1.0
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return float(np.dot(va, vb) / (na * nb)) if na and nb else -1.0
