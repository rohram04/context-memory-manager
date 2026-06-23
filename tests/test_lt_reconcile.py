"""LT/context reconciliation: augmented info is never lost.

Policy under test: the long-term store always holds the highest-fidelity (lossless
union) copy of a pointered block. Augment is deferred (marks dirty); the lossless
union reconcile runs lazily on compress / evict / re-promote.

Uses the concat-stub union fn (ContextManager default) + the real all-MiniLM-L6-v2
embedder. No API. Unique opaque tokens are grepped for in LT content to prove the
augmented fact survives each lossy operation. Run:
    python3 -m pytest tests/test_lt_reconcile.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ContextManager import ContextManager
from memory.block import CacheBlock
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.store import ContextStore

# Opaque tokens we can grep for in LT/context content.
TOK_ORIG = "ZZQ_ORIGINAL_7781"
TOK_AUG = "ZZQ_AUGMENTED_4432"


def _build_cm(max_tokens: int = 100_000) -> ContextManager:
    cfg = MemoryConfig()
    store = ContextStore(max_tokens=max_tokens, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    # No union_merge_fn passed -> ContextManager defaults to lossless _concat_union.
    return ContextManager(store, lt, embedding_model="all-MiniLM-L6-v2", config=cfg)


def _insert(cm: ContextManager, content: str) -> CacheBlock:
    block = CacheBlock(content=content, original_embedding=cm.embed(content), novelty_score=0.5)
    cm.insert(block)
    return block


def _augment(cm: ContextManager, block: CacheBlock, merged_content: str) -> None:
    cm.augment(block.id, merged_content, cm.embed(merged_content))


def _cos(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    return float(np.dot(va, vb) / (na * nb)) if na and nb else 0.0


def test_augment_sets_dirty() -> None:
    cm = _build_cm()
    block = _insert(cm, f"Base fact {TOK_ORIG}")
    assert block.dirty is False
    _augment(cm, block, f"Base fact {TOK_ORIG}\nUpdate {TOK_AUG}")
    assert cm._store.get(block.id).dirty is True


# A "compressed" summary that retains the opaque token stays semantically close to
# the original embedding, so fidelity remains above the removal threshold and the
# block survives in context as a stub. (Used where a stub must remain.)
def _survivable_compress(token: str) -> str:
    return f"compact note {token}"


def test_first_pass_compress_clears_dirty_and_lt_has_augmented_content() -> None:
    cm = _build_cm()
    block = _insert(cm, f"Base fact {TOK_ORIG}")
    _augment(cm, block, f"Base fact {TOK_ORIG}\nUpdate {TOK_AUG}")

    # First compression pass: pointer is created, current (augmented) content copied to LT.
    result = cm.compress(block.id, _survivable_compress(TOK_ORIG))
    assert result.pointer_to_lt_id is not None
    assert result.dirty is False
    lt = cm._lt.get(result.pointer_to_lt_id)
    assert TOK_AUG in lt.content, lt.content
    assert TOK_ORIG in lt.content, lt.content


def test_case2_compress_augment_compress_keeps_augmented_in_lt() -> None:
    # compress -> augment -> compress again ⇒ second augment token survives in LT.
    cm = _build_cm()
    block = _insert(cm, f"Base fact {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))  # first pass: LT from original
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None, "stub should survive first pass"

    # New info merged in after the block is already pointered.
    _augment(cm, block, f"compact note {TOK_ORIG}\n{TOK_AUG}")
    assert cm._store.get(block.id).dirty is True

    # Second compress reconciles the augmented fact into LT before the lossy pass.
    cm.compress(block.id, _survivable_compress(TOK_ORIG))
    lt = cm._lt.get(pointer)
    assert TOK_AUG in lt.content, lt.content
    surviving = cm._store.get(block.id)
    assert surviving is None or surviving.dirty is False


def test_evict_dirty_pointered_block_unions_into_lt() -> None:
    cm = _build_cm()
    block = _insert(cm, f"Original {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))  # create LT pointer
    pointer = r1.pointer_to_lt_id

    _augment(cm, block, f"compact note {TOK_ORIG}\nNew detail {TOK_AUG}")
    cm.evict(block.id)

    assert cm._store.get(block.id) is None  # removed from context
    lt = cm._lt.get(pointer)
    assert TOK_ORIG in lt.content, lt.content
    assert TOK_AUG in lt.content, lt.content


def test_promote_dirty_stub_unions_both_directions_and_clears_dirty() -> None:
    cm = _build_cm()
    block = _insert(cm, f"Original {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))  # LT pointer; block now a stub
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None, "stub should survive first pass"

    _augment(cm, block, f"Stub content {TOK_AUG}")
    assert cm._store.get(block.id).dirty is True

    promoted = cm.promote(pointer)
    assert promoted is not None
    # Union lives in context now: both LT's original token and the stub's new token.
    assert TOK_ORIG in promoted.content, promoted.content
    assert TOK_AUG in promoted.content, promoted.content
    assert promoted.dirty is False
    lt = cm._lt.get(pointer)
    assert TOK_ORIG in lt.content and TOK_AUG in lt.content, lt.content


def test_promote_clean_stub_sets_current_embedding_to_content() -> None:
    cm = _build_cm()
    block = _insert(cm, f"A fairly distinctive fact about {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))  # LT pointer; stub is clean
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None, "stub should survive first pass"

    promoted = cm.promote(pointer)
    assert promoted is not None
    assert promoted.dirty is False
    # current_embedding should describe the promoted (LT) content.
    sim = _cos(promoted.current_embedding, cm.embed(promoted.content))
    assert sim > 0.99, sim


def test_roundtrip_augmented_fact_recoverable_after_removal() -> None:
    # A fact augmented in then compressed-to-removal is still recoverable from LT.
    cfg = MemoryConfig()
    store = ContextStore(max_tokens=100_000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    cm = ContextManager(store, lt, embedding_model="all-MiniLM-L6-v2", config=cfg)

    block = _insert(cm, f"Project status {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))  # create LT pointer, stub survives
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None

    _augment(cm, block, f"compact note {TOK_ORIG}\nLate-breaking {TOK_AUG}")

    # Force fidelity-triggered removal on the next compress: garbage compressed text
    # has near-zero similarity to the original embedding, so the block exits context.
    cm.compress(block.id, "qwxz unrelated gibberish noise vector")
    assert cm._store.get(block.id) is None  # gone from context

    # ...but the augmented token survived into LT (reconciled before the lossy pass).
    lt_block = cm._lt.get(pointer)
    assert TOK_AUG in lt_block.content, lt_block.content
    assert TOK_ORIG in lt_block.content, lt_block.content

    # And it is retrievable via similarity search.
    hits = cm._lt.similarity_search(cm.embed(f"Late-breaking {TOK_AUG}"), top_k=1)
    assert hits and TOK_AUG in hits[0][0].content


def test_promote_records_access_on_context_block() -> None:
    """Promotion must refresh in-context recency (not just LT access)."""
    cfg = MemoryConfig(clock_seconds_per_turn=600.0)
    store = ContextStore(max_tokens=100_000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    cm = ContextManager(store, lt, embedding_model="all-MiniLM-L6-v2", config=cfg)
    block = _insert(cm, f"Original {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None

    cm.clock.tick()
    cm.clock.tick()
    cm.clock.tick()
    aged_decay = cm._store.get(block.id).decay_score

    promoted = cm.promote(pointer)
    assert promoted is not None
    assert promoted.access_count >= 1
    assert promoted.decay_score > aged_decay
    assert promoted.decay_score > 0.99

    # New-block path: fully evict stub, then promote from LT.
    cm.evict(promoted.id)
    assert cm._store.get(promoted.id) is None
    promoted2 = cm.promote(pointer)
    assert promoted2 is not None
    assert promoted2.access_count >= 1
    assert promoted2.decay_score > 0.99


def test_compress_low_fidelity_guard_reconciles_dirty_block() -> None:
    """Early compress() fidelity guard must reconcile dirty blocks before removal."""
    cm = _build_cm()
    block = _insert(cm, f"Original {TOK_ORIG}")
    r1 = cm.compress(block.id, _survivable_compress(TOK_ORIG))
    pointer = r1.pointer_to_lt_id
    assert cm._store.get(block.id) is not None

    _augment(cm, block, f"compact note {TOK_ORIG}\nStale guard {TOK_AUG}")
    stub = cm._store.get(block.id)
    assert stub is not None and stub.dirty is True
    stub.fidelity = 0.1  # simulates stale heap entry still in context

    cm.compress(block.id, "ignored summary")
    assert cm._store.get(block.id) is None
    lt = cm._lt.get(pointer)
    assert TOK_ORIG in lt.content, lt.content
    assert TOK_AUG in lt.content, lt.content


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"\nAll {len(tests)} reconcile tests passed.")


if __name__ == "__main__":
    _run_all()
