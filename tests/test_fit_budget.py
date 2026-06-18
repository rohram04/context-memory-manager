"""Regression tests for the fit_budget infinite-loop fix.

Before the fix, MemoryController.fit_budget looped `while over budget: compress`,
but next_candidate() never returns None while the store is non-empty and a faithful
summary of an already-short block neither shrinks it nor drops its fidelity below the
removal threshold — so the loop spun forever issuing one compress call per iteration.

These tests drive fit_budget with mock compress_fns (no live API) and assert it always
terminates, always lands within budget, and makes a bounded number of compress calls.
Run: python3 tests/test_fit_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager
from controller import MemoryController
from memory.block import CacheBlock
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.store import ContextStore

_EMBEDDER = "all-MiniLM-L6-v2"


def _noop_merge(existing: str, new: str):
    merged = f"{existing} {new}"
    return merged, []  # embedding unused by fit_budget


def _build(max_tokens: int, compress_fn, config: MemoryConfig | None = None):
    cfg = config or MemoryConfig()
    store = ContextStore(max_tokens=max_tokens, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    cm = ContextManager(store, lt, embedding_model=_EMBEDDER, config=cfg)
    controller = MemoryController(cm, compress_fn, _noop_merge, cfg)
    return cm, lt, controller


def _seed(cm: ContextManager, n: int) -> list[str]:
    """Insert n distinct ~15-token blocks; return their ids."""
    ids = []
    for i in range(n):
        content = (
            f"Memory block number {i} records a distinct fact about topic {i} "
            f"with several supporting details and entities for item {i}."
        )
        emb = cm.embed(content)
        block = CacheBlock(content=content, original_embedding=emb, novelty_score=0.5)
        cm.insert(block)
        ids.append(block.id)
    return ids


def test_cannot_compress_signal_evicts_and_terminates() -> None:
    """compress_fn returns '' (the model's cannot_compress signal) -> evict, don't loop."""
    cm, lt, controller = _build(max_tokens=30, compress_fn=lambda c: "")
    _seed(cm, 5)
    assert cm.budget_fraction > 1.0  # genuinely over budget

    controller.fit_budget()  # pre-fix: hangs forever

    assert cm.used_tokens <= cm._store.max_tokens
    assert len(lt) >= 1  # evicted blocks live in long-term store
    print("ok: cannot_compress signal evicts and terminates")


def test_non_shrinking_summary_terminates_with_bounded_calls() -> None:
    """A 'summary' that never shrinks must still terminate via the no-progress guard."""
    calls = {"n": 0}

    def echo(content: str) -> str:
        calls["n"] += 1
        return content  # non-empty, but identical length -> no progress

    cm, lt, controller = _build(max_tokens=30, compress_fn=echo)
    n_blocks = len(_seed(cm, 5))

    controller.fit_budget()  # pre-fix: hangs forever

    assert cm.used_tokens <= cm._store.max_tokens
    # Bounded work: at most one compress attempt per block before it is evicted.
    assert calls["n"] <= n_blocks, f"unbounded compress calls: {calls['n']}"
    print(f"ok: non-shrinking summary terminates ({calls['n']} compress calls)")


def test_shrinking_summary_compresses_without_evicting() -> None:
    """Happy path: a genuinely shorter summary shrinks blocks; they stay in context."""
    # Disable fidelity-triggered removal so this isolates the budget loop from the
    # (embedding-dependent) fidelity check.
    cfg = MemoryConfig(fidelity_removal_threshold=0.0)
    cm, lt, controller = _build(
        max_tokens=60, compress_fn=lambda c: c[: max(10, len(c) // 2)], config=cfg
    )
    _seed(cm, 5)

    controller.fit_budget()

    assert cm.used_tokens <= cm._store.max_tokens
    assert len(cm._store) > 0, "everything was evicted on the happy path"
    assert any(
        b.compression_count > 0 for b in cm._store.all_blocks()
    ), "no block was actually compressed"
    print("ok: shrinking summary compresses without evicting")


if __name__ == "__main__":
    test_cannot_compress_signal_evicts_and_terminates()
    test_non_shrinking_summary_terminates_with_bounded_calls()
    test_shrinking_summary_compresses_without_evicting()
    print("\nall fit_budget regression tests passed")
