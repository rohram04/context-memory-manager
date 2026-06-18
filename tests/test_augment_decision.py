"""Regression tests for LLM-judged augmentation (controller.insert_or_augment).

When an augment_decision_fn is injected and config.llm_augment_decision is on, the
controller surfaces the top-k similar in-context blocks to the judge and augments into
the chosen block (or inserts a new one). When disabled / not injected it falls back to
the fixed-cosine find_similar path. These tests use a stub judge (no live API) and a
low candidate floor so the routing is exercised deterministically.

Run: python3 tests/test_augment_decision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager
from controller import MemoryController
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.store import ContextStore

_EMBEDDER = "all-MiniLM-L6-v2"


def _build(decision_fn, *, llm_augment_decision: bool = True, floor: float = -1.0):
    cfg = MemoryConfig(
        llm_augment_decision=llm_augment_decision,
        augment_candidate_floor=floor,        # -1 => every block is a candidate
        augment_candidate_top_k=5,
        augment_similarity_threshold=0.99,    # fallback path: effectively never merges
    )
    store = ContextStore(max_tokens=100_000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    cm = ContextManager(store, lt, embedding_model=_EMBEDDER, config=cfg)

    def merge_fn(existing: str, new: str):
        merged = f"{existing} | {new}"
        return merged, cm.embed(merged)

    ctrl = MemoryController(
        cm,
        compress_fn=lambda c: c,
        merge_fn=merge_fn,
        config=cfg,
        augment_decision_fn=decision_fn,
    )
    return cm, ctrl


def _recv(cm, ctrl, text, novelty: float = 0.5):
    return ctrl.insert_or_augment(text, cm.embed(text), novelty)


def test_judge_augments_into_chosen_block():
    choice = {"id": None}
    cm, ctrl = _build(lambda content, cands: choice["id"])
    a = _recv(cm, ctrl, "The Left Hand of Darkness is a sci-fi novel by Ursula K. Le Guin.")
    assert len(cm._store) == 1  # first message: no candidates -> new block

    choice["id"] = a.id  # judge now picks the existing book block
    b = _recv(cm, ctrl, "What does that book say about gender?")
    assert b.id == a.id              # merged into the same block
    assert len(cm._store) == 1       # no new block created
    assert "gender" in cm._store.get(a.id).content


def test_judge_inserts_new_when_none():
    cm, ctrl = _build(lambda content, cands: None)
    _recv(cm, ctrl, "The Left Hand of Darkness is a novel.")
    _recv(cm, ctrl, "I adopted a three-legged greyhound named Comet.")
    assert len(cm._store) == 2       # distinct topics stay separate


def test_existing_blocks_surfaced_as_candidates():
    seen = {}

    def judge(content, cands):
        seen["cands"] = cands
        return None

    cm, ctrl = _build(judge)
    a = _recv(cm, ctrl, "A note about a Lisbon trip in October.")
    _recv(cm, ctrl, "A note about feeding a sourdough starter.")
    _recv(cm, ctrl, "A third unrelated note.")  # judge sees prior blocks as candidates
    assert "cands" in seen
    assert all(isinstance(t, tuple) and len(t) == 2 for t in seen["cands"])
    assert a.id in {bid for bid, _ in seen["cands"]}


def test_fallback_cosine_path_when_disabled():
    cm, ctrl = _build(None, llm_augment_decision=False)
    _recv(cm, ctrl, "First block.")
    _recv(cm, ctrl, "Second unrelated block.")
    assert len(cm._store) == 2       # threshold 0.99 never merges -> two blocks


if __name__ == "__main__":
    test_judge_augments_into_chosen_block()
    test_judge_inserts_new_when_none()
    test_existing_blocks_surfaced_as_candidates()
    test_fallback_cosine_path_when_disabled()
    print("ok")
