"""Per-agent clock isolation — concurrent agents must not share simulated time."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager
from memory.block import CacheBlock
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.store import ContextStore


class _FakeEmbedder:
    def encode(self, text, convert_to_numpy: bool = True):
        import numpy as np
        return np.ones(4, dtype=np.float32)


def _make_cm() -> ContextManager:
    cfg = MemoryConfig(clock_seconds_per_turn=600.0)
    store = ContextStore(max_tokens=1000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=4)
    return ContextManager(store, lt, embedding_model=_FakeEmbedder(), config=cfg)


def test_per_agent_clock_isolation() -> None:
    """Ticking agent A's clock must not affect agent B's decay scores."""
    cm_a = _make_cm()
    cm_b = _make_cm()

    block_a = CacheBlock(content="a", original_embedding=[1.0, 0.0], novelty_score=0.5)
    cm_a.insert(block_a)

    cm_a.clock.tick()
    cm_a.clock.tick()
    cm_a.clock.tick()

    decay_a_before = block_a.decay_score

    block_b = CacheBlock(content="b", original_embedding=[1.0, 0.0], novelty_score=0.5)
    cm_b.insert(block_b)
    decay_b_fresh = block_b.decay_score

    assert block_b.decay_score > 0.999, block_b.decay_score
    assert block_a.decay_score < 0.7, block_a.decay_score

    cm_b.clock.tick()

    assert block_a.decay_score == decay_a_before
    assert block_b.decay_score < decay_b_fresh
    assert math.isclose(block_b.decay_score, math.exp(-600 / 3600), rel_tol=1e-3)
