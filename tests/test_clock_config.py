"""Unit tests for the turn-based clock and MemoryConfig threading.

No embeddings, no API calls — runs in milliseconds. Run directly:
    python3 tests/test_clock_config.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory.block import CacheBlock
from memory.clock import CLOCK
from memory.config import MemoryConfig
from memory.store import ContextStore


def _block(content: str, novelty: float = 0.5) -> CacheBlock:
    return CacheBlock(content=content, original_embedding=[1.0, 0.0], novelty_score=novelty)


def test_wall_clock_decay_is_inert_for_fast_creation() -> None:
    """Default (wall-clock) mode: blocks created back-to-back all read decay ~= 1.0."""
    CLOCK.reset(0.0)
    a = _block("a")
    b = _block("b")
    assert a.decay_score > 0.999, a.decay_score
    assert b.decay_score > 0.999, b.decay_score


def test_simulated_clock_differentiates_blocks() -> None:
    """Simulated turn clock: a block created earlier (more ticks ago) decays more."""
    CLOCK.reset(600.0)  # 600s/turn vs 3600s decay constant
    early = _block("early")          # stamped at epoch (0 ticks)
    CLOCK.tick()                     # +600s
    CLOCK.tick()                     # +1200s
    CLOCK.tick()                     # +1800s
    late = _block("late")            # stamped at +1800s
    # early is 1800s old -> exp(-1800/3600)=exp(-0.5)~=0.607
    assert math.isclose(early.decay_score, math.exp(-0.5), rel_tol=1e-3), early.decay_score
    assert late.decay_score > 0.999, late.decay_score
    assert early.decay_score < late.decay_score
    CLOCK.reset(0.0)  # restore default for other tests


def test_record_access_refreshes_recency() -> None:
    CLOCK.reset(600.0)
    blk = _block("x")
    CLOCK.tick(); CLOCK.tick(); CLOCK.tick()
    aged = blk.decay_score
    blk.record_access()              # touch it "now"
    assert blk.access_count == 1
    assert blk.decay_score > aged
    assert blk.decay_score > 0.999
    CLOCK.reset(0.0)


def test_config_priority_weights_are_used() -> None:
    """_priority must reflect config weights, not the old hardcoded 0.6/0.4."""
    CLOCK.reset(0.0)
    # All-novelty config: priority depends only on (1 - novelty); decay term zeroed.
    cfg = MemoryConfig(compress_novelty_weight=1.0, compress_recency_weight=0.0)
    store = ContextStore(max_tokens=1000, config=cfg)
    high_nov = _block("aaaa", novelty=0.9)
    low_nov = _block("aaaa", novelty=0.1)  # same token_cost
    p_high = store._priority(high_nov)
    p_low = store._priority(low_nov)
    # low novelty => more compressible => higher priority
    assert p_low > p_high, (p_low, p_high)


def test_config_pressure_thresholds_are_used() -> None:
    cfg = MemoryConfig(pressure_thresholds={"critical": 0.5, "high": 0.3, "medium": 0.1, "low": 0.0})
    store = ContextStore(max_tokens=100, config=cfg)
    store.add(_block("token " * 60))  # ~60 tokens -> >50% of 100 -> critical
    assert store.budget_pressure == "critical", (store.used_tokens, store.budget_pressure)


def test_default_config_reproduces_legacy_constants() -> None:
    cfg = MemoryConfig()
    assert cfg.compress_novelty_weight == 0.6
    assert cfg.compress_recency_weight == 0.4
    assert cfg.fidelity_removal_threshold == 0.50
    assert cfg.promote_alpha == 0.7 and cfg.promote_beta == 0.3 and cfg.promote_threshold == 0.6
    assert cfg.augment_similarity_threshold == 0.85
    assert cfg.pressure_thresholds["critical"] == 0.90


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"\nAll {len(tests)} clock/config tests passed.")


if __name__ == "__main__":
    _run_all()
