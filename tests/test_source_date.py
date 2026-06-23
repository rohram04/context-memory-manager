"""source_date stamping + carry-through across the block lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from ContextManager import ContextManager
from memory.block import CacheBlock
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.store import ContextStore

D1 = datetime(2023, 5, 1, tzinfo=timezone.utc)
D2 = datetime(2023, 6, 1, tzinfo=timezone.utc)


class _FixedEmbedder:
    """Constant 8-d embedding — keeps the test fast and LLM/ST-free."""

    def encode(self, text, convert_to_numpy: bool = True):
        return np.ones(8, dtype=np.float32)


def _cm() -> ContextManager:
    cfg = MemoryConfig()
    store = ContextStore(max_tokens=10_000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=8)
    return ContextManager(store, lt, embedding_model=_FixedEmbedder(), config=cfg)


def _block(cm: ContextManager, content: str) -> CacheBlock:
    return CacheBlock(
        content=content, original_embedding=cm.embed(content), novelty_score=0.5
    )


def test_insert_stamps_ambient_source_date():
    cm = _cm()
    cm.set_source_date(D1)
    b = _block(cm, "fact one")
    cm.insert(b)
    assert cm._store.get(b.id).source_date == D1


def test_insert_keeps_explicit_source_date_over_ambient():
    cm = _cm()
    cm.set_source_date(D2)
    b = _block(cm, "fact")
    b.source_date = D1  # explicit wins
    cm.insert(b)
    assert cm._store.get(b.id).source_date == D1


def test_augment_updates_source_date_to_latest_mention():
    cm = _cm()
    cm.set_source_date(D1)
    b = _block(cm, "fact")
    cm.insert(b)
    cm.set_source_date(D2)
    cm.augment(b.id, "fact, revised", cm.embed("fact, revised"))
    assert cm._store.get(b.id).source_date == D2


def test_evict_then_promote_roundtrips_source_date_through_lt():
    cm = _cm()
    cm.set_source_date(D1)
    b = _block(cm, "fact")
    cm.insert(b)
    cm.evict(b.id)
    lt_id = b.pointer_to_lt_id
    assert cm._lt.get(lt_id).source_date == D1  # survived serialization to LT
    cm.set_source_date(None)  # query-time: no ambient date
    promoted = cm.promote(lt_id)
    assert promoted.source_date == D1  # carried back, not overwritten


def test_compress_copies_source_date_to_lt():
    cm = _cm()
    cm.set_source_date(D1)
    b = _block(cm, "a reasonably long memory block about a fact")
    cm.insert(b)
    cm.compress(b.id, "short summary")
    assert cm._lt.get(b.pointer_to_lt_id).source_date == D1
