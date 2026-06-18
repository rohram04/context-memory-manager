"""Test the retrieval-grounded novelty mode plumbing with a mock LLM client.

No paid API calls — a fake client captures the prompt and returns a canned float.
Verifies: top-k neighbors are gathered and injected, the float is parsed, and a
client error falls back to embedding novelty. Run:
    python3 tests/test_novelty_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager
from memory.block import CacheBlock
from memory.config import MemoryConfig
from memory.longterm import LongTermStore
from memory.novelty import NoveltyMode, build_novelty_fn, score_novelty_retrieval_llm
from memory.store import ContextStore


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeContent(text)]


class _FakeClient:
    """Captures the last prompt; returns a canned text or raises if boom=True."""

    def __init__(self, reply: str = "0.9", boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.last_prompt: str | None = None
        self.messages = self

    def create(self, **kwargs):  # mimics client.messages.create
        if self.boom:
            raise RuntimeError("simulated API failure")
        self.last_prompt = kwargs["messages"][0]["content"]
        return _FakeResponse(self.reply)


def _build_cm() -> ContextManager:
    cfg = MemoryConfig()
    store = ContextStore(max_tokens=10000, config=cfg)
    lt = LongTermStore("sqlite:///:memory:", embedding_dim=384)
    cm = ContextManager(store, lt, embedding_model="all-MiniLM-L6-v2", config=cfg)
    return cm


def test_retrieval_novelty_injects_neighbors_and_parses() -> None:
    cm = _build_cm()
    text = "The Helix-9 protocol failed its third pressure test."
    emb = cm.embed(text)
    cm.insert(CacheBlock(content=text, original_embedding=emb, novelty_score=0.5))

    client = _FakeClient(reply="0.9")
    new = "Update: the Helix-9 protocol passed its pressure test after all."
    score = score_novelty_retrieval_llm(new, cm.embed(new), cm, client, "m", top_k=5)

    assert abs(score - 0.9) < 1e-6, score
    assert client.last_prompt is not None
    # The nearest neighbor's content must appear in the grounded prompt.
    assert "Helix-9 protocol failed" in client.last_prompt
    assert new in client.last_prompt


def test_retrieval_novelty_falls_back_on_error() -> None:
    cm = _build_cm()
    t = "An unrelated fact about tidal estuaries."
    cm.insert(CacheBlock(content=t, original_embedding=cm.embed(t), novelty_score=0.5))

    boom = _FakeClient(boom=True)
    new = "A completely different topic: medieval glassblowing in Murano."
    score = score_novelty_retrieval_llm(new, cm.embed(new), cm, boom, "m")
    # Falls back to embedding novelty -> a valid [0,1] score, and high (dissimilar).
    assert 0.0 <= score <= 1.0, score
    assert score > 0.5, score


def test_factory_wires_retrieval_mode() -> None:
    cm = _build_cm()
    client = _FakeClient(reply="0.42")
    fn = build_novelty_fn(NoveltyMode.RETRIEVAL_LLM, cm, client, "m")
    s = fn("some content", cm.embed("some content"))
    assert abs(s - 0.42) < 1e-6, s


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"\nAll {len(tests)} retrieval-novelty tests passed.")


if __name__ == "__main__":
    _run_all()
