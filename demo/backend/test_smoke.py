"""Offline smoke test for the demo backend — NO network, NO API key.

Builds a SessionRegistry with a stub LLM client whose ``messages.create`` returns
fixed responses (a tool-use ``provide_summary`` when compression tools are passed,
plain text otherwise). The full turn pipeline (controller.receive -> on_receive
snapshot, agent.chat, fit_budget) runs end-to-end.

Run:  .venv/bin/python -m pytest demo/backend/test_smoke.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sentence_transformers import SentenceTransformer  # noqa: E402

from llm.types import LLMResponse, ToolCall  # noqa: E402

from demo.backend.sessions import SessionRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# Stub LLM backend (duck-typed: complete + stream_complete)
# ---------------------------------------------------------------------------


class StubBackend:
    def complete(self, **kwargs) -> LLMResponse:
        if kwargs.get("tools"):
            tool_name = kwargs["tools"][0]["name"]
            if tool_name == "provide_summary":
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(
                        id="tool_stub",
                        name="provide_summary",
                        arguments={"summary": "canned summary."},
                    )],
                )
            if tool_name == "augment_into":
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(
                        id="tool_stub",
                        name="insert_new",
                        arguments={},
                    )],
                )
            return LLMResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tool_stub",
                    name=tool_name,
                    arguments={},
                )],
            )
        return LLMResponse(text="Acknowledged. (stub reply)")

    def stream_complete(self, **kwargs):
        yield ("token", "Acknowledged. (stub reply)")
        yield ("final", LLMResponse(text="Acknowledged. (stub reply)"))


# ---------------------------------------------------------------------------
# Shared embedder (real, but local/offline)
# ---------------------------------------------------------------------------

_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

REQUIRED_BLOCK_FIELDS = {
    "id", "content", "tier", "novelty", "decay", "fidelity",
    "token_cost", "compression_count", "access_count",
}
REQUIRED_SNAPSHOT_FIELDS = {
    "index", "role", "text", "budget_used", "budget_max",
    "budget_pressure", "memory_prompt_text", "blocks", "latency_ms",
}


def _registry() -> SessionRegistry:
    return SessionRegistry(StubBackend(), _EMBEDDER, "stub-model", "stub-util")


def test_chat_produces_two_snapshots_with_blocks():
    reg = _registry()
    session = reg.create(budget=400)
    with session.lock:
        result = []
        # mimic the server's _run_messages without importing FastAPI bits
        start = len(session.context_snapshots)
        session.agent.chat("Remember: the launch code is ZX9.")
        result = session.context_snapshots[start:]

    assert len(result) == 2  # user + assistant
    roles = [s["role"] for s in result]
    assert roles == ["user", "assistant"]

    for snap in result:
        assert REQUIRED_SNAPSHOT_FIELDS.issubset(snap.keys())
        assert isinstance(snap["blocks"], list)
        for field, value in snap.items():
            assert not callable(value), f"{field} is callable (live ref leaked)"
        for block in snap["blocks"]:
            assert REQUIRED_BLOCK_FIELDS.issubset(block.keys())
            for field, value in block.items():
                assert not callable(value), f"block.{field} is callable"
            assert block["tier"] in ("full", "stub")
            assert "embedding" not in block

    # The assistant snapshot's blocks should be non-empty (content was stored).
    assert result[1]["blocks"], "assistant snapshot has no memory blocks"


def test_lt_events_present_and_shaped():
    reg = _registry()
    session = reg.create(budget=400)
    with session.lock:
        session.agent.chat("Fact one about the project deadline next Tuesday.")
    # Parallel timelines: one lt_events entry per snapshot.
    assert len(session.lt_events) == len(session.context_snapshots)
    for ev in session.lt_events:
        assert set(ev.keys()) == {"added", "updated", "accessed"}


def test_sessions_are_isolated():
    reg = _registry()
    s1 = reg.create(budget=400)
    s2 = reg.create(budget=400)
    assert s1.sid != s2.sid

    with s1.lock:
        s1.agent.chat("Session ONE secret: ALPHA-111.")
    with s2.lock:
        s2.agent.chat("Session TWO secret: BETA-222.")

    s1_text = " ".join(m["text"] for m in s1.messages)
    s2_text = " ".join(m["text"] for m in s2.messages)
    assert "ALPHA-111" in s1_text
    assert "BETA-222" in s2_text
    # Neither session's timeline contains the other's content.
    assert "BETA-222" not in s1_text
    assert "ALPHA-111" not in s2_text

    # And the LT stores are physically separate engines.
    assert s1.lt_store is not s2.lt_store
