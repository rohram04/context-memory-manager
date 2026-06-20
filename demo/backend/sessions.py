"""Session registry + lifecycle for the demo backend.

A Session owns a *live, non-serializable* Agent + ContextManager + its own
in-memory LongTermStore. The library-managed signed cookie (Starlette
SessionMiddleware) holds only the ``sid`` string; the heavy live objects live
here in a process-local dict keyed by that sid. A TTL sweeper frees idle
sessions' memory (and disposes the LT engine).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from sentence_transformers import SentenceTransformer

from agent import Agent, MemoryMode
from ContextManager import ContextManager
from controller import MemoryController
from functions.llm_fns import (
    make_augment_decision_fn,
    make_compress_fn,
    make_merge_fn,
)
from llm.interface import LLMBackend
from memory.longterm import LongTermStore
from memory.novelty import NoveltyMode
from memory.store import ContextStore

from demo.backend.snapshots import (
    build_context_snapshot,
    diff_lt,
    _lt_state,
)

# Caps (LOCAL-ONLY SCOPE — light hardening; see TODO(hosting) markers in server.py)
MAX_MESSAGES = 50
MAX_INPUT_CHARS = 4000
IDLE_TTL_SECONDS = 30 * 60  # drop sessions idle > 30 min
ALLOWED_BUDGETS = (400, 800, 1200)
DEFAULT_BUDGET = 400

MODE = MemoryMode.ALGORITHMIC
NOVELTY_MODE = NoveltyMode.EMBEDDING
VALID_MODES = {m.value for m in MemoryMode}  # {"algorithmic", "llm"}


@dataclass
class Session:
    sid: str
    agent: Agent
    cm: ContextManager
    lt_store: LongTermStore
    budget: int
    model: str
    mode: str = MODE.value
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0
    # Per-message timelines (parallel lists, indexed by snapshot position).
    context_snapshots: list[dict[str, Any]] = field(default_factory=list)
    lt_events: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Internal: frozen LT state held from the previous message, for diffing.
    _prev_lt: dict[str, dict] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def touch(self) -> None:
        self.last_active = time.time()

    def record_snapshot(self, role: str, content: str, latency_ms: float | None) -> None:
        """on_receive callback target: append one per-message snapshot + LT delta."""
        index = len(self.context_snapshots)
        snap = build_context_snapshot(
            self.cm, index=index, role=role, text=content, latency_ms=latency_ms
        )
        curr_lt = _lt_state(self.cm)
        events = diff_lt(self._prev_lt, curr_lt)
        self._prev_lt = curr_lt
        self.context_snapshots.append(snap)
        self.lt_events.append(events)
        self.messages.append({"index": index, "role": role, "text": content})

    def limits(self) -> dict[str, int]:
        return {"max_messages": MAX_MESSAGES, "max_input_chars": MAX_INPUT_CHARS}


class SessionRegistry:
    """Process-local registry of live sessions, keyed by signed-cookie sid."""

    def __init__(
        self,
        backend: LLMBackend,
        embedder: SentenceTransformer,
        model: str,
        util_model: str,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._model = model
        self._util_model = util_model
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, budget: int | None, mode: str | None = None) -> Session:
        budget = budget if budget in ALLOWED_BUDGETS else DEFAULT_BUDGET
        mode_val = mode if mode in VALID_MODES else MODE.value
        agent_mode = MemoryMode(mode_val)
        sid = uuid.uuid4().hex

        store = ContextStore(max_tokens=budget)
        # In-memory LT store (StaticPool wiring handled inside LongTermStore).
        lt_store = LongTermStore("sqlite:///:memory:")
        cm = ContextManager(store, lt_store, embedding_model=self._embedder)
        controller = MemoryController(
            cm,
            compress_fn=make_compress_fn(self._backend, self._util_model),
            merge_fn=make_merge_fn(self._backend, cm, self._util_model),
            augment_decision_fn=make_augment_decision_fn(self._backend, self._util_model),
        )
        session = Session(
            sid=sid,
            agent=None,  # set below
            cm=cm,
            lt_store=lt_store,
            budget=budget,
            model=self._model,
            mode=mode_val,
        )
        agent = Agent(
            controller,
            self._backend,
            model=self._model,
            mode=agent_mode,
            novelty_mode=NOVELTY_MODE,
            novelty_model=self._util_model,
            # Latency is unknown at receive time (the user-message snapshot fires
            # before the LLM call). The server patches latency_ms onto the
            # assistant-message snapshot after agent.chat() returns.
            on_receive=lambda role, content: session.record_snapshot(
                role, content, None
            ),
        )
        session.agent = agent

        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        self.sweep()
        with self._lock:
            session = self._sessions.get(sid)
        if session is not None:
            session.touch()
        return session

    def drop(self, sid: str | None) -> None:
        if not sid:
            return
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is not None:
            self._dispose(session)

    def _dispose(self, session: Session) -> None:
        try:
            session.lt_store._engine.dispose()
        except Exception:
            pass

    def sweep(self) -> None:
        """Drop sessions idle past the TTL and free their memory/engine."""
        now = time.time()
        expired: list[Session] = []
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if now - session.last_active > IDLE_TTL_SECONDS:
                    expired.append(self._sessions.pop(sid))
        for session in expired:
            self._dispose(session)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
