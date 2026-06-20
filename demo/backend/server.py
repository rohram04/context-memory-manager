"""FastAPI demo backend: chat with the algorithmic memory agent and observe the
context-window + long-term-store state per message.

Wiring mirrors ``eval/agent_server.py`` but with these differences:
  - in-memory LT store (one per session)
  - ALGORITHMIC mode only, EMBEDDING novelty
  - an ``on_receive`` callback that captures a per-message snapshot + LT delta
  - one shared SentenceTransformer across all sessions (loading is expensive)

Session identity is handled by Starlette's signed-cookie SessionMiddleware (no
hand-rolled tokens). The signed cookie carries only the ``sid``; the live Agent
+ context heap live in a process-local registry keyed by that sid.

Run from the repo root:
    uvicorn demo.backend.server:app --port 8000
    uvicorn demo.backend.server:app --port 8000 --reload
For a no-cost local LLM via LiteLLM proxy, set DEMO_LOCAL=1 (see _build_client).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repo wiring (must precede project imports)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from starlette.middleware.sessions import SessionMiddleware  # noqa: E402

from llm.interface import LLMBackend  # noqa: E402
from llm_client import default_models, make_llm_backend  # noqa: E402

from demo.backend.scripts import list_scripts, script_messages  # noqa: E402
from demo.backend.sessions import (  # noqa: E402
    MAX_INPUT_CHARS,
    MAX_MESSAGES,
    Session,
    SessionRegistry,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    budget: Optional[int] = None
    mode: Optional[str] = None  # "algorithmic" (default) | "llm"


class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Client / embedder construction
# ---------------------------------------------------------------------------


def _build_backend() -> tuple[LLMBackend, str, str]:
    """Return (backend, model, util_model).

    DEMO_LOCAL=1 routes LLM calls through a local LiteLLM proxy (no cost), exactly
    like agent_server's --local path. Otherwise use the normal OpenRouter/Anthropic
    backend from llm_client.
    """
    if os.environ.get("DEMO_LOCAL") == "1":
        base_url = os.environ.get("DEMO_LOCAL_BASE_URL", "http://localhost:4000")
        model = os.environ.get("DEMO_LOCAL_MODEL", "ollama/llama3.1:8b")
        backend = make_llm_backend(local=True, local_base_url=base_url)
        return backend, model, model
    backend = make_llm_backend()
    model, util_model = default_models()
    return backend, model, util_model


# ---------------------------------------------------------------------------
# App + middleware + registry (module-level so `uvicorn demo.backend.server:app`)
# ---------------------------------------------------------------------------

_SECRET = os.environ.get("DEMO_SESSION_SECRET", uuid.uuid4().hex)

app = FastAPI(title="MemoryManager demo backend")

# Signed-cookie sessions (depends on itsdangerous). For local dev, same-site lax
# over http is fine; in prod set SameSite=None; Secure (see TODO(hosting)).
app.add_middleware(SessionMiddleware, secret_key=_SECRET, same_site="lax")

# CORS fallback for a non-proxied Vite dev server. allow_credentials so the
# signed session cookie is sent.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[demo] loading embedding model: all-MiniLM-L6-v2", flush=True)
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
_BACKEND, _MODEL, _UTIL_MODEL = _build_backend()
registry = SessionRegistry(_BACKEND, _EMBEDDER, _MODEL, _UTIL_MODEL)


# ---------------------------------------------------------------------------
# Dependency: resolve the signed-cookie session
# ---------------------------------------------------------------------------


def require_session(request: Request) -> Session:
    sid = request.session.get("sid")
    session = registry.get(sid)
    if session is None:
        raise HTTPException(status_code=403, detail="No active session")
    return session


# ---------------------------------------------------------------------------
# Core chat runner (shared by /chat and /script)
# ---------------------------------------------------------------------------


def _run_messages(session: Session, messages: list[str]) -> dict:
    """Run a sequence of chat messages, enforcing caps, and return the new
    per-message entries produced across all of them."""
    for msg in messages:
        if not msg or not msg.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        if len(msg) > MAX_INPUT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Message exceeds {MAX_INPUT_CHARS} chars",
            )

    start_index = len(session.context_snapshots)
    last_reply = ""
    last_latency_ms: float | None = None

    for msg in messages:
        if session.message_count >= MAX_MESSAGES:
            # 429: per-session message cap. TODO(hosting): add slowapi per-IP
            # rate limiting + a global cost kill-switch here.
            raise HTTPException(status_code=429, detail="Message cap reached")

        t0 = time.perf_counter()
        reply = session.agent.chat(msg)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        session.message_count += 1

        # The chat fired two on_receive callbacks (user, assistant). Attach
        # latency to the assistant-message snapshot (the last one appended).
        if session.context_snapshots:
            session.context_snapshots[-1]["latency_ms"] = latency_ms
        last_reply = reply
        last_latency_ms = latency_ms

    new_snapshots = session.context_snapshots[start_index:]
    new_events = session.lt_events[start_index:]
    new_messages = session.messages[start_index:]
    return {
        "messages": new_messages,
        "context_snapshots": new_snapshots,
        "lt_events": new_events,
        "reply": last_reply,
        "latency_ms": last_latency_ms,
    }


def _stream_messages(session: Session, messages: list[str]):
    """Sync NDJSON generator: drive agent.stream_chat per message under the
    session lock and map agent events to the streaming protocol lines."""
    with session.lock:
        for msg in messages:
            if session.message_count >= MAX_MESSAGES:
                yield json.dumps(
                    {"type": "error", "detail": "Message cap reached"}
                ) + "\n"
                return

            t0 = time.perf_counter()
            for ev in session.agent.stream_chat(msg):
                etype = ev.get("type")
                if etype == "user":
                    yield json.dumps({
                        "type": "user",
                        "snapshot": session.context_snapshots[-1],
                        "lt_event": session.lt_events[-1],
                    }) + "\n"
                elif etype == "token":
                    yield json.dumps(
                        {"type": "token", "delta": ev["delta"]}
                    ) + "\n"
                elif etype == "assistant":
                    session.context_snapshots[-1]["latency_ms"] = (
                        time.perf_counter() - t0
                    ) * 1000.0
                    yield json.dumps({
                        "type": "assistant",
                        "snapshot": session.context_snapshots[-1],
                        "lt_event": session.lt_events[-1],
                        "text": ev["text"],
                    }) + "\n"

            session.message_count += 1

        yield json.dumps({"type": "done"}) + "\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/api/session")
def create_session(req: CreateSessionRequest, request: Request) -> dict:
    # TODO(hosting): per-IP session-creation rate limit + global live-session cap.
    session = registry.create(req.budget, req.mode)
    request.session["sid"] = session.sid
    return {
        "budget": session.budget,
        "mode": session.mode,
        "model": session.model,
        "limits": session.limits(),
    }


@app.get("/api/session")
def get_session(session: Session = Depends(require_session)) -> dict:
    return {
        "budget": session.budget,
        "mode": session.mode,
        "model": session.model,
        "message_count": session.message_count,
        "limits": session.limits(),
    }


@app.delete("/api/session")
def delete_session(request: Request) -> dict:
    sid = request.session.get("sid")
    registry.drop(sid)
    request.session.pop("sid", None)
    return {"status": "ended"}


@app.post("/api/chat")
def chat(req: ChatRequest, session: Session = Depends(require_session)) -> dict:
    with session.lock:
        return _run_messages(session, [req.message])


@app.post("/api/chat/stream")
def chat_stream(
    req: ChatRequest, session: Session = Depends(require_session)
) -> StreamingResponse:
    return StreamingResponse(
        _stream_messages(session, [req.message]),
        media_type="application/x-ndjson",
    )


@app.get("/api/timeline")
def timeline(session: Session = Depends(require_session)) -> dict:
    return {
        "messages": session.messages,
        "context_snapshots": session.context_snapshots,
        "lt_events": session.lt_events,
    }


@app.get("/api/scripts")
def scripts() -> list[dict]:
    return list_scripts()


@app.post("/api/script/{name}")
def run_script(name: str, session: Session = Depends(require_session)) -> dict:
    messages = script_messages(name)
    if messages is None:
        raise HTTPException(status_code=404, detail=f"Unknown script: {name}")
    with session.lock:
        return _run_messages(session, messages)


@app.post("/api/script/{name}/stream")
def run_script_stream(
    name: str, session: Session = Depends(require_session)
) -> StreamingResponse:
    messages = script_messages(name)
    if messages is None:
        raise HTTPException(status_code=404, detail=f"Unknown script: {name}")
    return StreamingResponse(
        _stream_messages(session, messages),
        media_type="application/x-ndjson",
    )
