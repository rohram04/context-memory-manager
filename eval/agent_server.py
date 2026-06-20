"""FastAPI shim around `agent.Agent` for MemoryAgentBench.

Exposes one endpoint, `POST /send_message`, matching the benchmark's
`AgentWrapper.send_message(message, memorizing, query_id, context_id)` contract.

MemoryAgentBench drives contexts strictly sequentially: all queries for context N
finish before context N+1 begins. So this server keeps exactly one live `Agent`
at a time, keyed by `context_id`. When a new `context_id` arrives the previous
agent is dropped and its SQLite file deleted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Project imports — same wiring as cli.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent import Agent, MemoryMode  # noqa: E402
from ContextManager import ContextManager  # noqa: E402
from controller import MemoryController  # noqa: E402
from functions.llm_fns import make_compress_fn, make_merge_fn  # noqa: E402
from llm.interface import LLMBackend  # noqa: E402
from llm_client import has_llm_key, make_llm_backend  # noqa: E402
from memory.block import _enc  # shared tiktoken cl100k_base encoder  # noqa: E402
from memory.longterm import LongTermStore  # noqa: E402
from memory.novelty import NoveltyMode  # noqa: E402
from memory.store import ContextStore  # noqa: E402


CONTEXTS_DIR = REPO_ROOT / "eval" / "contexts"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SendMessageRequest(BaseModel):
    message: str
    memorizing: bool = False
    query_id: Optional[int] = None
    context_id: int


class SendMessageResponse(BaseModel):
    output: str
    input_len: int
    output_len: int
    memory_construction_time: float
    query_time_len: float


# ---------------------------------------------------------------------------
# Server state — exactly one live agent at a time
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(
        self,
        backend: LLMBackend,
        embedder: SentenceTransformer,
        default_mode: MemoryMode,
        max_tokens: int,
        model: str,
        util_model: str,
        novelty_mode: NoveltyMode,
    ) -> None:
        self.backend = backend
        self.embedder = embedder
        self.default_mode = default_mode
        self.max_tokens = max_tokens
        self.model = model
        self.util_model = util_model
        self.novelty_mode = novelty_mode

        self._context_id: Optional[int] = None
        self._agent: Optional[Agent] = None
        self._db_path: Optional[Path] = None
        self._mode: Optional[MemoryMode] = None

    def get_or_create(self, context_id: int, mode: MemoryMode) -> Agent:
        if self._agent is not None and self._context_id == context_id:
            return self._agent

        # New context_id — tear down the previous agent (if any) and its DB.
        self._teardown_current()

        db_path = CONTEXTS_DIR / f"eval_context_{context_id}.db"
        # Start fresh: if a stale file from a prior run exists, drop it.
        if db_path.exists():
            db_path.unlink()

        store = ContextStore(max_tokens=self.max_tokens)
        lt_store = LongTermStore(f"sqlite:///{db_path}")
        cm = ContextManager(store, lt_store, embedding_model=self.embedder)
        controller = MemoryController(
            cm,
            compress_fn=make_compress_fn(self.backend, self.util_model),
            merge_fn=make_merge_fn(self.backend, cm, self.util_model),
        )
        agent = Agent(
            controller,
            self.backend,
            model=self.model,
            mode=mode,
            novelty_mode=self.novelty_mode,
            novelty_model=self.util_model,
        )

        self._context_id = context_id
        self._agent = agent
        self._db_path = db_path
        self._mode = mode
        print(
            f"[agent_server] new agent for context_id={context_id} "
            f"mode={mode.value} db={db_path.name}",
            flush=True,
        )
        return agent

    def _teardown_current(self) -> None:
        if self._agent is None:
            return
        old_id = self._context_id
        old_db = self._db_path
        self._context_id = None
        self._agent = None
        self._db_path = None
        self._mode = None
        if old_db is not None and old_db.exists():
            try:
                old_db.unlink()
            except OSError as e:
                print(
                    f"[agent_server] warn: could not delete {old_db}: {e}",
                    flush=True,
                )
        print(f"[agent_server] tore down agent for context_id={old_id}", flush=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def build_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="MemoryManager agent server")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "default_mode": state.default_mode.value}

    @app.post("/send_message", response_model=SendMessageResponse)
    def send_message(
        req: SendMessageRequest,
        mode: Optional[str] = Query(default=None),
    ) -> SendMessageResponse:
        effective_mode = MemoryMode(mode) if mode else state.default_mode
        agent = state.get_or_create(req.context_id, effective_mode)

        t0 = time.perf_counter()
        if req.memorizing:
            # Ingest only — skip the LLM conversation turn. Batch ingestion (e.g.
            # MemoryAgentBench memorize phase) feeds many chunks per context; a
            # full Sonnet reply per chunk is wildly expensive and unnecessary.
            agent.ingest(req.message)
            reply = "Memorized"
        else:
            reply = agent.chat(req.message)
        elapsed = time.perf_counter() - t0

        input_len = len(_enc.encode(req.message))
        output_len = len(_enc.encode(reply))

        return SendMessageResponse(
            output=reply,
            input_len=input_len,
            output_len=output_len,
            memory_construction_time=elapsed if req.memorizing else 0.0,
            query_time_len=0.0 if req.memorizing else elapsed,
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _load_api_key() -> str:
    """Read API-Key from the project's .env (handoff specifies this exact name)."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        values = dotenv_values(env_path)
        key = values.get("API-Key") or values.get("ANTHROPIC_API_KEY")
        if key:
            return key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    print(
        "Error: no API key found. Set API-Key in .env or ANTHROPIC_API_KEY in env.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastAPI shim wrapping the memory-manager Agent for MemoryAgentBench.",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in MemoryMode],
        default=MemoryMode.ALGORITHMIC.value,
        help="Default memory mode. Can be overridden per-request via ?mode=... .",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--util-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--novelty",
        choices=[m.value for m in NoveltyMode],
        default=NoveltyMode.EMBEDDING.value,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Route LLM calls through a local LiteLLM proxy (set up via start_local.sh) "
        "instead of the real Anthropic API. Useful for smoke tests without burning credits.",
    )
    parser.add_argument(
        "--local-base-url",
        default="http://localhost:4000",
        help="Base URL for the LiteLLM proxy when --local is set.",
    )
    parser.add_argument(
        "--local-model",
        default="ollama/llama3.1:8b",
        help="Model name passed to LiteLLM when --local is set. Overrides --model and --util-model.",
    )
    args = parser.parse_args()

    CONTEXTS_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe any leftover per-context DBs from prior server runs so each
    # invocation starts from a clean slate.
    stale = list(CONTEXTS_DIR.glob("eval_context_*.db"))
    for f in stale:
        try:
            f.unlink()
        except OSError as e:
            print(f"[agent_server] warn: could not delete {f}: {e}", flush=True)
    if stale:
        print(f"[agent_server] cleared {len(stale)} stale context DB(s)", flush=True)

    if args.local:
        backend = make_llm_backend(
            local=True,
            local_base_url=args.local_base_url,
        )
        args.model = args.local_model
        args.util_model = args.local_model
    else:
        if not has_llm_key():
            _load_api_key()
        backend = make_llm_backend()

    print(f"[agent_server] loading embedding model: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)

    state = ServerState(
        backend=backend,
        embedder=embedder,
        default_mode=MemoryMode(args.mode),
        max_tokens=args.max_tokens,
        model=args.model,
        util_model=args.util_model,
        novelty_mode=NoveltyMode(args.novelty),
    )

    app = build_app(state)
    print(
        f"[agent_server] listening on {args.host}:{args.port} "
        f"(default mode={args.mode}, model={args.model}, budget={args.max_tokens})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
