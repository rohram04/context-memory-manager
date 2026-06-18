# Demo backend

FastAPI app that lets a browser chat with the **algorithmic** memory agent and
observe the context-window + long-term-store state per message.

## Run (from repo root)

```bash
uvicorn demo.backend.server:app --port 8000
# add --reload for dev
```

> Default port is **8000** (8080 is intentionally avoided — it commonly
> conflicts with other local tooling).

LLM client:
- Default: uses `llm_client.make_anthropic_client()` / `default_models()`
  (native Anthropic via `ANTHROPIC_API_KEY`, or OpenRouter if its key is set).
- No-cost local path: `DEMO_LOCAL=1` routes LLM calls through a LiteLLM proxy
  (`DEMO_LOCAL_BASE_URL`, default `http://localhost:4000`; `DEMO_LOCAL_MODEL`,
  default `ollama/llama3.1:8b`), mirroring `eval/agent_server.py --local`.

A single shared `SentenceTransformer("all-MiniLM-L6-v2")` is loaded once at
startup and reused across all sessions.

## Storage approach

**In-memory SQLite per session with `StaticPool`.** Each session gets its own
`LongTermStore("sqlite:///:memory:")`. `memory/longterm.py` configures the engine
with `connect_args={"check_same_thread": False}` and `poolclass=StaticPool` when
the URL is in-memory, so the schema/data persist across the sync-endpoint
threadpool workers (an in-memory DB otherwise lives inside one connection). The
per-session temp-file fallback under `_sessions/` was **not** needed — in-memory
works cleanly. Engines are disposed on session drop / idle-TTL sweep.

## Endpoints

All session-scoped endpoints use a signed cookie (Starlette `SessionMiddleware`,
signed via `itsdangerous`) — no id in the path. Send cookies with
`credentials: "include"` (CORS allows `http://localhost:5173`).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | — | `{status:"ok"}` |
| POST | `/api/session` | `{budget?}` (400/800/1200) | `{budget, mode, model, limits}` + sets cookie |
| GET | `/api/session` | — | `{budget, mode, model, message_count, limits}` |
| DELETE | `/api/session` | — | `{status:"ended"}`, drops session + LT engine |
| POST | `/api/chat` | `{message}` | `{messages, context_snapshots, lt_events, reply, latency_ms}` (2 new entries) |
| GET | `/api/timeline` | — | `{messages, context_snapshots, lt_events}` (all) |
| GET | `/api/scripts` | — | `[{name, description}]` |
| POST | `/api/script/{name}` | — | same shape as `/api/chat`, for ALL new messages |

`limits = {max_messages: 50, max_input_chars: 4000}`. Per-session message cap
returns HTTP 429; missing/unknown session returns 403. `slowapi` rate limiting
and a global cost kill-switch are intentionally deferred (see `TODO(hosting)` in
`server.py` / `sessions.py`).

## Snapshot shapes

`context_snapshots[i]`: `{index, role, text, budget_used, budget_max,
budget_pressure, memory_prompt_text, blocks:[{id, content, tier, novelty, decay,
fidelity, token_cost, compression_count, access_count}], latency_ms}`
(`latency_ms` is null on user-message snapshots, set on assistant-message ones).

`lt_events[i]`: `{added:[ltView...], updated:[ltView...], accessed:[id...]}` where
`ltView = {id, content, novelty, fidelity, compression_count, access_count, decay,
is_reconstructed}`. Diff is computed server-side vs. the previous message's LT
view. **Embeddings are never sent to the client.** Computed fields (`token_cost`,
`decay`) are captured as plain values at snapshot time.

## Tests

```bash
.venv/bin/python -m pytest demo/backend/test_smoke.py -q   # offline, no API key
```
