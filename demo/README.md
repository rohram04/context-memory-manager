# Memory Manager — Interactive Demo

An IDE-style web app for *watching* the novelty-governed memory lifecycle happen.
You chat with the algorithmic memory agent and, by clicking any message, inspect
the exact state of the context window and the long-term store at that point — with
the per-message changes (insert / compress / evict / promote / augment) highlighted.
Compression turns full blocks into **stubs** that point back to their long-term
copies, and a stub's "→ LT" button jumps straight to that source block ("go to
definition"). The whole point is to make the algorithm visible turn by turn, on a
small token budget that forces the lifecycle to fire within a few messages.

The demo focuses entirely on **our system** (no Letta/baseline comparison here).
Mode is fixed to algorithmic; budget is chosen from {400, 800, 1200}, default 400.

## Layout

```
demo/
├── backend/    FastAPI app (sessions, per-message snapshots, LT deltas)
└── frontend/   Vite + React + TypeScript SPA (two-pane inspector)
```

## Prerequisites

- **Python** with the repo dependencies installed (from the repo root):
  ```bash
  pip install -r requirements.txt
  ```
- **Node** (18+) for the frontend.

## Ports

The backend defaults to **port 8000**. Port **8080 is intentionally avoided** — it
commonly conflicts with other local tooling. The Vite dev server runs on **5173**
and proxies `/api` + `/healthz` to `http://localhost:8000`, so the SPA stays
same-origin with the backend and the signed httpOnly session cookie just works.
If you change the backend port, update `frontend/vite.config.ts` to match.

## Run the backend (from the repo root)

```bash
uvicorn demo.backend.server:app --port 8000
# add --reload for dev
```

LLM client (`llm_client.make_anthropic_client()`):

- **API keys via `.env`** — the repo `.env` is read by the client. With
  `OPENROUTER_API_KEY` set, calls route through OpenRouter (model
  `anthropic/claude-sonnet-4.6`); export it before launching so a real run works:
  ```bash
  set -a; source .env; set +a
  uvicorn demo.backend.server:app --port 8000
  ```
  Native Anthropic (`ANTHROPIC_API_KEY`) also works.
- **No-cost local option** — set `DEMO_LOCAL=1` to route LLM calls through a local
  LiteLLM proxy instead (no API spend):
  ```bash
  DEMO_LOCAL=1 uvicorn demo.backend.server:app --port 8000
  # tune with DEMO_LOCAL_BASE_URL (default http://localhost:4000)
  #     and DEMO_LOCAL_MODEL    (default ollama/llama3.1:8b)
  ```

A single shared `SentenceTransformer("all-MiniLM-L6-v2")` is loaded once at startup
and reused across all sessions (embeddings power novelty scoring). Long-term memory
is an in-memory SQLite store, one per session, so sessions are fully ephemeral.

Health check: `curl -s localhost:8000/healthz` → `{"status":"ok"}`.

See `backend/README.md` for the full endpoint list and snapshot shapes.

## Run the frontend

```bash
cd demo/frontend
npm install
npm run dev          # opens http://localhost:5173, proxies /api → :8000
```

With no backend running, the SPA falls back to built-in fixtures so the UI still
renders a sample lifecycle (set `VITE_USE_FIXTURES=1` to force this).

## Walkthrough

1. **Pick a budget** (400 / 800 / 1200) — this creates a fresh session. 400 makes
   the lifecycle fire within a handful of messages.
2. **Chat**, or run a **pre-baked script** from the script bar:
   - **early_fact** — seeds one memorable "needle" fact early.
   - **flood** — sends several distractor messages to drive budget pressure.
   - **recall** — probes for the early seeded fact.
   - **full_demo** — end-to-end: early fact → flood → mid fact → flood → recall.
3. **Click any message** in the chat list (arrow keys also navigate). The left
   **Inspector** updates to that message's state:
   - **Context** tab — memory-block cards with novelty / decay / fidelity bars,
     token cost, tier badge (full | stub), compression and access counts. Cards
     diff-highlight the net change *that message* (green = inserted, red = removed,
     amber = compressed, blue = augmented). A **raw text** toggle swaps the cards
     for the exact `MEMORY STATUS` prompt the model saw. A **stub** card's "→ LT"
     button jumps to its source block in the Long-term tab.
   - **Long-term** tab — the LT store folded to the selected message, with blocks
     added / updated / accessed *this* message highlighted.
   - **split** toggle shows Context and Long-term side by side.

Animations are tied to the lifecycle (compression visibly shrinks a card,
promotion springs a block back into context) and degrade to instant under
`prefers-reduced-motion`.
