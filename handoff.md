# Handoff: Agent Server for MemoryAgentBench

## What You're Building

A FastAPI server (`eval/agent_server.py`) that wraps the project's `Agent` class and exposes a single `/send_message` endpoint. MemoryAgentBench will call this endpoint instead of importing a Python adapter directly — the memory system stays completely isolated from the benchmark code.

---

## Project Overview

A biologically-inspired LLM memory manager that replaces MemGPT's FIFO eviction with a novelty-scored, access-frequency-governed compression/eviction lifecycle. The goal is to beat MemGPT on MemoryAgentBench, which already includes Letta (the current MemGPT) as a baseline.

**Two-tier architecture:**
- **Context window** — active `CacheBlock`s in a priority heap, ordered by compressibility (novelty + decay + token cost)
- **Long-term store** — SQLite-backed `LongTermBlock`s, searched via vector similarity

**Two memory modes (both must work with the server):**
- `algorithmic` — `MemoryController` handles insert/compress/evict/promote automatically each turn; LLM never sees memory tools
- `llm` — Claude manages its own memory via structured function calls (`store`, `compress`, `evict`, `promote`, `augment`, `query_lt`, `update_novelty`)

---

## File Structure

```
MemoryManager/
├── agent.py              # Agent class — the main entry point
├── controller.py         # MemoryController — algorithmic scheduling
├── ContextManager.py     # Primitive operations (compress, promote, evict, etc.)
├── cli.py                # Reference for how to construct an Agent
├── functions/
│   ├── llm_fns.py        # make_compress_fn, make_merge_fn (LLM-based)
│   └── memory_tools.py   # Tool schemas + dispatch for LLM mode
├── memory/
│   ├── block.py          # CacheBlock + LongTermBlock schemas
│   ├── store.py          # ContextStore — heap, token budget, find_similar
│   ├── longterm.py       # LongTermStore — SQLite backend
│   └── novelty.py        # NoveltyMode enum + scorer factory
├── .env                  # Contains API-Key=sk-ant-...
└── eval/
    └── agent_server.py   # ← BUILD THIS
```

---

## How to Construct an Agent

This is the exact pattern used in `cli.py`:

```python
import anthropic
from agent import Agent, MemoryMode
from ContextManager import ContextManager
from controller import MemoryController
from functions.llm_fns import make_compress_fn, make_merge_fn
from memory.longterm import LongTermStore
from memory.novelty import NoveltyMode
from memory.store import ContextStore

# Load API key from .env (key name is "API-Key", not "ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key="<value of API-Key from .env>")

store = ContextStore(max_tokens=4000)          # token budget for context window
lt_store = LongTermStore("sqlite:///memory.db") # long-term store path
cm = ContextManager(store, lt_store, embedding_model="all-MiniLM-L6-v2")

controller = MemoryController(
    cm,
    compress_fn=make_compress_fn(client, "claude-haiku-4-5-20251001"),
    merge_fn=make_merge_fn(client, cm, "claude-haiku-4-5-20251001"),
)

agent = Agent(
    controller,
    client,
    model="claude-sonnet-4-6",
    mode=MemoryMode.ALGORITHMIC,      # or MemoryMode.LLM
    novelty_mode=NoveltyMode.EMBEDDING,
    novelty_model="claude-haiku-4-5-20251001",
)
```

**The only public method you need:**
```python
response_text: str = agent.chat(user_message: str)
```

`chat()` handles the full turn: memory insert/compress/promote, LLM call, memory update. It appends to `agent._messages` (the conversation history) so state accumulates across calls automatically.

---

## MemoryAgentBench Interface

The benchmark calls `AgentWrapper.send_message()` in `agent.py` (repo root of MemoryAgentBench). The endpoint must match this contract:

**Request:**
```json
{
  "message": "string — either raw context text or a question",
  "memorizing": true,
  "query_id": 0,
  "context_id": 42
}
```

**Response:**
```json
{
  "output": "string — agent's reply",
  "input_len": 128,
  "output_len": 64,
  "memory_construction_time": 1.23,
  "query_time_len": 0.45
}
```

- `memorizing=true` → benchmark is feeding context (raw text to memorize). Call `agent.chat(message)` and discard or return the response — the memory storage is the point.
- `memorizing=false` → benchmark is asking a question. Call `agent.chat(message)` and return the answer in `output`.
- `memory_construction_time` → time taken when `memorizing=true`
- `query_time_len` → time taken when `memorizing=false`
- `input_len` / `output_len` → token counts (use `len(message.split())` as an approximation if needed, or tiktoken)

---

## Critical: State Management via context_id

Each benchmark example feeds 60–100 sequential messages to the same agent before resetting. The `context_id` parameter identifies which example is currently running.

The server must:
1. **Keep a dict of live `Agent` instances keyed by `context_id`**
2. **Create a new `Agent` when a new `context_id` is seen**
3. **Reuse the existing `Agent` for subsequent messages with the same `context_id`**
4. **Clean up (delete) the agent after the context is done** — or at minimum don't let them accumulate indefinitely

Each agent needs its own SQLite file to avoid cross-context contamination:
```python
lt_store = LongTermStore(f"sqlite:///eval_context_{context_id}.db")
```
Delete the file when the context is done.

---

## Environment

- **Python:** 3.12.13 (pyenv), venv at `.venv/`
- **Run with:** `.venv/bin/python`
- **API key:** stored in `.env` as `API-Key=sk-ant-api03-...` — load with `python-dotenv` or read manually
- **FastAPI and uvicorn** are already installed in the venv (came in with litellm)
- **Port:** use 8080 (4000 is taken by LiteLLM proxy)

---

## Two Modes to Support

The server should accept a `mode` query param or be configured at startup:

- `?mode=algorithmic` (default) — recommended for initial eval run
- `?mode=llm` — Claude manages its own memory via tools

Start with `algorithmic` mode since it's simpler and more reliable. `llm` mode works correctly with Claude but costs more (more API calls per turn).

---

## What NOT to Do

- Don't share `Agent` instances across `context_id`s — state will bleed between benchmark examples
- Don't use a single shared SQLite file — concurrent contexts will corrupt each other
- Don't reset `agent._messages` between memorizing and querying phases — they're the same continuous conversation
- Don't skip the `memorizing=true` phase — the agent needs to ingest the context before it can answer questions
