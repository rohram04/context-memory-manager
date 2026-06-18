# LLM Memory Manager — CLAUDE.md

## Project Overview

A biologically-inspired LLM memory manager that replaces MemGPT's naive FIFO eviction system with a novelty-scored, access-frequency-governed, bidirectional compression/expansion lifecycle. The system operates over the standard two-tier LLM memory architecture (context window + long-term store) and gives the LLM itself intelligent tools to manage its own memory.

## Core Research Contribution

**Primary claim:** The first system to replace FIFO eviction in LLM memory management with a bidirectional, novelty-governed, compression-expansion lifecycle using stubs as persistent context anchors.

**Baseline to beat:** MemGPT / Letta (arxiv: 2310.08560)

---

## Architecture

### Two Tiers (Standard)
```
CONTEXT WINDOW     →  full blocks + stubs + pointers (in-prompt)
LONG-TERM MEMORY   →  source of truth, own compression lifecycle (vector store)
```

### Memory Block Schema

```python
# In-context representation
CacheBlock {
  id: str
  content: str                    # current content at whatever compression state
  original_embedding: List[float] # embedding of original full block, set at creation, never updated
  fidelity: float                 # cosine_similarity(embed(content), original_embedding)
  compression_count: int          # number of compression passes applied
  pointer_to_lt_id: str | None    # set on first compression pass
  novelty_score: float            # 0.0 - 1.0
  access_count: int
  last_accessed: datetime
  decay_score: float              # time-weighted retention score
  token_cost: int                 # tokens this block consumes in context
}

# Long-term store
LongTermBlock {
  id: str
  content: str                    # current content at whatever compression state
  original_embedding: List[float] # embedding of original full block, never updated
  embedding: List[float]          # embedding of current content, for similarity retrieval
  fidelity: float                 # cosine_similarity(embedding, original_embedding)
  compression_count: int
  novelty_score: float
  access_count: int
  last_accessed: datetime
  decay_score: float
  is_reconstructed: bool          # true if rebuilt via LLM hallucination
}
```

---

## Memory Lifecycle

### Compression (in context)
No fixed tiers. A block can be compressed many times. The LLM or algorithm determines when and how aggressively to compress each pass.
1. Queue nominates block for a compression pass
2. LLM summarizes current content into a shorter version
3. Fidelity recomputed: `cosine_similarity(embed(new_content), original_embedding)`
4. Token cost updated, block re-sorted in heap
5. On first compression pass: current content copied to LT before compressing, pointer set
6. If fidelity drops below threshold → fidelity-triggered removal (see below)

### Fidelity-Triggered Removal
When fidelity drops below threshold after any compression pass:
1. Block removed from context entirely
2. Block exits the compression heap
3. LT copy remains as source of truth
4. Recovered via pre-prompt LT similarity search if needed

### Eviction — Full (no compression)
Used when a block is unlikely to be needed again soon.
1. Block copied to LT in current state
2. Block removed from context and heap entirely
3. Recovered via pre-prompt LT similarity search if needed

### Augmentation (merge into existing block)
When new content arrives and a semantically similar block already exists in context:
1. Embed incoming content
2. `find_similar` scans context blocks' `original_embedding`s (cosine sim, threshold 0.85)
3. If match found: LLM-synthesized merge of existing + new content (eager, at write time)
4. Merged content replaces block content, `original_embedding` updated, `fidelity` reset to 1.0
5. Heap re-sorted via `update_priority`
6. If no match: new `CacheBlock` created and inserted

Pre-prompt promotion runs before augmentation each turn so that relevant LT blocks are in context and can be augmentation candidates before the similarity check runs.

### Promotion (LT → Context)
Applies regardless of how a block left context.
1. Pre-prompt embedding similarity search over LT
2. Score candidates: `0.7 × similarity + 0.3 × decay_score`, promote above 0.6 threshold
3. Full content retrieved from LT, inserted into context (or stub updated in place)
4. Fidelity and compression_count carried over from LT record
5. LT access count incremented, block re-enters compression heap

### LT Compression (rare)
LT blocks have their own independent, slower compression lifecycle.
1. LT blocks decay at a slower rate than context blocks
2. Very low access + low novelty → LT block gets a compression pass
3. Fidelity recomputed against original_embedding
4. If later promoted with very low fidelity: LLM reconstruction required
5. Reconstructed blocks flagged `is_reconstructed: true`

### Access Decay Function
```
retention = e^(-time_since_access / decay_constant)
```
- Context decay constant: fast (hours/turns)
- LT decay constant: slow (days/sessions)

---

## Novelty Scoring

A score assigned to each block at creation time, decaying slowly over time. It captures how semantically distinct, surprising, or significant a memory block is relative to what the system already knows — drawing on signals from meaning, context, and relevance to the user.

**Novelty acts as a compression floor** — high-novelty blocks resist compression regardless of access frequency. Specific scoring dimensions and weights are left as an empirical design decision to be determined during implementation and evaluation.

---

## Compression Fidelity

Fidelity measures how much semantic meaning remains in a block relative to its original full content. Computed after every compression pass:

```
fidelity = cosine_similarity(embed(current_content), original_embedding)
```

`original_embedding` is computed once at block creation and never updated. Every fidelity check measures total information loss from source — not just drift from the last compression pass.

High fidelity — current content preserves the semantic essence of the original.
Low fidelity — significant meaning has been lost across compression passes.

A block can be compressed many times before fidelity drops below threshold. Fidelity acts as a **removal trigger** — once it falls below threshold after any compression pass, the block is removed from context and exits the heap. It lives in LT only and is recovered via pre-prompt similarity search if needed.

---

## Compression Queue

A max-heap ordered by a continuous priority function. Re-sorted each turn as access patterns and decay scores change.

### Priority Function

```python
compressibility = 0.6 * (1 - novelty_score) + 0.4 * (1 - decay_score)
priority = token_cost * compressibility
```

Additive combination of novelty and decay (not multiplicative) — prevents either signal from zeroing out priority entirely. The 60/40 split reflects novelty as the primary compression-resistance signal. `token_cost` scales priority so large blocks are compressed before small ones at equal compressibility.

### Signals

| Signal | Effect |
|---|---|
| Novelty score | High novelty pushes block down the queue (resist compression) |
| Decay score | Low access frequency pushes block up the queue (compress sooner) |
| Token cost | High token cost pushes block up the queue (more valuable to compress) |
| Fidelity | Below threshold (0.50) after any compression pass: block **removed from context**, exits heap |

### Queue Behavior

- Compression priority is a continuous function of novelty, decay, and token cost
- Each turn the top of the queue triggers one compression pass, not a fixed tier jump
- Fidelity is the only exit trigger — blocks are compressed repeatedly until fidelity drops below threshold
- Fidelity always measured against `original_embedding`, not the previous version
- Blocks move up or down dynamically as access frequency and decay scores update
- Novelty decays slowly over time — a once-high-novelty block may eventually become compressible
- Fully evicted blocks exit the heap entirely
- Compressed blocks remain in the heap and may be compressed again on future turns

---

## The LLM Controller

The LLM manages its own memory via function calls. Your code provides:
- Token budget metadata in the system prompt
- Memory block metadata (novelty, access count, decay score)
- A set of memory management functions

### System Prompt Injection (each turn)
```
=== MEMORY STATUS ===
Token budget: {used} / {max} ({pct}% full)
Context blocks: {n_full} full, {n_stubs} stubs
Long-term blocks available: {n_lt}
Budget pressure: {low|medium|high|critical}

[MEMORY BLOCKS]
{block_id} | tier: full | novelty: 0.87 | access: 12 | decay: 0.94
  "User prefers concise explanations. Discussed in session 3."

{block_id} | tier: stub | novelty: 0.23 | access: 1 | decay: 0.12
  [STUB] Summary of initial project setup discussion → LT:{lt_id}
...
===================
```

### LLM Tool Surface (`ContextManager` primitives)

The LLM is given access to `ContextManager` only — not `MemoryController`. This exposes three action primitives and two query/update tools:

```python
compress(block_id: str, compress_fn, embed_fn) -> CacheBlock | None
# Compress one specific block. Creates LT entry on first pass, sets pointer_to_lt_id.
# Removes block from context if fidelity drops below threshold after compression.

promote(lt_block_id: str, compress_fn, embed_fn) -> CacheBlock | None
# Promote a long-term block back into context.
# Updates existing stub in place, or creates a new block if fully evicted.

evict(block_id: str, embed_fn) -> CacheBlock | None
# Evict a block to LT and remove from context immediately (no compression pass).
# Creates LT entry if none exists.

query_lt_by_similarity(query: str, top_k: int) -> List[LTBlock]
# Embeds query, runs vector similarity search over LT store.

update_novelty(block_id: str, novelty: float) -> None
# Set novelty score directly (0.0–1.0) — affects compression priority.

score_novelty(embedding: list[float], top_k: int = 5) -> float
# Score how novel an embedding is vs existing context + LT. Returns 1 - mean_top_k_cosine_sim.
# Used to assign novelty_score at block creation time.
```

The algorithmic layer (`MemoryController`) handles routine compression and promotion automatically each turn. The LLM uses these tools for targeted interventions the algorithm cannot make — e.g., promoting a block it knows is relevant to the current task, evicting something it recognises as irrelevant, or updating novelty on a block that just became significant.

---

## Pre-Prompt Pipeline (each turn)

```
1. Receive incoming user/assistant message → MemoryController.receive()
     a. Embed the message
     b. pre_prompt_promote: LT similarity search, score = 0.7×sim + 0.3×decay,
        promote blocks above 0.6 threshold via ContextManager.promote()
     c. insert_or_augment: find_similar in context (cosine sim, threshold 0.85),
        merge via merge_fn if match found, else insert new block
2. Build system prompt → ContextManager.build_memory_prompt()
3. Fire LLM call — model may call compress/promote/evict via tool use
4. Receive LLM response → MemoryController.receive() (stores assistant turn)
5. Post-turn: run proactive compression_tick() calls at high/critical pressure
```

Promotion score: `α × similarity + β × decay_score` (α=0.7, β=0.3, threshold=0.6)

---

## Compression Mechanism

**Text summarization only** (not dimensionality reduction, not quantization).

- Compression = LLM-generated summary of current block content
- A block can be compressed many times — no fixed tiers, continuous lifecycle
- After each pass: fidelity recomputed against `original_embedding`, token cost updated, block re-sorted in heap
- `original_embedding` is set once at block creation and never changes
- On first compression pass: full content copied to LT before compressing, pointer set
- LT maintains its own copy compressed independently on a slower decay schedule

---

## Evaluation

### Baselines
```
Baseline A  →  Vanilla RAG (no memory management)
Baseline B  →  MemGPT with FIFO (primary comparison)
System A    →  This system, algorithmic controller
System B    →  This system, LLM decides eviction
System C    →  This system, hybrid (algorithm nominates, LLM confirms)
```

### Key Metrics
- **Memory fidelity** — recall accuracy of facts stored earlier in conversation
- **Token efficiency** — useful content tokens / total context tokens
- **Novelty retention** — do high-novelty blocks survive longer than low-novelty?
- **Reconstruction rate** — how often does LT compression force LLM reconstruction?
- **Compression fidelity** — cosine similarity between full block and stub embeddings
- **Latency** — cost of pre-prompt embedding pass vs baseline

### Key Experiment
Long conversation stress test: multi-session conversation that deliberately overflows context many times, seeding facts of varying novelty and access frequency. Probe for recall at end. MemGPT loses high-novelty facts to FIFO. This system should not.

### Why MemGPT's FIFO fails on these benchmarks
MemGPT evicts the oldest content first regardless of importance. The shared failure mode across all benchmarks below: **a fact established early in a long conversation must be recalled much later**. FIFO systematically loses those early facts. Novelty scoring keeps high-signal facts alive regardless of age.

### Tier 1 — Primary benchmarks (directly evaluate the core claim)

**LongMemEval** — ICLR 2025 | arxiv:2410.10813 | https://github.com/xiaowu0162/LongMemEval
500 hand-curated questions over multi-session user-assistant histories (up to 1.5M tokens). Five subtasks:
- Single-session information extraction (baseline — both systems pass)
- Multi-session reasoning — early facts evicted by FIFO before they're needed
- Temporal reasoning — time-ordered facts span many sessions; decay alone doesn't evict by age
- Knowledge updates — old fact must survive to be updated; FIFO evicts it silently
- Abstention — fidelity tracking signals when content is unreliable; FIFO has no such signal

**LoCoMo** — Snap Research 2024 | arxiv:2402.17753 | https://github.com/snap-research/locomo
50 long conversations, ~300 turns, up to 35 sessions, ~9K tokens each. Tasks: single-hop factual recall, multi-hop inference, temporal understanding, open-domain QA. The 35-session depth is the best existing analogue to the custom stress test.

**MemoryAgentBench** — ICLR 2026 | arxiv:2507.05257 | https://github.com/HUST-AI-HYZ/MemoryAgentBench
Four competencies: accurate retrieval, test-time learning, long-range understanding, conflict resolution. "Conflict resolution" and "test-time learning" are strong differentiators — novelty scoring upweights surprising/contradictory new information; FIFO evicts the conflicting old fact before resolution is possible.

### Tier 2 — Strong secondary benchmarks

**Sequential-NIAH** — arxiv:2504.04713
Needle-in-a-Haystack adapted for memory systems. Plant N facts, flood with distractors, query each. 8K–128K context, 14,000 samples. FIFO predictably loses the oldest needles. Cleanest controlled test of eviction policy — a synthetic version of the custom stress test.

**EvolMem** — arxiv:2601.03543
Multi-session dialogue memory. Tests how memory evolves as facts become outdated or are corrected. Relevant to the `update_novelty()` path and fidelity-triggered removal.

### Recommended eval implementation order
```
eval/
├── stress_test.py         ← custom; mirrors LoCoMo structure; primary paper contribution
├── niah_memory.py         ← Sequential-NIAH; cleanest controlled FIFO comparison
├── longmemeval.py         ← LongMemEval-S; multi-session + temporal subtasks
├── locomo.py              ← LoCoMo subset; multi-hop cross-session recall
└── baselines/
    ├── memgpt_sim.py      ← FIFO baseline (already planned)
    └── rag.py             ← vanilla RAG baseline (already planned)
```
Minimum viable eval for the paper's core claim: stress_test + Sequential-NIAH. LongMemEval-S and LoCoMo add credibility as established community benchmarks.

---

## File Structure

```
memory_manager/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── ContextManager.py     # primitive operations (compress, promote, evict, insert_or_augment,
│                         # augment, update_novelty, score_novelty) — LLM tool surface
├── controller.py         # MemoryController — algorithmic scheduling layer
│                         # (compression_tick, pre_prompt_promote, receive)
├── agent.py              # main loop, API calls, turn management
├── memory/
│   ├── block.py          # CacheBlock + LongTermBlock schemas
│   ├── store.py          # ContextStore — heap, token budget, find_similar
│   ├── longterm.py       # LongTermStore — SQLite/pgvector dual backend
│   └── novelty.py        # LLM-based novelty scorers (score_novelty_llm, score_novelty_hybrid)
│                         # embedding-based scorer lives on ContextManager.score_novelty()
├── functions/
│   └── memory_tools.py   # LLM tool wrappers — thin shims over ContextManager primitives
├── eval/
│   ├── stress_test.py    # long conversation overflow test
│   ├── fidelity.py       # compression fidelity measurement
│   └── baselines/
│       ├── rag.py
│       └── memgpt_sim.py
└── experiments/
    └── results/
```

---

## Tech Stack

- **Language:** Python
- **LLM API:** Anthropic (claude-sonnet-4-6) or OpenAI
- **Token counting:** tiktoken / Anthropic token count endpoint
- **Embeddings:** sentence-transformers or OpenAI text-embedding-3-small
- **Storage (default):** SQLite + sqlite-vec — zero setup, matches MemGPT's default install for fair real-world comparison
- **Storage (paper comparison):** PostgreSQL + pgvector — matches MemGPT paper stack; identical storage means storage performance is not a variable
- **Abstraction layer:** SQLAlchemy — handles all standard CRUD on both backends; dialect-aware `similarity_search` handles the vector query difference

### Long-Term Memory Store — SQLAlchemy (dual backend)

SQLAlchemy handles all standard CRUD on both backends. The only backend-specific code is `similarity_search` (~15 lines), selected at init time via the connection URL dialect:

```python
class LongTermStore:
    def __init__(self, url: str):
        # url = "sqlite:///memory.db"  or  "postgresql://..."
        self._engine = create_engine(url)
        self._dialect = self._engine.dialect.name  # "sqlite" or "postgresql"

    def similarity_search(self, query_vec: list[float], top_k: int) -> list[tuple[LongTermBlock, float]]:
        # returns (block, similarity_score) pairs, highest similarity first
        if self._dialect == "sqlite":
            # cosine similarity over embedding column
            ...
        else:
            # pgvector <=> operator, returns 1 - distance as similarity
            ...
```

**Default: SQLite + sqlite-vec** (no Docker, no server — matches MemGPT's default install)

Setup:
```python
import sqlite_vec
conn = sqlite3.connect("memory.db")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
```

Schema:
```sql
CREATE TABLE lt_blocks (
    id                  TEXT PRIMARY KEY,
    content             TEXT,
    fidelity            REAL    NOT NULL DEFAULT 1.0,
    compression_count   INTEGER NOT NULL DEFAULT 0,
    novelty_score       REAL    NOT NULL,
    access_count        INTEGER NOT NULL DEFAULT 0,
    last_accessed       TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    is_reconstructed    INTEGER NOT NULL DEFAULT 0
);

-- sqlite-vec virtual table for ANN search
CREATE VIRTUAL TABLE vec_lt_blocks USING vec0(
    id                  TEXT PRIMARY KEY,
    embedding           float[1536],
    original_embedding  float[1536]
);
```

Similarity retrieval (SQLite):
```sql
SELECT m.id, m.content, m.novelty_score, m.access_count, v.distance
FROM vec_lt_blocks v
JOIN lt_blocks m ON m.id = v.id
WHERE v.embedding MATCH ? AND k = ?
ORDER BY v.distance;
```

**Paper comparison: PostgreSQL + pgvector** (matches MemGPT paper stack)

Setup:
```
docker run -e POSTGRES_PASSWORD=pass -p 5432:5432 pgvector/pgvector:pg16
```

Schema:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE lt_blocks (
    id                  TEXT PRIMARY KEY,
    content             TEXT,
    original_embedding  vector(1536),
    embedding           vector(1536),
    fidelity            FLOAT   NOT NULL DEFAULT 1.0,
    compression_count   INT     NOT NULL DEFAULT 0,
    novelty_score       FLOAT   NOT NULL,
    access_count        INT     NOT NULL DEFAULT 0,
    last_accessed       TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_reconstructed    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX ON lt_blocks USING hnsw (embedding vector_cosine_ops);
```

Similarity retrieval (PostgreSQL):
```sql
SELECT id, content, novelty_score, access_count,
       embedding <=> $1 AS distance
FROM lt_blocks
ORDER BY embedding <=> $1
LIMIT $2;
```

---

## Key Papers (Related Work)

| Paper | Relevance |
|---|---|
| MemGPT (Packer et al., 2023) — arxiv:2310.08560 | Primary baseline |
| RAG (Lewis et al., 2020) — arxiv:2005.11401 | Retrieval foundation |
| CoALA (Sumers et al., 2023) — arxiv:2309.02427 | Theoretical framework |
| Titans (Behrouz et al., NeurIPS 2025) — arxiv:2501.00663 | Latent space stretch goal |
| A-MAC (2026) | Novelty scoring prior art |
| ACT-R-inspired LLM memory (HAI 2025) | Decay function prior art |

---

## Novel Contributions Over MemGPT

1. Replaces FIFO with novelty + access-frequency governed eviction
2. Stubs as an optional caching optimization — full eviction is valid, LT similarity search handles rediscovery
3. Bidirectional lifecycle — same signal governs compression and expansion
4. Pre-prompt embedding similarity for proactive promotion
5. Novelty as a continuous compression resistance signal — no fixed tiers, governs compression frequency
6. Independent compression lifecycles for context and LT memory
7. Fidelity tracked against original embedding across all compression passes — measures total information loss, not just single-pass drift

---

## Build Order

- [x] `memory/block.py` — CacheBlock + LongTermBlock schemas
- [x] `memory/store.py` — ContextStore: heap, token budget, find_similar, next_candidate
- [x] `memory/longterm.py` — LongTermStore: SQLite/pgvector dual backend, scored similarity_search
- [x] `ContextManager.py` — primitives: compress, promote, evict, insert, insert_or_augment, build_memory_prompt
- [x] `controller.py` — MemoryController: compression_tick, pre_prompt_promote, receive
- [x] `functions/memory_tools.py` — thin LLM tool wrappers over ContextManager primitives
- [x] `agent.py` — main loop, Anthropic API calls, turn management
- [x] `ContextManager.score_novelty()` — embedding-based novelty scoring
- [x] `memory/novelty.py` — LLM-based (`score_novelty_llm`) and hybrid (`score_novelty_hybrid`) scorers
- [ ] `eval/stress_test.py` — key experiment
- [ ] Baseline implementations
- [ ] Full evaluation suite