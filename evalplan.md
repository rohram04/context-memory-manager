# Evaluation Plan

How we evaluate the algorithmic memory manager against the primary baseline (real
Letta / MemGPT), and the supporting config/clock changes that make that evaluation
fair, deterministic, and cheap.

---

## Overview & research claim

The core research claim is that replacing FIFO eviction with a novelty-governed,
access-frequency-aware compression/expansion lifecycle keeps high-signal facts alive
that FIFO loses. The evaluation makes this case in two distinct parts, each its own
harness and its own claim.

### Part A — General parity (LoCoMo)
On a realistic, real-world conversational workload (LoCoMo: long multi-session
dialogues with annotated QA), the algorithmic manager should perform **on par** with
Letta. This is the "we don't regress on normal workloads" claim. We do not expect to
*beat* Letta here — we expect comparable answer accuracy at far lower cost.
Harness: `eval/locomo_eval.py`.

### Part B — Recall advantage under overflow (the core claim)
On a context-overflow workload that seeds facts of varying **novelty**, **re-access
frequency**, and **stream position**, the manager should **retain and recall more** of
the high-signal / early facts that an eviction policy loses. This is the headline
research result. Harness: `eval/recall_advantage.py`.

### Honest framing of the metrics
- Against the **naive FIFO simulator** (a strawman with no long-term store), raw
  retention is decisively in our favor — the free smoke shows retention 1.00 vs 0.375.
- Against **real Letta**, which has *unbounded archival memory*, raw "retention" (does
  the token survive anywhere) is high for **both** systems. So vs real Letta the real
  discriminators are: **recall accuracy under overflow**, **cost**, and **latency**.
  Lead with recall and cost-per-correct-answer; keep retention as a secondary metric.

---

## Architecture additions

These changes were made to the core system specifically to support fair, fast,
deterministic evaluation. Defaults preserve the original behavior, so production / CLI
callers that pass no config are unaffected.

### `MemoryConfig` — tunable knobs (`memory/config.py`)
A single dataclass collecting every previously-hardcoded constant from `store.py`,
`ContextManager.py`, and `controller.py`, threaded through the stack so an eval can
sweep them without editing source. Key fields (defaults shown):

| Field | Default | Governs |
|---|---|---|
| `compress_novelty_weight` / `compress_recency_weight` | 0.6 / 0.4 | compression priority formula |
| `fidelity_removal_threshold` | 0.50 | fidelity-triggered removal |
| `promote_alpha` / `promote_beta` / `promote_threshold` | 0.7 / 0.3 / 0.6 | pre-prompt promotion |
| `augment_similarity_threshold` | 0.85 | merge-vs-insert decision |
| `novelty_top_k` / `novelty_hybrid_band` / `novelty_retrieval_top_k` | 5 / (0.35, 0.65) / 5 | novelty scoring |
| `clock_seconds_per_turn` | 0.0 | logical clock (see below) |
| `pressure_thresholds` | {critical 0.90, high 0.75, medium 0.50} | budget-pressure label |

`max_tokens` is intentionally not in the config — it is set per-run by each harness.

### Turn-based logical clock (`memory/clock.py`) — and why it exists
Decay in this system is `e^(-elapsed / decay_constant)`. In a fast batch eval, hundreds
of items are ingested in milliseconds of wall-clock time, so wall-clock `dt ≈ 0` and
**every block's decay_score collapses to ~1.0** — the recency / access-frequency signal
is inert and untestable.

The clock fixes this. With `clock_seconds_per_turn > 0`, the process-global `CLOCK`
switches to *simulated mode*: it starts at a fixed epoch and advances by that many
seconds on each `MemoryController.receive()` (one tick per turn). A fact re-mentioned 2
turns ago is then genuinely "fresher" than one untouched for 50 turns — deterministically
and independent of how fast the eval runs. `clock_seconds_per_turn = 0.0` (the default)
keeps wall-clock behavior for production. The clock is module-global shared state; evals
run one system at a time and reset it per run via `MemoryController.__init__`, so there
is no cross-run contamination.

### `RETRIEVAL_LLM` novelty mode (`memory/novelty.py`)
A new "grounded" novelty mode added as an ablation axis alongside `EMBEDDING` (primary,
model-independent) and `HYBRID`. Instead of dumping all of memory into the prompt (as the
older `score_novelty_llm` does), it retrieves the top-k nearest blocks from context **and**
long-term, and asks the LLM to grade novelty against just those neighbors. This keeps the
prompt bounded and sharpens redundancy/contradiction judgments. `EMBEDDING` is the primary
mode for the main runs; `HYBRID` and `RETRIEVAL_LLM` are paid ablations.

### OpenRouter routing (`llm_client.py`)
All Claude calls go through the Anthropic SDK (`client.messages.create`).
`make_anthropic_client()` points that client at OpenRouter's Anthropic-compatible endpoint
(`https://openrouter.ai/api`) when an OpenRouter key is present, so **both our system and
the self-hosted Letta server route Sonnet through the same provider** — a fair,
matched-model comparison. Key precedence: `OPENROUTER_API_KEY_MANAGER` (manager's calls) →
`OPENROUTER_API_KEY` (shared, also used by Letta) → `ANTHROPIC_API_KEY` (native fallback).
Canonical model ids live here: `OPENROUTER_SONNET = "anthropic/claude-sonnet-4.6"`,
`OPENROUTER_HAIKU = "anthropic/claude-haiku-4.5"`, and Letta's provider-prefixed
`LETTA_OPENROUTER_SONNET = "openrouter/anthropic/claude-sonnet-4.6"`.

---

## Harnesses

All harnesses write JSON to `eval/results/` and print a running cost tally. Run them
from the repo root with the venv active and the keys exported (see Setup).

### `eval/recall_advantage.py` — Part B (recall under overflow)
Seeds tiered facts (high vs low novelty, some re-accessed) into an overflowing stream of
distractors, ingests into each system, then probes recall of every fact at the end.
Reports recall and retention split by **novelty tier**, **stream position**
(early/mid/late thirds), and **re-access**. Grading is opaque-token exact match (free,
deterministic) on the retrieved block (free systems) or the LLM answer (paid systems).

Systems (`--systems`): `ours_stub` (free), `fifo` (free), `ours` (paid), `letta` (paid).

```bash
# Free smoke (no API):
python eval/recall_advantage.py --systems ours_stub,fifo \
  --facts 10 --distractors 50,100,200 --max-tokens 400

# Discriminating paid run (equal usable memory, see Fairness):
python eval/recall_advantage.py --systems ours,letta \
  --facts 10 --distractors 80 --max-tokens 800 \
  --letta-context-window 2336 --trials 1 --max-spend 6
```

Key flags: `--facts`, `--distractors` (comma-separated sweep), `--max-tokens` (our
block budget), `--letta-context-window` (Letta's total window — different unit),
`--trials`, `--clock-seconds-per-turn` (default 600), `--novelty`
(embedding/hybrid/retrieval_llm), `--util-model`, `--query-model`, `--letta-model`,
`--letta-base-url` (default `http://localhost:8283`), `--letta-embedding` (default
`ollama/nomic-embed-text:latest`), `--max-spend` (default 50), `--out`.

### `eval/locomo_eval.py` — Part A (LoCoMo parity)
Ingests each LoCoMo conversation's turns into both systems, probes a sample of its QA
pairs, and grades free-form answers with a cheap Haiku judge (abstention-aware). Reports
accuracy overall and **by LoCoMo category** (multi-hop / temporal / open-domain /
single-hop / adversarial), plus cost. **PAID** — both systems call the LLM.

```bash
python eval/locomo_eval.py --systems ours,letta \
  --n-convs 3 --turn-cap 80 --qa-cap 20 \
  --max-tokens 2000 --letta-context-window 8192 --max-spend 13
```

Key flags: `--n-convs`, `--turn-cap`, `--qa-cap`, `--max-tokens`,
`--letta-context-window` (default 8192), `--model`, `--util-model`, `--judge-model`
(default Haiku), `--letta-model`, `--letta-base-url`, `--letta-embedding`,
`--max-spend`, `--out`.

### `eval/stress_test.py` — Sequential-NIAH (free)
The original needle-in-a-haystack stress test (ours vs FIFO sim, no LLM anywhere). Also
houses the tiered-fact generators (`generate_tiered_facts`, `interleave_tiered`,
`generate_distractors`) that `recall_advantage.py` imports.

```bash
python eval/stress_test.py --needles 10 --distractors 0,25,50,100,200,400,800 --max-tokens 4000
```

### Supporting modules
- `eval/runners.py` — `OursSystem`, the shared wiring of the algorithmic manager for
  evals. Supports `cost_mode="stub"` (free: truncation compress + concat merge,
  embedding novelty, no API) and `cost_mode="live"` (paid: LLM compress/merge, metered).
  Exposes `ingest`, `retained_token` (token survives in context ∪ LT), `retrieve_top1`
  (free), and `llm_probe` (paid).
- `eval/datasets/locomo.py` — downloads and caches `locomo10.json`, flattens sessions to
  chronological `"<speaker>: <text>"` turns, normalizes QA (folds `adversarial_answer`
  into `answer`; coerces string categories to ints).
- `eval/baselines/judge.py` — `make_judge_fn(client, model)` → `(question, gold,
  predicted) -> bool`, one small Haiku call per probe, abstention-aware, defaults False
  on error.
- `eval/baselines/letta_runner.py` — `LettaRunner` (one agent per trial: ingest, query,
  close). Connects to a self-hosted server (`base_url`) or Letta Cloud (`LETTA_API_KEY`);
  metered via the `CostMeter`.

---

## Baselines

### FIFO simulator (`eval/baselines/memgpt_sim.py`) — free strawman
A MemGPT-style FIFO baseline with **no long-term store**. Cheap and instant; useful for
dev iteration and to demonstrate the failure mode FIFO has by construction. Because it has
no archival tier, it loses evicted facts outright — which is why the free smoke shows a
large retention gap (1.00 vs 0.375). Treat this as a naive strawman, not the headline
comparison.

### Real Letta — the primary baseline
Self-hosted Letta in Docker, routing Sonnet through OpenRouter (handle
`openrouter/anthropic/claude-sonnet-4.6`) with Ollama embeddings. Real Letta has
**unbounded archival memory**, so it rarely loses a fact entirely — raw retention stays
high for both systems. The honest implication: against real Letta the discriminators are
**recall accuracy under overflow**, **$ cost**, and **latency**, not raw retention. The
matched-model setup (both sides on Sonnet via OpenRouter, including our compress/merge
util model) keeps the comparison fair — Letta summarizes with its configured model, so we
must too.

---

## Fairness: budget sizing

Our `--max-tokens` and Letta's `--letta-context-window` are **different units** and must
not be set to the same number:

- **Our `max_tokens`** = the managed *memory-block pool* — just the content blocks the
  manager juggles.
- **Letta's `context_window_limit`** = the *total* prompt budget, which includes Letta's
  system prompt + tool definitions (~1000 tokens measured) plus a conversation/response
  buffer.

The chosen framing is **equal usable memory**: pick one usable-memory budget `N`, set ours
to `--max-tokens N` and Letta to `--letta-context-window = N + overhead`, where overhead
≈ 1000 (system/tools) + ~500 (conversation/response buffer) ≈ **1536**.

```
ours:  --max-tokens N
letta: --letta-context-window  ≈  N + 1536
```

Default Part B run: `N = 800` → `--letta-context-window ≈ 2336`, with `facts=10,
distractors≈80` (~90 short items ≈ ~1600 tokens > 800) so the stream overflows `N` and
forces eviction on both sides. The `--letta-context-window` flag exists precisely to keep
this unit difference explicit; do not collapse it onto `--max-tokens`.

---

## Cost

Cost is a **first-class metric**, not just a budget guard (`eval/cost.py`):

- Both systems route Sonnet through OpenRouter (Anthropic pass-through, same list price).
- `CostMeter` accumulates input/output tokens and USD per system. Pricing table: Sonnet
  $3/$15 per M, Haiku $1/$5 per M. Provider-prefixed model ids are stripped before lookup.
- `CostMeter.wrap_client` meters **every** of our LLM calls (compress, merge, novelty-LLM,
  probe) — not just probes — so reported cost is complete. `LettaRunner` meters each Letta
  turn via `add_letta_response`.
- `--max-spend` is a hard cap: crossing it raises `BudgetExceeded`, the harness stops
  gracefully and writes partial results. Total budget constraint for the whole eval is
  **< $50**.
- Report **recall-per-dollar** alongside accuracy. Letta dominates both cost (~$0.014/turn)
  and latency (sequential agent turns, ~5–8s each); our system makes far fewer calls per
  turn. Standout framing: ours @ Sonnet (~$8) is cheaper than a Letta @ Haiku run would be,
  at equal-or-better recall.

---

## Tuning methodology

Tuning runs entirely on the **free Tier-1 path** ($0): `OursSystem(cost_mode="stub")` with
embedding novelty and local embeddings — no API calls. (A `eval/tune.py` harness to
automate this is being built.)

- **Dev/test seed split** — tune on dev seeds (e.g. 1..6), report on held-out test seeds
  (e.g. 101..106). Never report on the seeds you tuned on.
- **One-factor-at-a-time** sweep from current defaults:
  - `compress_novelty_weight` {0.4, 0.6, 0.8}  (`compress_recency_weight = 1 − novelty_weight`)
  - `fidelity_removal_threshold` {0.4, 0.5, 0.6}
  - `promote_threshold` {0.5, 0.6, 0.7}
  - `augment_similarity_threshold` {0.8, 0.85, 0.9}
  - `clock_seconds_per_turn` {300, 600, 1200}
- **Objective = in-context retention by novelty tier**, NOT raw retention. Raw retention is
  ~100% for our system (LT is unbounded) and can't discriminate configs. Instead measure
  whether each seeded fact survives in the *context store* (`cm._store.all_blocks()`,
  excluding LT) at probe time, and maximize high-novelty / early-fact in-context retention
  and the high-minus-low-novelty gap — this directly exercises the novelty governance the
  params control.
- Output a single best frozen `MemoryConfig` (plus dev/test scores JSON) to feed the later
  paid runs. The novelty-mode ablation (embedding vs hybrid vs grounded) is a small *paid*
  add-on since the latter two call the LLM.

---

## Setup / prerequisites

### Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install letta-client     # only needed for the real-Letta baseline
```

### Environment variables (`.env`)
- `OPENROUTER_API_KEY` — shared OpenRouter key; also used by the Letta server.
- `OPENROUTER_API_KEY_MANAGER` — optional separate key for the manager's own calls
  (takes precedence over `OPENROUTER_API_KEY`).
- `ANTHROPIC_API_KEY` — native Anthropic fallback if no OpenRouter key is set.
- `LETTA_API_KEY` — only for Letta Cloud (not needed when using the self-hosted server
  via `--letta-base-url`).

**The keys must be exported into the shell that runs the harness** (the harness reads
`os.environ`); a `.env` file alone is not loaded automatically.

```bash
export OPENROUTER_API_KEY=sk-or-...
export OPENROUTER_API_KEY_MANAGER=sk-or-...   # optional
```

### Self-hosted Letta server (Docker)
Runs Letta locally, routing Sonnet through OpenRouter and using Ollama for embeddings via
`host.docker.internal`. The default `--letta-base-url http://localhost:8283` targets it.

```bash
docker run -d --name letta \
  -p 8283:8283 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -e OLLAMA_BASE_URL="http://host.docker.internal:11434" \
  letta/letta:latest
```

(Ensure Ollama is running on the host with the `nomic-embed-text` model pulled.)

---

## Status

### Verified
- **Free ($0):** foundation (`MemoryConfig`, turn clock, config threaded through the
  stack, `augment` refreshes recency, `RETRIEVAL_LLM` novelty mode); all eval modules
  (`cost.py`, `runners.py`, `recall_advantage.py`, `locomo_eval.py`, dataset/judge/Letta
  modules); unit tests (`tests/test_clock_config.py` 6/6, `tests/test_novelty_retrieval.py`
  3/3); backward-compat confirmed (no-config path reproduces legacy constants). Free Part B
  smoke: retention ours 1.00 vs FIFO 0.375 (by tier 1.0/1.0 vs 0.5/0.25), recall@1 0.75 vs
  0.25 → `eval/results/free_smoke.json`.
- **Paid plumbing (~$0.40 smoke):** both systems round-trip Sonnet through OpenRouter
  end-to-end; metering is complete (ours compress/merge + all Letta turns) and bounded by
  `--max-spend`. Surfaced and fixed two bugs: Letta `context_window` unit mismatch (added
  `--letta-context-window`) and incomplete metering (added `CostMeter.wrap_client`).

### Pending
- Build and run the free `eval/tune.py` sweep; freeze the best `MemoryConfig`.
- Discriminating paid Part B run (equal-usable-memory: `--max-tokens 800`,
  `--letta-context-window ≈ 2336`, facts 10, distractors ≈ 80, trials 1) with the frozen
  config. Budget ~$3–6 and ~30–45 min.
- Part A `eval/locomo_eval.py` (3 convs), then `eval/plot_results.py` figures.
- Watch `--max-spend` against the $50 cap throughout.
