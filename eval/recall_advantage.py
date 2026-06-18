"""Part B — recall-advantage stress test: algorithmic manager vs. MemGPT/Letta.

Seeds facts of varying NOVELTY (high/low tier) and RE-ACCESS frequency into an
overflowing stream of distractors, then probes recall of every fact at the end.
Reports recall split by novelty tier, stream position, and re-access — the axes
where novelty-governed retention should beat FIFO-style eviction.

Systems (--systems):
  ours_stub  free  — full algorithmic pipeline, stub (truncation) compression,
                     embedding novelty, retrieval-token scoring. No API calls.
  fifo       free  — MemGPT-style FIFO baseline (eval/baselines/memgpt_sim.py).
  ours       paid  — algorithmic pipeline with real LLM compression + LLM-answer probe.
  letta      paid  — real Letta agent.

Grading is opaque-token exact match (free, deterministic) on both the retrieved
block (free systems) and the LLM answer (paid systems).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.baselines.memgpt_sim import FIFOMemory  # noqa: E402
from eval.cost import BudgetExceeded, CostMeter  # noqa: E402
from eval.runners import OursSystem  # noqa: E402
from llm_client import (  # noqa: E402
    LETTA_OPENROUTER_SONNET, OPENROUTER_SONNET, has_llm_key, make_anthropic_client,
)
from eval.stress_test import (  # noqa: E402
    generate_distractors,
    generate_tiered_facts,
    interleave_tiered,
)
from memory.config import MemoryConfig  # noqa: E402
from memory.novelty import NoveltyMode  # noqa: E402

FREE_SYSTEMS = {"ours_stub", "fifo"}
PAID_SYSTEMS = {"ours", "letta"}


def _first_mention(items: list[dict], fact: dict) -> dict | None:
    for it in items:
        if it["kind"] == "fact" and it["fact"]["needle_id"] == fact["needle_id"]:
            return it
    return None


def _position_band(tokens_after: int, window: int) -> str:
    """Bucket a fact by how much content arrives after its first mention, relative
    to the context window W. < W survives in both systems (control); >= W is at
    risk under FIFO. Uses a single reference W (our --max-tokens) for every system
    so the same fact gets the same label and recall-by-band is comparable."""
    if window <= 0:
        return "recent"
    return "recent" if tokens_after < window else ("mid" if tokens_after < 2 * window else "far")


def _result_row(system: str, fact: dict, items: list[dict], window: int,
                found: bool, retained: bool, excerpt: str,
                retained_in_context: bool | None = None) -> dict:
    first = _first_mention(items, fact)
    pos = first["stream_pos"] if first else -1
    tokens_after = first["tokens_after"] if first else None
    return {
        "system": system,
        "needle_id": fact["needle_id"],
        "novelty_tier": fact["novelty_tier"],
        "reaccessed": fact["reaccessed"],
        "stream_pos": pos,
        "tokens_after": tokens_after,
        "position_band": _position_band(tokens_after, window) if first else "unplaced",
        "found": found,        # recall@1: token in top-1 retrieved block / LLM answer
        "retained": retained,  # token survives anywhere in memory (context ∪ LT)
        # in-context retention: token still resident in the CONTEXT window (LT excluded).
        # Unlike `retained` (which saturates at ~1.0 for ours because LT is unbounded),
        # this can drop, so it shows what the eviction policy keeps live in context.
        # None for systems where it isn't introspectable (Letta's archival store).
        "retained_in_context": retained_in_context,
        "excerpt": excerpt[:120],
    }


def run_ours(
    items: list[dict],
    facts: list[dict],
    max_tokens: int,
    embedder: SentenceTransformer,
    config: MemoryConfig,
    *,
    cost_mode: str,
    novelty_mode: NoveltyMode,
    system_name: str,
    client: Any = None,
    util_model: str = "claude-sonnet-4-6",
    query_model: str = "claude-sonnet-4-6",
    meter: CostMeter | None = None,
) -> list[dict]:
    sysm = OursSystem(
        max_tokens, embedder, config,
        cost_mode=cost_mode, novelty_mode=novelty_mode, client=client,
        util_model=util_model, meter=meter, system_name=system_name,
    )
    for it in items:
        sysm.ingest(it["content"])

    rows = []
    for fact in facts:
        token = fact["unique_token"]
        retained = sysm.retained_token(token)               # context ∪ LT
        in_ctx = sysm.context_contains(token)               # context only (can drop)
        if cost_mode == "live":
            answer = sysm.llm_probe(fact["query"], query_model)
            found = token in answer
            rows.append(_result_row(system_name, fact, items, max_tokens, found,
                                    retained, answer, retained_in_context=in_ctx))
        else:
            retrieved = sysm.retrieve_top1(fact["query"]) or ""
            found = token in retrieved
            rows.append(_result_row(system_name, fact, items, max_tokens, found,
                                    retained, retrieved, retained_in_context=in_ctx))
    return rows


def run_fifo(
    items: list[dict],
    facts: list[dict],
    max_tokens: int,
    embedder: SentenceTransformer,
) -> list[dict]:
    fifo = FIFOMemory(max_tokens=max_tokens, embedder=embedder)
    for it in items:
        fifo.ingest(it["content"])
    rows = []
    for fact in facts:
        token = fact["unique_token"]
        retained = any(token in b.content for b in fifo.blocks)
        block = fifo.query(fact["query"])
        retrieved = block.content if block else ""
        found = token in retrieved
        # FIFO has no long-term store, so in-context retention == retention.
        rows.append(_result_row("fifo", fact, items, max_tokens, found, retained,
                                retrieved, retained_in_context=retained))
    return rows


def run_letta(
    items: list[dict],
    facts: list[dict],
    context_window: int,
    letta_model: str,
    letta_base_url: str | None,
    letta_embedding: str | None,
    meter: CostMeter | None,
    bucket_window: int,
    letta_timeout: float = 600.0,
) -> list[dict]:
    from eval.baselines.letta_runner import LettaRunner
    runner = LettaRunner(
        model=letta_model, context_window_limit=context_window,
        base_url=letta_base_url, api_key=os.environ.get("LETTA_API_KEY"),
        embedding=letta_embedding, meter=meter, system="letta",
        timeout=letta_timeout,
    )
    try:
        for it in items:
            runner.ingest(it["content"])
        rows = []
        for fact in facts:
            answer = runner.query(fact["query"])
            found = fact["unique_token"] in answer
            # Letta's archival memory isn't introspectable here, so retention is
            # only observable through the answer; use found as the proxy. Bucket by
            # our --max-tokens (bucket_window), not Letta's larger context window, so
            # the same fact lands in the same band across systems.
            rows.append(_result_row("letta", fact, items, bucket_window, found, found, answer))
        return rows
    finally:
        runner.close()


def _summarize(rows: list[dict]) -> dict[str, Any]:
    # Empty subsets return None (not 0.0) so a band/tier with no facts is never
    # mistaken for a measured zero-recall result.
    def recall(subset: list[dict]) -> float | None:
        return round(sum(r["found"] for r in subset) / len(subset), 3) if subset else None

    def retention(subset: list[dict]) -> float | None:
        return round(sum(r["retained"] for r in subset) / len(subset), 3) if subset else None

    def in_context(subset: list[dict]) -> float | None:
        # Average over facts where in-context retention is measurable (None = not
        # introspectable, e.g. Letta's archival store — excluded, not counted as 0).
        vals = [r["retained_in_context"] for r in subset if r["retained_in_context"] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    out: dict[str, Any] = {}
    for system in sorted({r["system"] for r in rows}):
        sr = [r for r in rows if r["system"] == system]
        out[system] = {
            "overall": recall(sr),
            "retention_overall": retention(sr),
            "in_context_retention_overall": in_context(sr),
            "by_tier": {
                t: recall([r for r in sr if r["novelty_tier"] == t])
                for t in ("high", "low")
            },
            "retention_by_tier": {
                t: retention([r for r in sr if r["novelty_tier"] == t])
                for t in ("high", "low")
            },
            "in_context_retention_by_tier": {
                t: in_context([r for r in sr if r["novelty_tier"] == t])
                for t in ("high", "low")
            },
            "by_position": {
                p: recall([r for r in sr if r["position_band"] == p])
                for p in ("recent", "mid", "far")
            },
            "retention_by_position": {
                p: retention([r for r in sr if r["position_band"] == p])
                for p in ("recent", "mid", "far")
            },
            "in_context_retention_by_position": {
                p: in_context([r for r in sr if r["position_band"] == p])
                for p in ("recent", "mid", "far")
            },
            "by_reaccess": {
                "reaccessed": recall([r for r in sr if r["reaccessed"]]),
                "single": recall([r for r in sr if not r["reaccessed"]]),
            },
        }
    return out


def _parse_sweep(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _load_config(path: str | None, clock_seconds_per_turn: float) -> MemoryConfig:
    """Build a MemoryConfig from a tuning JSON (its 'tuned_params') or a plain
    params dict; else defaults with the given clock rate."""
    if not path:
        return MemoryConfig(clock_seconds_per_turn=clock_seconds_per_turn)
    with open(path) as f:
        data = json.load(f)
    params = data.get("tuned_params", data)
    return MemoryConfig(**params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B recall-advantage stress test.")
    parser.add_argument("--systems", default="ours_stub,fifo",
                        help="Comma-separated: ours_stub, fifo, ours, letta")
    parser.add_argument("--facts", type=int, default=10)
    parser.add_argument("--distractors", type=_parse_sweep, default="50,100,200")
    parser.add_argument("--distractor-tokens", type=int, default=0,
                        help="If >0, pad each distractor to ~this many tokens so the "
                        "stream overflows a large context window with fewer items.")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Our memory-block budget (forces overflow).")
    parser.add_argument("--letta-context-window", type=int, default=4096,
                        help="Letta's TOTAL context window — different unit from our "
                        "--max-tokens (Letta's system prompt + tools alone are ~1000 tokens, "
                        "so this must be well above that).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--clock-seconds-per-turn", type=float, default=600.0)
    parser.add_argument("--novelty", default="embedding",
                        choices=[m.value for m in NoveltyMode])
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--util-model", default=OPENROUTER_SONNET)
    parser.add_argument("--query-model", default=OPENROUTER_SONNET)
    parser.add_argument("--letta-model", default=LETTA_OPENROUTER_SONNET)
    # Defaults target the verified self-hosted Letta server (Docker). For Letta
    # Cloud instead, pass --letta-base-url "" and set LETTA_API_KEY.
    parser.add_argument("--letta-base-url", default="http://localhost:8283")
    parser.add_argument("--letta-embedding", default="ollama/nomic-embed-text:latest")
    parser.add_argument("--letta-timeout", type=float, default=600.0,
                        help="Per-request HTTP timeout (s) for the Letta client. Big-context "
                        "turns near overflow are slow; the ~60s client default times out.")
    parser.add_argument("--max-spend", type=float, default=50.0)
    parser.add_argument("--config", default=None,
                        help="Path to a tuning JSON (uses its 'tuned_params') or a plain "
                        "params dict, to build the MemoryConfig. Overrides --clock-seconds-per-turn.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    paid = [s for s in systems if s in PAID_SYSTEMS]
    if "ours" in paid and not has_llm_key():
        sys.exit("Error: an LLM key (OPENROUTER_API_KEY or ANTHROPIC_API_KEY) is required for 'ours'.")
    if "letta" in paid and not args.letta_base_url and not os.environ.get("LETTA_API_KEY"):
        sys.exit("Error: Letta requires --letta-base-url or LETTA_API_KEY.")

    client = make_anthropic_client() if paid else None

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / "results" / f"recall_advantage_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[recall_advantage] loading embedder: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)
    meter = CostMeter(max_spend=args.max_spend)

    runs: list[dict] = []
    try:
        for n_distractors in args.distractors:
            for trial in range(args.trials):
                run_seed = args.seed + 1000 * trial
                facts = generate_tiered_facts(args.facts, run_seed)
                distractors = generate_distractors(n_distractors, run_seed, args.distractor_tokens)
                items = interleave_tiered(facts, distractors, run_seed, args.max_tokens)
                print(f"[recall_advantage] distractors={n_distractors} trial={trial} "
                      f"stream_len={len(items)} systems={systems}", flush=True)

                trial_rows: list[dict] = []
                for system in systems:
                    cfg = _load_config(args.config, args.clock_seconds_per_turn)
                    t0 = time.perf_counter()
                    if system == "ours_stub":
                        trial_rows += run_ours(
                            items, facts, args.max_tokens, embedder, cfg,
                            cost_mode="stub", novelty_mode=NoveltyMode.EMBEDDING,
                            system_name="ours_stub",
                        )
                    elif system == "ours":
                        trial_rows += run_ours(
                            items, facts, args.max_tokens, embedder, cfg,
                            cost_mode="live", novelty_mode=NoveltyMode(args.novelty),
                            system_name="ours", client=client,
                            util_model=args.util_model, query_model=args.query_model,
                            meter=meter,
                        )
                    elif system == "fifo":
                        trial_rows += run_fifo(items, facts, args.max_tokens, embedder)
                    elif system == "letta":
                        trial_rows += run_letta(
                            items, facts, args.letta_context_window, args.letta_model,
                            args.letta_base_url, args.letta_embedding, meter,
                            bucket_window=args.max_tokens, letta_timeout=args.letta_timeout,
                        )
                    print(f"  {system} done in {time.perf_counter() - t0:.1f}s", flush=True)

                for r in trial_rows:
                    r.update(n_distractors=n_distractors, trial=trial,
                             max_tokens=args.max_tokens, stream_len=len(items))
                runs += trial_rows

                summ = _summarize(trial_rows)
                for system in systems:
                    s = summ.get(system, {})
                    print(f"  {system}: recall@1={s.get('overall')} "
                          f"recall_tier={s.get('by_tier')} "
                          f"in_ctx_retention={s.get('in_context_retention_overall')} "
                          f"in_ctx_tier={s.get('in_context_retention_by_tier')} "
                          f"retention(ctx∪LT)={s.get('retention_overall')}", flush=True)
                if paid:
                    meter.print_tally()
    except BudgetExceeded as e:
        print(f"[recall_advantage] ABORTED: {e}", file=sys.stderr, flush=True)

    # Summarize per distractor count: each is a distinct overflow regime (mild vs
    # severe), so blending them hides the discriminating condition. This is the
    # headline breakdown; the flat `summary` below is a secondary quick-glance
    # number that intentionally conflates regimes.
    summary_by_distractors = {
        str(n): _summarize([r for r in runs if r["n_distractors"] == n])
        for n in sorted({r["n_distractors"] for r in runs})
    }
    out = {
        "config": vars(args),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metric": "found — unique token present in retrieved block (free) or LLM answer (paid)",
        "summary_by_distractors": summary_by_distractors,
        "summary": _summarize(runs),  # overall blend across all distractor counts
        "cost": meter.summary(),
        "results": runs,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[recall_advantage] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
