"""32k-window overflow test — leveled, with a free pre-flight cost estimator.

Defines three overflow severities for a 32,000-token Letta context window and,
by default, projects the USD cost of running each level for both our system and
real Letta **without making any paid API call**. The actual run reuses the
existing harness (eval/recall_advantage.py) unchanged.

Levels (seed 42, facts=20, distractor-tokens=800) overflow the 32k window:
    mild      80 distractors  ~107 items  ~65k stream   ~2.0x window
    moderate 160 distractors  ~187 items  ~130k stream  ~4.0x window
    severe   320 distractors  ~347 items  ~259k stream  ~8.1x window

Cost model
----------
ours  : measured exactly from a FREE stub ingest — real compression-pass count and
        real per-pass input sizes are captured, and the real query-time memory
        prompt is sized. Only per-call OUTPUT token sizes are assumed (flags).
letta : analytical. Letta resends ~the whole window every turn, so per-ingest-turn
        input is modelled as min(prefix_tokens + overhead, window); queries hit a
        full window. Reported as a LOW-HIGH band because Letta's agent loop may
        invoke the model 1-2.5x per user message (the one genuine unknown until a
        real run is observed). Tune the band with --letta-* flags.

Usage
-----
    # free projection (default)
    python -m eval.overflow_32k --estimate

    # print / run the real harness command for a level
    python -m eval.overflow_32k --run ours,letta --level mild           # prints command
    python -m eval.overflow_32k --run ours,letta --level mild --exec     # runs it
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.cost import _price  # noqa: E402  (reuse pricing — never redefine prices)
from eval.runners import OursSystem, QUERY_SYSTEM_SUFFIX  # noqa: E402
from eval.stress_test import (  # noqa: E402
    _tok,
    generate_distractors,
    generate_tiered_facts,
    interleave_tiered,
)
from llm_client import (  # noqa: E402
    LETTA_OPENROUTER_SONNET,
    OPENROUTER_SONNET,
)
from memory.config import MemoryConfig  # noqa: E402
from memory.novelty import NoveltyMode  # noqa: E402

# Fixed across levels (the 32k overflow regime). max_tokens is our message budget
# == Letta's effective buffer (32k window minus ~6k fixed overhead), matching the
# prior runs' convention of max-tokens < Letta window.
FACTS = 20
DISTRACTOR_TOKENS = 800
MAX_TOKENS = 26000
LETTA_CONTEXT_WINDOW = 32000
SEED = 42

LEVELS: dict[str, int] = {  # level -> distractor count
    "mild": 80,
    "moderate": 160,
    "severe": 320,
}


# ---------------------------------------------------------------------------
# ours — measured from a free stub ingest
# ---------------------------------------------------------------------------


def estimate_ours(
    items: list[dict],
    facts: list[dict],
    embedder: SentenceTransformer,
    *,
    util_model: str,
    query_model: str,
    compress_out: int,
    merge_out: int,
    probe_out: int,
) -> dict:
    """Run the FREE stub pipeline to capture exact compression/merge structure,
    then price the equivalent live run. Embeddings + embedding-novelty are local
    (free); only the LLM compress/merge/probe calls cost in live mode."""
    cfg = MemoryConfig(clock_seconds_per_turn=600.0)
    sysm = OursSystem(
        MAX_TOKENS, embedder, cfg,
        cost_mode="stub", novelty_mode=NoveltyMode.EMBEDDING, system_name="ours",
    )

    # Wrap the stub compress/merge fns to count passes and capture real input sizes.
    stats = {"compress_n": 0, "compress_in": 0, "merge_n": 0, "merge_in": 0}
    _orig_compress = sysm.controller._compress_fn
    _orig_merge = sysm.controller._merge_fn

    def _counting_compress(content: str):
        stats["compress_n"] += 1
        stats["compress_in"] += _tok(content)
        return _orig_compress(content)

    def _counting_merge(existing: str, new: str):
        stats["merge_n"] += 1
        stats["merge_in"] += _tok(existing) + _tok(new)
        return _orig_merge(existing, new)

    sysm.controller._compress_fn = _counting_compress
    sysm.controller._merge_fn = _counting_merge

    for it in items:
        sysm.ingest(it["content"])

    # Real query-time memory prompt size (what each probe sends as system context).
    memory_prompt = sysm.cm.build_memory_prompt()
    probe_in_each = _tok(memory_prompt + QUERY_SYSTEM_SUFFIX) + int(
        sum(_tok(f["query"]) for f in facts) / max(1, len(facts))
    )
    n_probe = len(facts)

    util_in, util_out = _price(util_model)
    q_in, q_out = _price(query_model)

    compress_usd = stats["compress_in"] * util_in + stats["compress_n"] * compress_out * util_out
    merge_usd = stats["merge_in"] * util_in + stats["merge_n"] * merge_out * util_out
    probe_usd = n_probe * (probe_in_each * q_in + probe_out * q_out)
    total = compress_usd + merge_usd + probe_usd

    return {
        "calls": stats["compress_n"] + stats["merge_n"] + n_probe,
        "compress_passes": stats["compress_n"],
        "merge_passes": stats["merge_n"],
        "probe_calls": n_probe,
        "probe_in_each": probe_in_each,
        "input_tokens": stats["compress_in"] + stats["merge_in"] + n_probe * probe_in_each,
        "usd": total,
        "usd_compress": compress_usd,
        "usd_merge": merge_usd,
        "usd_probe": probe_usd,
    }


# ---------------------------------------------------------------------------
# letta — analytical, banded
# ---------------------------------------------------------------------------


def estimate_letta(
    item_toks: list[int],
    n_query: int,
    *,
    letta_model: str,
    window: int,
    overhead: int,
    out_lo: int,
    out_hi: int,
    mult_lo: float,
    mult_hi: float,
) -> dict:
    """Letta resends ~the whole context each turn: per-ingest-turn input is
    min(prefix + overhead, window); each query hits a full window. The agent-loop
    multiplier (model calls per user message) and per-turn output are unknown until
    observed, so cost is a low-high band."""
    in_price, out_price = _price(letta_model)
    prefix = 0
    in_tot = 0
    for t in item_toks:
        in_tot += min(prefix + overhead, window)
        prefix += t
    in_tot += n_query * window
    n_turns = len(item_toks) + n_query

    def usd(mult: float, out_per: int) -> float:
        return mult * (in_tot * in_price + n_turns * out_per * out_price)

    return {
        "turns": n_turns,
        "input_tokens_x1": in_tot,
        "usd_low": usd(mult_lo, out_lo),
        "usd_high": usd(mult_hi, out_hi),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_stream(distractors_n: int) -> tuple[list[dict], list[dict]]:
    facts = generate_tiered_facts(FACTS, SEED)
    distractors = generate_distractors(distractors_n, SEED, DISTRACTOR_TOKENS)
    items = interleave_tiered(facts, distractors, SEED, MAX_TOKENS)
    return facts, items


def recall_advantage_cmd(level: str, systems: str, max_spend: float) -> list[str]:
    return [
        sys.executable, "-m", "eval.recall_advantage",
        "--systems", systems,
        "--facts", str(FACTS),
        "--distractors", str(LEVELS[level]),
        "--distractor-tokens", str(DISTRACTOR_TOKENS),
        "--max-tokens", str(MAX_TOKENS),
        "--letta-context-window", str(LETTA_CONTEXT_WINDOW),
        "--trials", "1",
        "--max-spend", str(max_spend),
        "--out", f"eval/results/overflow_32k_{level}.json",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="32k overflow test + cost estimator.")
    ap.add_argument("--estimate", action="store_true",
                    help="(default) Free per-level/per-system cost projection — no API calls.")
    ap.add_argument("--run", default=None,
                    help="Emit the recall_advantage command for the given systems (e.g. ours,letta).")
    ap.add_argument("--level", default=None, choices=list(LEVELS),
                    help="Restrict to one level (default: all for --estimate, required for --run).")
    ap.add_argument("--exec", action="store_true", help="Actually run the emitted --run command.")
    ap.add_argument("--max-spend", type=float, default=30.0, help="--max-spend passed to the real run.")
    # pricing models (both Sonnet by default == matched-model paper comparison)
    ap.add_argument("--util-model", default=OPENROUTER_SONNET)
    ap.add_argument("--query-model", default=OPENROUTER_SONNET)
    ap.add_argument("--letta-model", default=LETTA_OPENROUTER_SONNET)
    ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    # ours output-size assumptions (inputs are measured)
    ap.add_argument("--compress-out", type=int, default=150)
    ap.add_argument("--merge-out", type=int, default=200)
    ap.add_argument("--probe-out", type=int, default=120)
    # letta band assumptions — CALIBRATED from the real mild run (2026-06-17):
    # 32k-window Sonnet turns averaged ~3,800 output tok/turn and a ~1.71x input
    # multiplier (Letta re-sends context across ~2 internal calls per user turn:
    # reasoning + memory tool call + heartbeat continuation + reply). Predicted
    # mild $22-28 vs actual $24.56.
    ap.add_argument("--letta-overhead", type=int, default=6000)
    ap.add_argument("--letta-out-lo", type=int, default=3500)
    ap.add_argument("--letta-out-hi", type=int, default=4000)
    ap.add_argument("--letta-mult-lo", type=float, default=1.6)
    ap.add_argument("--letta-mult-hi", type=float, default=1.8)
    args = ap.parse_args()

    if args.run:
        level = args.level or "mild"
        cmd = recall_advantage_cmd(level, args.run, args.max_spend)
        print("[overflow_32k] command:\n  " + " ".join(cmd), flush=True)
        if args.exec:
            print(f"[overflow_32k] running level={level} systems={args.run} "
                  f"(--max-spend {args.max_spend})...", flush=True)
            raise SystemExit(subprocess.call(cmd, cwd=str(REPO_ROOT)))
        return

    # default: estimate
    levels = [args.level] if args.level else list(LEVELS)
    print(f"[overflow_32k] loading embedder: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)

    print()
    print("Assumptions: matched model both sides "
          f"(ours util={args.util_model}, query={args.query_model}; letta={args.letta_model}); "
          "embeddings local/free. ours INPUTS measured from a free stub run; ours OUTPUTS and "
          "the letta band are assumptions (see --*-out / --letta-* flags).")
    print(f"Window={LETTA_CONTEXT_WINDOW} (effective buffer {MAX_TOKENS}); "
          f"facts={FACTS}, distractor_tokens={DISTRACTOR_TOKENS}, seed={SEED}.")
    print()
    hdr = (f"{'level':9} {'dist':>4} {'items':>5} {'stream':>7} {'xWin':>5}   "
           f"{'ours $':>7} {'ours calls':>10}   {'letta $ (low-high)':>20} {'letta turns':>11}")
    print(hdr)
    print("-" * len(hdr))
    for level in levels:
        facts, items = build_stream(LEVELS[level])
        item_toks = [_tok(it["content"]) for it in items]
        stream = sum(item_toks)
        ours = estimate_ours(
            items, facts, embedder,
            util_model=args.util_model, query_model=args.query_model,
            compress_out=args.compress_out, merge_out=args.merge_out, probe_out=args.probe_out,
        )
        letta = estimate_letta(
            item_toks, FACTS,
            letta_model=args.letta_model, window=LETTA_CONTEXT_WINDOW,
            overhead=args.letta_overhead, out_lo=args.letta_out_lo, out_hi=args.letta_out_hi,
            mult_lo=args.letta_mult_lo, mult_hi=args.letta_mult_hi,
        )
        band = f"${letta['usd_low']:.2f} - ${letta['usd_high']:.2f}"
        print(f"{level:9} {LEVELS[level]:>4} {len(items):>5} {stream:>7} "
              f"{stream / LETTA_CONTEXT_WINDOW:>4.1f}x   "
              f"${ours['usd']:>6.2f} "
              f"{ours['compress_passes']}c+{ours['merge_passes']}m+{ours['probe_calls']}p".rjust(10)
              + f"   {band:>20} {letta['turns']:>11}")
    print()
    print("ours $ = compress(util) + merge(util) + probe(query). "
          "Switch --util-model to a Haiku id to cut compression cost further.")


if __name__ == "__main__":
    main()
