"""Part A — general parity on LoCoMo: algorithmic manager vs. real Letta.

Ingests each LoCoMo conversation's turns into both systems, probes a sample of
its QA pairs, and grades free-form answers with a cheap Haiku judge. Reports
accuracy overall and by LoCoMo category. The claim here is parity (the manager
keeps up with Letta on realistic conversational recall) — at far lower cost.

PAID: both systems call the LLM. Use --qa-cap and --n-convs to bound spend; the
CostMeter enforces --max-spend. Letta runs via Letta Cloud (LETTA_API_KEY).
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

from eval.baselines.judge import make_judge_fn  # noqa: E402
from eval.cost import BudgetExceeded, CostMeter  # noqa: E402
from eval.datasets.locomo import CATEGORY_NAMES, load_locomo  # noqa: E402
from eval.runners import OursSystem  # noqa: E402
from llm_client import (  # noqa: E402
    LETTA_OPENROUTER_SONNET, OPENROUTER_HAIKU, OPENROUTER_SONNET,
    has_llm_key, make_anthropic_client,
)
from memory.config import MemoryConfig  # noqa: E402
from memory.novelty import NoveltyMode  # noqa: E402


def _cat_name(category: Any) -> str:
    try:
        return CATEGORY_NAMES.get(int(category), str(category))
    except (TypeError, ValueError):
        return str(category)


def run_ours_locomo(conv, max_tokens, embedder, config, client, util_model,
                    query_model, qa_cap, judge, meter) -> list[dict]:
    sysm = OursSystem(
        max_tokens, embedder, config, cost_mode="live",
        novelty_mode=NoveltyMode.EMBEDDING, client=client,
        util_model=util_model, meter=meter, system_name="ours",
    )
    for turn in conv.turns:
        sysm.ingest(turn)
    rows = []
    for qa in conv.qa[:qa_cap]:
        answer = sysm.llm_probe(qa["question"], query_model)
        correct = judge(qa["question"], qa["answer"], answer)
        rows.append({
            "system": "ours", "conv_id": conv.conv_id,
            "category": _cat_name(qa["category"]), "correct": correct,
            "answer_excerpt": answer[:160],
        })
    return rows


def run_letta_locomo(conv, context_window, letta_model, letta_base_url, letta_embedding,
                     qa_cap, judge, meter) -> list[dict]:
    from eval.baselines.letta_runner import LettaRunner
    runner = LettaRunner(
        model=letta_model, context_window_limit=context_window,
        base_url=letta_base_url, api_key=os.environ.get("LETTA_API_KEY"),
        embedding=letta_embedding, meter=meter, system="letta",
    )
    try:
        for turn in conv.turns:
            runner.ingest(turn)
        rows = []
        for qa in conv.qa[:qa_cap]:
            answer = runner.query(qa["question"])
            correct = judge(qa["question"], qa["answer"], answer)
            rows.append({
                "system": "letta", "conv_id": conv.conv_id,
                "category": _cat_name(qa["category"]), "correct": correct,
                "answer_excerpt": answer[:160],
            })
        return rows
    finally:
        runner.close()


def _summarize(rows: list[dict]) -> dict[str, Any]:
    def acc(subset: list[dict]) -> float:
        return round(sum(r["correct"] for r in subset) / len(subset), 3) if subset else 0.0
    out: dict[str, Any] = {}
    for system in sorted({r["system"] for r in rows}):
        sr = [r for r in rows if r["system"] == system]
        cats = sorted({r["category"] for r in sr})
        out[system] = {
            "overall": acc(sr),
            "by_category": {c: acc([r for r in sr if r["category"] == c]) for c in cats},
            "n": len(sr),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Part A LoCoMo parity: ours vs. Letta.")
    parser.add_argument("--systems", default="ours,letta")
    parser.add_argument("--n-convs", type=int, default=3)
    parser.add_argument("--turn-cap", type=int, default=80)
    parser.add_argument("--qa-cap", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Our memory-block budget.")
    parser.add_argument("--letta-context-window", type=int, default=8192,
                        help="Letta's TOTAL context window (different unit from --max-tokens; "
                        "must exceed Letta's ~1000-token system/tool overhead).")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--model", default=OPENROUTER_SONNET)
    parser.add_argument("--util-model", default=OPENROUTER_SONNET)
    parser.add_argument("--judge-model", default=OPENROUTER_HAIKU)
    parser.add_argument("--letta-model", default=LETTA_OPENROUTER_SONNET)
    # Defaults target the verified self-hosted Letta server (Docker). For Letta
    # Cloud instead, pass --letta-base-url "" and set LETTA_API_KEY.
    parser.add_argument("--letta-base-url", default="http://localhost:8283")
    parser.add_argument("--letta-embedding", default="ollama/nomic-embed-text:latest")
    parser.add_argument("--max-spend", type=float, default=50.0)
    parser.add_argument("--config", default=None,
                        help="Path to a tuning JSON ('tuned_params') or params dict for MemoryConfig.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not has_llm_key():
        sys.exit("Error: an LLM key (OPENROUTER_API_KEY or ANTHROPIC_API_KEY) is required (ours + judge).")
    if "letta" in systems and not args.letta_base_url and not os.environ.get("LETTA_API_KEY"):
        sys.exit("Error: Letta requires --letta-base-url or LETTA_API_KEY.")

    client = make_anthropic_client()
    meter = CostMeter(max_spend=args.max_spend)
    # Meter the judge under its own "judge" system key so its tokens are reported
    # separately but still accumulate toward --max-spend (CostMeter._enforce caps
    # on the total across all systems).
    judge = make_judge_fn(meter.wrap_client(client, "judge"), args.judge_model)

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / "results" / f"locomo_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[locomo] loading {args.n_convs} convs (turn_cap={args.turn_cap})", flush=True)
    convs = load_locomo(args.n_convs, args.turn_cap)
    embedder = SentenceTransformer(args.embedding_model)

    rows: list[dict] = []
    try:
        for conv in convs:
            print(f"[locomo] {conv.conv_id}: {len(conv.turns)} turns, "
                  f"probing {min(args.qa_cap, len(conv.qa))} QA", flush=True)
            if "ours" in systems:
                cfg = MemoryConfig(**json.load(open(args.config))["tuned_params"]) if args.config else MemoryConfig()
                rows += run_ours_locomo(conv, args.max_tokens, embedder, cfg, client,
                                        args.util_model, args.model, args.qa_cap, judge, meter)
            if "letta" in systems:
                rows += run_letta_locomo(conv, args.letta_context_window, args.letta_model,
                                         args.letta_base_url, args.letta_embedding,
                                         args.qa_cap, judge, meter)
            s = _summarize([r for r in rows if r["conv_id"] == conv.conv_id])
            print(f"  {conv.conv_id} acc: "
                  + ", ".join(f"{k}={v['overall']}" for k, v in s.items()), flush=True)
            meter.print_tally()
    except BudgetExceeded as e:
        print(f"[locomo] ABORTED: {e}", file=sys.stderr, flush=True)

    out = {
        "config": vars(args),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metric": "Haiku-judged correctness vs gold answer (abstention-aware)",
        "summary": _summarize(rows),
        "cost": meter.summary(),
        "results": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[locomo] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
