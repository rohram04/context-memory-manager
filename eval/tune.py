"""Free ($0) tuning harness for the algorithmic memory manager's config knobs.

Runs entirely on the Tier-1 path (OursSystem cost_mode="stub": text-truncation
compression, embedding novelty, local embeddings — NO API calls). Sweeps the
tunable MemoryConfig knobs one-factor-at-a-time on DEV seeds, picks the config
that best exhibits novelty-governed retention, then reports it on held-out TEST
seeds (so we never report on the seeds we tuned on).

Objective — in-context retention by novelty tier, NOT raw retention:
  Raw retention (token survives anywhere) is ~100% for ours because the long-term
  store is unbounded, so it can't discriminate configs. Instead we measure whether
  each fact's block survives in the CONTEXT store under overflow (via its anchor,
  which survives stub truncation). The config knobs (compress weights, fidelity
  threshold, promote threshold, augment similarity, clock) directly govern this.

  score = 2 * high_retention - low_retention
  Rewards keeping high-novelty facts in context (high_retention high) AND evicting
  redundant low-novelty ones (separation). Keep-everything and keep-nothing both
  score poorly relative to selective novelty-governance.

Run:  python eval/tune.py            # ~5 min, $0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.runners import OursSystem  # noqa: E402
from eval.stress_test import (  # noqa: E402
    generate_distractors, generate_tiered_facts, interleave_tiered,
)
from memory.config import MemoryConfig  # noqa: E402
from memory.novelty import NoveltyMode  # noqa: E402

DEV_SEEDS = [1, 2, 3, 4, 5, 6]
TEST_SEEDS = [101, 102, 103, 104, 105, 106]

# One-factor-at-a-time sweep grid. compress_recency_weight is derived as
# 1 - compress_novelty_weight, so it isn't swept independently.
SWEEP: dict[str, list] = {
    "compress_novelty_weight": [0.4, 0.6, 0.8],
    "fidelity_removal_threshold": [0.4, 0.5, 0.6],
    "promote_threshold": [0.5, 0.6, 0.7],
    "augment_similarity_threshold": [0.80, 0.85, 0.90],
    "clock_seconds_per_turn": [300.0, 600.0, 1200.0],
}


def _first_tokens_after(items: list[dict], needle_id: str) -> int:
    """Tokens arriving after a fact's first mention; -1 if unplaced. A fact is
    'at risk' (would be evicted under a fixed window) when this exceeds max_tokens."""
    for it in items:
        if it["kind"] == "fact" and it["fact"]["needle_id"] == needle_id:
            return it["tokens_after"]
    return -1


def evaluate(
    cfg: MemoryConfig, seeds: list[int], facts_n: int, distractors_n: int,
    max_tokens: int, embedder: SentenceTransformer,
) -> dict[str, float]:
    """Mean in-context retention metrics over `seeds` for a config (no API)."""
    hi_hits = hi_tot = lo_hits = lo_tot = atrisk_hits = atrisk_tot = 0
    for seed in seeds:
        facts = generate_tiered_facts(facts_n, seed)
        distractors = generate_distractors(distractors_n, seed)
        items = interleave_tiered(facts, distractors, seed, max_tokens)
        sysm = OursSystem(
            max_tokens, embedder, cfg, cost_mode="stub",
            novelty_mode=NoveltyMode.EMBEDDING, system_name="ours_stub",
        )
        for it in items:
            sysm.ingest(it["content"])
        for f in facts:
            retained = sysm.context_contains(f["name"])  # anchor survives truncation
            if f["novelty_tier"] == "high":
                hi_tot += 1; hi_hits += int(retained)
            else:
                lo_tot += 1; lo_hits += int(retained)
            # "At risk" = enough content arrives after the fact to overflow the
            # window; these are the facts whose retention actually discriminates.
            if _first_tokens_after(items, f["needle_id"]) >= max_tokens:
                atrisk_tot += 1; atrisk_hits += int(retained)

    high = hi_hits / hi_tot if hi_tot else 0.0
    low = lo_hits / lo_tot if lo_tot else 0.0
    atrisk = atrisk_hits / atrisk_tot if atrisk_tot else 0.0
    return {
        "high_retention": round(high, 4),
        "low_retention": round(low, 4),
        "gap": round(high - low, 4),
        "atrisk_retention": round(atrisk, 4),
        "objective": round(2 * high - low, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Free Tier-1 tuning sweep for MemoryConfig.")
    parser.add_argument("--facts", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "eval" / "results" / f"tuning_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[tune] loading embedder: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)

    def score(cfg: MemoryConfig, seeds: list[int]) -> dict[str, float]:
        return evaluate(cfg, seeds, args.facts, args.distractors, args.max_tokens, embedder)

    # Start from defaults, but enable the simulated clock so decay is live.
    best = MemoryConfig(clock_seconds_per_turn=600.0)
    base_dev = score(best, DEV_SEEDS)
    print(f"[tune] baseline dev: {base_dev}", flush=True)

    sweep_log: list[dict[str, Any]] = []
    for param, values in SWEEP.items():
        best_val = getattr(best, param)
        best_obj = score(best, DEV_SEEDS)["objective"]
        results = []
        for v in values:
            kwargs = {param: v}
            if param == "compress_novelty_weight":
                kwargs["compress_recency_weight"] = round(1.0 - v, 4)
            cand = dataclasses.replace(best, **kwargs)
            m = score(cand, DEV_SEEDS)
            results.append({"value": v, **m})
            print(f"[tune] {param}={v}: obj={m['objective']} "
                  f"high={m['high_retention']} low={m['low_retention']}", flush=True)
            if m["objective"] > best_obj:
                best_obj, best_val = m["objective"], v
        # commit best value for this param before moving to the next (OFAT)
        commit = {param: best_val}
        if param == "compress_novelty_weight":
            commit["compress_recency_weight"] = round(1.0 - best_val, 4)
        best = dataclasses.replace(best, **commit)
        sweep_log.append({"param": param, "chosen": best_val, "results": results})
        print(f"[tune] -> chose {param}={best_val}\n", flush=True)

    tuned_dev = score(best, DEV_SEEDS)
    # Held-out test-seed comparison: tuned vs defaults.
    default_cfg = MemoryConfig(clock_seconds_per_turn=600.0)
    tuned_test = score(best, TEST_SEEDS)
    default_test = score(default_cfg, TEST_SEEDS)

    tuned_params = {
        "compress_novelty_weight": best.compress_novelty_weight,
        "compress_recency_weight": best.compress_recency_weight,
        "fidelity_removal_threshold": best.fidelity_removal_threshold,
        "promote_threshold": best.promote_threshold,
        "augment_similarity_threshold": best.augment_similarity_threshold,
        "clock_seconds_per_turn": best.clock_seconds_per_turn,
    }

    print("\n=== RESULT ===", flush=True)
    print(f"tuned params: {tuned_params}", flush=True)
    print(f"dev   tuned: {tuned_dev}", flush=True)
    print(f"test  tuned: {tuned_test}", flush=True)
    print(f"test default: {default_test}", flush=True)

    out = {
        "config": vars(args),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "2*high_retention - low_retention (in-context, by novelty tier)",
        "dev_seeds": DEV_SEEDS, "test_seeds": TEST_SEEDS,
        "baseline_dev": base_dev,
        "sweep": sweep_log,
        "tuned_params": tuned_params,
        "tuned_dev": tuned_dev,
        "tuned_test": tuned_test,
        "default_test": default_test,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[tune] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
