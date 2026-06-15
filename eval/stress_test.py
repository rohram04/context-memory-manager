"""Sequential-NIAH stress test: novelty-scored memory vs. MemGPT-style FIFO.

Plants N "needles" (distinctly-identifiable facts) inside a stream of M
"distractors" of comparable shape, ingests the entire stream into both memory
systems, then queries each needle and reports recall@1 over the union of
context + long-term store.

Tier 1: no LLM calls anywhere. Compress and merge are stubbed with text
truncation / concatenation. The eviction-policy difference is what we want
to show; summarization quality isn't part of the claim.

The needle's "presence" after eviction/compression is tested by checking for a
unique opaque token (e.g. ``X92QF1``) in the top-1 retrieved block's content.
The token survives compression and lives in the LT copy, so this is a fair
test of whether the system *retained the fact in any form*, not whether it
kept verbatim text.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ContextManager import ContextManager  # noqa: E402
from controller import MemoryController  # noqa: E402
from eval.baselines.memgpt_sim import FIFOMemory  # noqa: E402
from memory.longterm import LongTermStore  # noqa: E402
from memory.novelty import score_novelty_embedding  # noqa: E402
from memory.store import ContextStore  # noqa: E402


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


_FIRST_NAMES = [
    "Alice", "Bob", "Carlos", "Dana", "Elena", "Farouk", "Gita", "Hiro",
    "Ines", "Jonas", "Kira", "Lakshmi", "Mateo", "Nadia", "Owen", "Priya",
    "Quincy", "Ravi", "Sana", "Theo", "Uma", "Viktor", "Wren", "Xan",
    "Yuki", "Zara",
]

_DISTRACTOR_TEMPLATES = [
    "The {animal} ate {food} during the {event} last {season}.",
    "Note: {city} reported a {color} sky after the {weather} on Tuesday.",
    "Reminder: send {document} to {name} before the {time_of_day} meeting.",
    "Project log: {tool} crashed twice while parsing {file_type} files yesterday.",
    "Status update: the {team} team finished migrating {service} to {cloud}.",
    "Trivia: {animal}s can recognize {number} distinct {object}s on average.",
    "Schedule: {event} moved from {city} to {city2} due to {weather}.",
    "Receipt: bought {number} {object}s at the {place} for {price} dollars.",
    "Observation: {color} light reflects off {material} more than {material2}.",
    "Diary: spent the afternoon reading about {topic} in the {place}.",
]

_FILL = {
    "animal": ["badger", "otter", "raven", "lemur", "octopus", "panther", "wolf", "ibis"],
    "food": ["sourdough", "kimchi", "polenta", "tamarind", "ramen", "gnocchi"],
    "event": ["festival", "summit", "marathon", "regatta", "carnival"],
    "season": ["spring", "summer", "autumn", "winter", "monsoon"],
    "city": ["Helsinki", "Quito", "Osaka", "Lagos", "Reykjavik", "Antwerp"],
    "city2": ["Manila", "Kraków", "Hobart", "Sofia"],
    "color": ["amber", "indigo", "vermilion", "ochre", "teal"],
    "weather": ["hailstorm", "fog bank", "heat wave", "dust devil"],
    "document": ["the audit report", "the design doc", "the budget memo"],
    "name": ["the auditor", "the new intern", "the planning committee"],
    "time_of_day": ["morning", "afternoon", "evening", "midnight"],
    "tool": ["the parser", "the linter", "the build script", "the migration tool"],
    "file_type": [".yaml", ".csv", ".parquet", ".proto", ".toml"],
    "team": ["platform", "infra", "ML ops", "growth", "security"],
    "service": ["the billing service", "the metrics pipeline", "the search index"],
    "cloud": ["GCP", "AWS", "Fly.io", "Cloudflare"],
    "number": ["seven", "twelve", "thirty", "fifty-eight"],
    "object": ["pebble", "trinket", "scroll", "rune", "card"],
    "place": ["bazaar", "library", "marina", "warehouse"],
    "price": ["18", "42", "97", "203"],
    "material": ["copper", "obsidian", "velvet", "porcelain"],
    "material2": ["slate", "linen", "amber", "leather"],
    "topic": ["Carthaginian trade routes", "tidal estuaries", "Byzantine glasswork"],
}


def _rand_token(rng: random.Random, n: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choices(alphabet, k=n))


def generate_needles(n: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    names = rng.sample(_FIRST_NAMES, k=min(n, len(_FIRST_NAMES)))
    while len(names) < n:
        names.append(rng.choice(_FIRST_NAMES) + str(rng.randint(2, 99)))
    needles = []
    for i, name in enumerate(names):
        token = _rand_token(rng)
        needles.append({
            "needle_id": f"needle_{i:03d}",
            "name": name,
            "unique_token": token,
            "content": f"{name}'s secret access code is {token}.",
            "query": f"What is {name}'s secret access code?",
        })
    return needles


def generate_distractors(n: int, seed: int) -> list[str]:
    rng = random.Random(seed + 1)
    out = []
    for _ in range(n):
        template = rng.choice(_DISTRACTOR_TEMPLATES)
        filled = template.format(**{k: rng.choice(v) for k, v in _FILL.items()})
        out.append(filled)
    return out


def interleave(needles: list[dict], distractors: list[str], seed: int) -> list[dict]:
    """Place needles at roughly even positions across the distractor stream."""
    rng = random.Random(seed + 2)
    total = len(needles) + len(distractors)
    items: list[dict | None] = [None] * total

    if needles:
        # Even-ish positions, then jitter slightly to avoid hitting exactly the
        # same slots every trial.
        step = total / len(needles)
        positions = sorted({
            min(total - 1, max(0, int(i * step + rng.uniform(0, step * 0.5))))
            for i in range(len(needles))
        })
        # Fill any collisions by shifting forward.
        used = set()
        adjusted = []
        for p in positions:
            while p in used and p < total - 1:
                p += 1
            used.add(p)
            adjusted.append(p)
        for pos, needle in zip(adjusted, needles):
            items[pos] = {
                "kind": "needle",
                "content": needle["content"],
                "needle": needle,
                "stream_pos": pos,
            }

    distractor_iter = iter(distractors)
    for i in range(total):
        if items[i] is None:
            items[i] = {
                "kind": "distractor",
                "content": next(distractor_iter),
                "needle": None,
                "stream_pos": i,
            }
    return items  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dummy compress / merge — no LLM
# ---------------------------------------------------------------------------


def _truncate_compress(content: str) -> str:
    """Halve the content (by character count) with a 20-char floor."""
    if len(content) <= 20:
        return content
    return content[: max(20, len(content) // 2)]


def _make_concat_merge(cm: ContextManager):
    def merge(existing: str, new: str) -> tuple[str, list[float]]:
        merged = f"{existing} | {new}"
        return merged, cm.embed(merged)
    return merge


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(va, vb) / (na * nb))


def retrieve_top1(cm: ContextManager, lt: LongTermStore, query_emb: list[float]) -> str | None:
    """Return content of the highest-similarity block across context ∪ LT."""
    ctx_block = cm.find_similar(query_emb, threshold=-1.0)
    ctx_sim = _cosine(query_emb, ctx_block.original_embedding) if ctx_block else -1.0

    lt_results = lt.similarity_search(query_emb, top_k=1)
    if lt_results:
        lt_block, lt_sim = lt_results[0]
    else:
        lt_block, lt_sim = None, -1.0

    if ctx_sim >= lt_sim:
        return ctx_block.content if ctx_block else None
    return lt_block.content if lt_block else None


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------


def run_our_system(
    items: list[dict],
    needles: list[dict],
    max_tokens: int,
    embedder: SentenceTransformer,
) -> list[dict]:
    store = ContextStore(max_tokens=max_tokens)
    lt_store = LongTermStore("sqlite:///:memory:")
    cm = ContextManager(store, lt_store, embedding_model=embedder)
    controller = MemoryController(
        cm,
        compress_fn=_truncate_compress,
        merge_fn=_make_concat_merge(cm),
    )

    for item in items:
        emb = cm.embed(item["content"])
        novelty = score_novelty_embedding(emb, cm, top_k=5)
        controller.receive(item["content"], emb, novelty)

    results = []
    for needle in needles:
        q_emb = cm.embed(needle["query"])
        retrieved = retrieve_top1(cm, lt_store, q_emb)
        found = bool(retrieved and needle["unique_token"] in retrieved)
        results.append({
            "system": "ours",
            "needle_id": needle["needle_id"],
            "stream_pos": next(it["stream_pos"] for it in items if it["needle"] == needle),
            "found": found,
            "retrieved_excerpt": (retrieved or "")[:120],
        })
    return results


def run_fifo_system(
    items: list[dict],
    needles: list[dict],
    max_tokens: int,
    embedder: SentenceTransformer,
) -> list[dict]:
    fifo = FIFOMemory(max_tokens=max_tokens, embedder=embedder)
    for item in items:
        fifo.ingest(item["content"])

    results = []
    for needle in needles:
        block = fifo.query(needle["query"])
        retrieved = block.content if block else None
        found = bool(retrieved and needle["unique_token"] in retrieved)
        results.append({
            "system": "fifo",
            "needle_id": needle["needle_id"],
            "stream_pos": next(it["stream_pos"] for it in items if it["needle"] == needle),
            "found": found,
            "retrieved_excerpt": (retrieved or "")[:120],
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_sweep(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential-NIAH stress test: ours vs. FIFO baseline.",
    )
    parser.add_argument("--needles", type=int, default=10)
    parser.add_argument(
        "--distractors",
        type=_parse_sweep,
        default="0,25,50,100,200,400,800",
        help="Comma-separated distractor counts to sweep.",
    )
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "eval" / "results" / f"stress_test_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stress_test] loading embedding model: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)

    runs: list[dict[str, Any]] = []
    for n_distractors in args.distractors:
        for trial in range(args.trials):
            run_seed = args.seed + 1000 * trial
            needles = generate_needles(args.needles, run_seed)
            distractors = generate_distractors(n_distractors, run_seed)
            items = interleave(needles, distractors, run_seed)

            print(
                f"[stress_test] n_distractors={n_distractors} trial={trial} "
                f"stream_len={len(items)}",
                flush=True,
            )
            t0 = time.perf_counter()
            ours = run_our_system(items, needles, args.max_tokens, embedder)
            t_ours = time.perf_counter() - t0
            t0 = time.perf_counter()
            fifo = run_fifo_system(items, needles, args.max_tokens, embedder)
            t_fifo = time.perf_counter() - t0

            # Stamp run-level fields onto each row.
            for r in ours + fifo:
                r["n_distractors"] = n_distractors
                r["trial"] = trial
                r["max_tokens"] = args.max_tokens
                r["stream_len"] = len(items)

            ours_recall = sum(r["found"] for r in ours) / len(ours)
            fifo_recall = sum(r["found"] for r in fifo) / len(fifo)
            print(
                f"  ours={ours_recall:.0%} ({t_ours:.1f}s) | "
                f"fifo={fifo_recall:.0%} ({t_fifo:.1f}s)",
                flush=True,
            )

            runs.extend(ours + fifo)

    out = {
        "config": {
            "needles": args.needles,
            "distractors_sweep": args.distractors,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "trials": args.trials,
            "embedding_model": args.embedding_model,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": runs,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[stress_test] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
