"""Live needle-in-a-haystack: full system (real LLM compression) vs. real Letta.

Unlike eval/stress_test.py (Tier 1, stub compress, retrieval-only scoring), this
harness uses:
  - Ours: Haiku summarization + merge via MemoryController.receive(), then a
    probe LLM call with build_memory_prompt() injected as system context.
  - Letta: real agent ingest + probe messages via letta-client.

Both sides see the same synthetic needle stream. Primary metric: unique needle
token appears in the probe LLM's answer text.

Cost control: defaults are small (5 needles, 50 distractors). Scale up only when
you are ready to spend on API calls.
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

from ContextManager import ContextManager  # noqa: E402
from controller import MemoryController  # noqa: E402
from eval.baselines.letta_runner import LettaRunner  # noqa: E402
from eval.stress_test import (  # noqa: E402
    generate_distractors,
    generate_needles,
    interleave,
    retrieve_top1,
)
from functions.llm_fns import make_compress_fn, make_merge_fn  # noqa: E402
from llm.interface import LLMBackend  # noqa: E402
from llm_client import has_llm_key, make_llm_backend  # noqa: E402
from memory.longterm import LongTermStore  # noqa: E402
from memory.novelty import score_novelty_embedding  # noqa: E402
from memory.store import ContextStore  # noqa: E402

_QUERY_SYSTEM_SUFFIX = (
    "\n\nAnswer the user's question using only information from MEMORY STATUS above. "
    "Reply with the fact only — no preamble."
)


def _embedding_dim(embedder: SentenceTransformer) -> int:
    return int(embedder.get_sentence_embedding_dimension())


def run_ours_live(
    items: list[dict],
    needles: list[dict],
    max_tokens: int,
    embedder: SentenceTransformer,
    backend: LLMBackend,
    util_model: str,
    query_model: str,
) -> list[dict]:
    dim = _embedding_dim(embedder)
    store = ContextStore(max_tokens=max_tokens)
    lt_store = LongTermStore("sqlite:///:memory:", embedding_dim=dim)
    cm = ContextManager(store, lt_store, embedding_model=embedder)
    controller = MemoryController(
        cm,
        compress_fn=make_compress_fn(backend, util_model),
        merge_fn=make_merge_fn(backend, cm, util_model),
    )

    print(f"  [ours] ingesting {len(items)} items...", flush=True)
    for i, item in enumerate(items):
        emb = cm.embed(item["content"])
        novelty = score_novelty_embedding(emb, cm, top_k=5)
        controller.receive(item["content"], emb, novelty)
        if (i + 1) % 25 == 0:
            print(f"  [ours]   {i + 1}/{len(items)}", flush=True)

    results = []
    print(f"  [ours] probing {len(needles)} needles...", flush=True)
    for needle in needles:
        q_emb = cm.embed(needle["query"])
        controller.pre_prompt_promote(q_emb)
        memory_prompt = controller.build_memory_prompt()
        response = backend.complete(
            model=query_model,
            max_tokens=256,
            system=memory_prompt + _QUERY_SYSTEM_SUFFIX,
            messages=[{"role": "user", "content": needle["query"]}],
        )
        answer = response.text
        retrieved = retrieve_top1(cm, lt_store, q_emb)
        token = needle["unique_token"]
        results.append({
            "system": "ours",
            "needle_id": needle["needle_id"],
            "stream_pos": next(it["stream_pos"] for it in items if it.get("needle") == needle),
            "found_answer": token in answer,
            "found_retrieval": bool(retrieved and token in retrieved),
            "answer_excerpt": answer[:160],
            "retrieved_excerpt": (retrieved or "")[:120],
        })
    return results


def run_letta_live(
    items: list[dict],
    needles: list[dict],
    max_tokens: int,
    letta_model: str,
    letta_base_url: str | None,
    letta_api_key: str | None,
    letta_embedding: str | None,
) -> list[dict]:
    runner = LettaRunner(
        model=letta_model,
        context_window_limit=max_tokens,
        base_url=letta_base_url,
        api_key=letta_api_key,
        embedding=letta_embedding,
    )
    try:
        print(f"  [letta] agent={runner.agent_id} ingesting {len(items)} items...", flush=True)
        for i, item in enumerate(items):
            runner.ingest(item["content"])
            if (i + 1) % 25 == 0:
                print(f"  [letta]   {i + 1}/{len(items)}", flush=True)

        results = []
        print(f"  [letta] probing {len(needles)} needles...", flush=True)
        for needle in needles:
            answer = runner.query(needle["query"])
            token = needle["unique_token"]
            results.append({
                "system": "letta",
                "needle_id": needle["needle_id"],
                "stream_pos": next(it["stream_pos"] for it in items if it.get("needle") == needle),
                "found_answer": token in answer,
                "found_retrieval": None,
                "answer_excerpt": answer[:160],
                "retrieved_excerpt": None,
            })
        return results
    finally:
        runner.close()


def _parse_sweep(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live NIAH eval: full system vs. real Letta agent.",
    )
    parser.add_argument("--systems", default="ours,letta", help="Comma-separated: ours, letta")
    parser.add_argument("--needles", type=int, default=5)
    parser.add_argument("--distractors", type=_parse_sweep, default="50,100,200")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--util-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--query-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--letta-model", default="anthropic/claude-haiku-4-5-20251001")
    parser.add_argument("--letta-base-url", default=None, help="Self-hosted Letta, e.g. http://localhost:8283")
    parser.add_argument("--letta-embedding", default=None, help="Required for self-hosted, e.g. ollama/mxbai-embed-large")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    systems = {s.strip() for s in args.systems.split(",") if s.strip()}
    if "letta" in systems and not args.letta_base_url and not os.environ.get("LETTA_API_KEY"):
        print("Error: Letta requires --letta-base-url or LETTA_API_KEY.", file=sys.stderr)
        sys.exit(1)
    if not has_llm_key() and "ours" in systems:
        print("Error: OPENROUTER_API_KEY or ANTHROPIC_API_KEY required for our system.", file=sys.stderr)
        sys.exit(1)

    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "eval" / "results" / f"niah_live_{int(time.time())}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[niah_live] loading embedder: {args.embedding_model}", flush=True)
    embedder = SentenceTransformer(args.embedding_model)
    backend = make_llm_backend() if "ours" in systems else None

    runs: list[dict[str, Any]] = []
    for n_distractors in args.distractors:
        for trial in range(args.trials):
            run_seed = args.seed + 1000 * trial
            needles = generate_needles(args.needles, run_seed)
            distractors = generate_distractors(n_distractors, run_seed)
            items = interleave(needles, distractors, run_seed)

            print(
                f"[niah_live] distractors={n_distractors} trial={trial} "
                f"stream_len={len(items)} systems={sorted(systems)}",
                flush=True,
            )

            trial_rows: list[dict] = []
            if "ours" in systems:
                t0 = time.perf_counter()
                trial_rows.extend(
                    run_ours_live(
                        items, needles, args.max_tokens, embedder,
                        backend, args.util_model, args.query_model,
                    )
                )
                print(f"  ours done in {time.perf_counter() - t0:.1f}s", flush=True)

            if "letta" in systems:
                t0 = time.perf_counter()
                trial_rows.extend(
                    run_letta_live(
                        items, needles, args.max_tokens,
                        args.letta_model, args.letta_base_url,
                        os.environ.get("LETTA_API_KEY"), args.letta_embedding,
                    )
                )
                print(f"  letta done in {time.perf_counter() - t0:.1f}s", flush=True)

            for r in trial_rows:
                r["n_distractors"] = n_distractors
                r["trial"] = trial
                r["max_tokens"] = args.max_tokens
                r["stream_len"] = len(items)

            for sys_name in sorted(systems):
                sys_rows = [r for r in trial_rows if r["system"] == sys_name]
                if sys_rows:
                    ans = sum(r["found_answer"] for r in sys_rows) / len(sys_rows)
                    print(f"  {sys_name} answer_recall={ans:.0%}", flush=True)

            runs.extend(trial_rows)

    out = {
        "config": vars(args),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metric": "found_answer — unique token in probe LLM response",
        "results": runs,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[niah_live] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
