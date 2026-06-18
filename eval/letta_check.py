"""Connection-only smoke check for Letta — NOT a benchmark.

Validates the full Letta integration cheaply before any real eval run:
  1. connect (self-hosted server via --base-url, or Letta Cloud via LETTA_API_KEY)
  2. discover available LLM + embedding model handles the server exposes
  3. create one agent, send ONE ingest message and ONE query message
  4. dump the raw response shape (message types + usage fields) so we can confirm
     the cost meter (eval/cost.py add_letta_response) reads the right fields
  5. delete the agent

Cost: a couple of model calls (cents). Run:
    python eval/letta_check.py --base-url http://localhost:8283 \
        --model anthropic/claude-sonnet-4-6 --embedding ollama/nomic-embed-text
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _describe_usage(usage) -> dict:
    """Pull every plausible token-count field off a usage object for inspection."""
    if usage is None:
        return {"<no usage attr>": None}
    fields = [
        "prompt_tokens", "completion_tokens", "total_tokens",
        "input_tokens", "output_tokens", "step_count",
    ]
    out = {f: getattr(usage, f, None) for f in fields}
    out["_type"] = type(usage).__name__
    out["_all_attrs"] = [a for a in dir(usage) if not a.startswith("_")]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Letta connection-only smoke check.")
    parser.add_argument("--base-url", default="http://localhost:8283",
                        help="Self-hosted Letta server. Ignored if LETTA_API_KEY is set for Cloud.")
    parser.add_argument("--model", default="openrouter/anthropic/claude-sonnet-4.6")
    # Ollama embedding handles carry the ":latest" tag; letta/letta-free is the
    # zero-setup hosted alternative (no local model needed).
    parser.add_argument("--embedding", default="ollama/nomic-embed-text:latest")
    parser.add_argument("--use-cloud", action="store_true",
                        help="Use Letta Cloud via LETTA_API_KEY instead of --base-url.")
    args = parser.parse_args()

    from letta_client import Letta

    kwargs = {}
    if args.use_cloud or (os.environ.get("LETTA_API_KEY") and not args.base_url):
        kwargs["api_key"] = os.environ["LETTA_API_KEY"]
        print("[letta_check] connecting to Letta Cloud")
    else:
        kwargs["base_url"] = args.base_url
        print(f"[letta_check] connecting to self-hosted Letta at {args.base_url}")
    client = Letta(**kwargs)

    # 1. discover available model handles
    print("\n=== available LLM handles (client.models.list) ===")
    try:
        for m in client.models.list()[:20]:
            print("   ", getattr(m, "handle", None) or getattr(m, "model", m))
    except Exception as e:
        print(f"  models.list() -> {type(e).__name__}: {e}")
    print("=== available embedding handles (client.models.embeddings.list) ===")
    try:
        for m in client.models.embeddings.list()[:20]:
            print("   ", getattr(m, "handle", None) or getattr(m, "embedding_model", m))
    except Exception as e:
        print(f"  models.embeddings.list() -> {type(e).__name__}: {e}")

    # 2. create an agent
    print("\n=== creating agent ===")
    agent = client.agents.create(
        name="letta_conn_check",
        model=args.model,
        embedding=args.embedding,
        memory_blocks=[
            {"label": "human", "value": "The user gives facts to memorize.", "limit": 2000},
            {"label": "persona", "value": "I store facts and answer from memory.", "limit": 1000},
        ],
    )
    print(f"agent_id = {agent.id}")

    try:
        # 3. one ingest, one query — dump raw shapes
        print("\n=== ingest message ===")
        r1 = client.agents.messages.create(
            agent_id=agent.id,
            input="Memorize this fact: the vault passcode is ZX9-QF1. Reply: stored.",
        )
        _dump_response("ingest", r1)

        print("\n=== query message ===")
        r2 = client.agents.messages.create(
            agent_id=agent.id, input="What is the vault passcode?",
        )
        _dump_response("query", r2)

        # quick correctness signal
        text = _assistant_text(r2)
        print(f"\n[letta_check] assistant answer: {text!r}")
        print(f"[letta_check] passcode recalled: {'ZX9-QF1' in text}")
    finally:
        print("\n=== deleting agent ===")
        try:
            client.agents.delete(agent.id)
            print("deleted")
        except Exception as e:
            print(f"delete failed: {e}")


def _dump_response(label: str, response) -> None:
    msgs = getattr(response, "messages", []) or []
    print(f"[{label}] {len(msgs)} messages:")
    for m in msgs:
        mtype = getattr(m, "message_type", None) or getattr(m, "type", None)
        snippet = (getattr(m, "content", None) or getattr(m, "text", None)
                   or getattr(m, "reasoning", None) or "")
        print(f"   - {mtype}: {str(snippet)[:80]!r}")
    print(f"[{label}] usage: {_describe_usage(getattr(response, 'usage', None))}")


def _assistant_text(response) -> str:
    parts = []
    for m in getattr(response, "messages", []) or []:
        if (getattr(m, "message_type", None) or getattr(m, "type", None)) in ("assistant_message", "assistant"):
            c = getattr(m, "content", None) or getattr(m, "text", None)
            if c:
                parts.append(str(c))
    return "\n".join(parts)


if __name__ == "__main__":
    main()
