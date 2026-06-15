from __future__ import annotations

import argparse
import os
import sys

import anthropic

from agent import Agent, MemoryMode
from ContextManager import ContextManager
from controller import MemoryController
from functions.llm_fns import make_compress_fn, make_merge_fn
from memory.longterm import LongTermStore
from memory.novelty import NoveltyMode
from memory.store import ContextStore


def print_memory(cm: ContextManager) -> None:
    print("\n" + cm.build_memory_prompt() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive REPL for the LLM memory manager. Chat with the agent "
        "and watch the compression/eviction/promotion lifecycle run after each turn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes (--mode):
  algorithmic   MemoryController manages context automatically every turn:
                pre-prompt LT promotion, insert-or-augment, then compress the
                highest-priority blocks until the turn fits the token budget.
                The conversation model never sees memory tools.
  llm           The conversation model manages its own context via tools
                (store/compress/evict/promote/augment/query_lt/update_novelty).
                Nothing is inserted automatically — the model decides what to keep.

novelty scoring (--novelty): how each new block's novelty_score is assigned.
  embedding     1 - mean top-k cosine similarity vs existing memory. Fast, no
                extra API call. Good default for smoke-testing the pipeline.
  llm           One util-model call per block. Captures contradiction/surprise
                that embeddings miss. Slower, one extra API call per block.
  hybrid        Embedding first; only calls the util-model for ambiguous scores
                (0.35-0.65). Balances cost and quality.

interactive commands:
  /status   print the current MEMORY STATUS block (context + LT counts)
  /reset    clear the chat transcript (the memory store is preserved)
  /quit     exit

env:
  ANTHROPIC_API_KEY   required.

examples:
  python cli.py                                  # algorithmic mode, tiny budget
  python cli.py --mode llm --max-tokens 800      # let the model self-manage
  python cli.py --novelty hybrid --db sqlite:///run1.db
""",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in MemoryMode],
        default=MemoryMode.ALGORITHMIC.value,
        help="Who manages context: 'algorithmic' (controller, automatic) or "
        "'llm' (model self-manages via tools). Default: algorithmic.",
    )
    parser.add_argument(
        "--novelty",
        choices=[m.value for m in NoveltyMode],
        default=NoveltyMode.EMBEDDING.value,
        help="How new blocks are scored for novelty (compression resistance): "
        "embedding | llm | hybrid. See epilog. Default: embedding.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Context window token budget. Blocks are compressed/evicted once this "
        "is exceeded. Set small (e.g. 400) to force the lifecycle to fire often. "
        "Default: 400.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model for the main conversation turns. Default: claude-sonnet-4-6.",
    )
    parser.add_argument(
        "--util-model",
        default="claude-haiku-4-5-20251001",
        help="Cheaper model used for compression summaries, block merges, and "
        "(when --novelty is llm/hybrid) novelty scoring. Default: claude-haiku-4-5-20251001.",
    )
    parser.add_argument(
        "--db",
        default="sqlite:///memory.db",
        help="SQLAlchemy URL for the long-term store. 'sqlite:///memory.db' (default, "
        "zero setup) or a 'postgresql://...' URL for the pgvector backend. The file "
        "persists across runs — point at a fresh path for a clean store.",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model for embeddings (novelty, similarity, "
        "promotion). Default: all-MiniLM-L6-v2 (384-dim, local).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use a local LiteLLM proxy instead of the Anthropic API. "
        "Requires 'ollama serve' and 'litellm --model ollama/<model> --port 4000' "
        "to be running. No ANTHROPIC_API_KEY needed.",
    )
    parser.add_argument(
        "--local-base-url",
        default="http://localhost:4000",
        help="Base URL for the local LiteLLM proxy. Default: http://localhost:4000.",
    )
    parser.add_argument(
        "--local-model",
        default="ollama/llama3.1:8b",
        help="Model name passed to LiteLLM when --local is set. "
        "Default: ollama/llama3.1:8b.",
    )
    args = parser.parse_args()

    if args.local:
        client = anthropic.Anthropic(
            api_key="local",
            base_url=args.local_base_url,
        )
        # Override both the main model and util model with the local model so
        # compress/merge/novelty calls all hit the same local endpoint.
        args.model = args.local_model
        args.util_model = args.local_model
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
            sys.exit(1)
        client = anthropic.Anthropic()

    store = ContextStore(max_tokens=args.max_tokens)
    lt_store = LongTermStore(args.db)
    cm = ContextManager(store, lt_store, embedding_model=args.embedding_model)
    controller = MemoryController(
        cm,
        compress_fn=make_compress_fn(client, args.util_model),
        merge_fn=make_merge_fn(client, cm, args.util_model),
    )
    agent = Agent(
        controller,
        client,
        model=args.model,
        mode=MemoryMode(args.mode),
        novelty_mode=NoveltyMode(args.novelty),
        novelty_model=args.util_model,
    )

    backend = f"local ({args.local_base_url})" if args.local else "anthropic"
    print(f"Memory manager REPL — mode={args.mode}, novelty={args.novelty}, budget={args.max_tokens} tokens, backend={backend}")
    print("Commands: /status (show memory), /reset (clear chat history), /quit\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/status":
            print_memory(cm)
            continue
        if user_input == "/reset":
            agent._messages.clear()
            print("Chat history cleared (memory store preserved).")
            continue

        try:
            reply = agent.chat(user_input)
        except Exception as e:
            print(f"  [chat error: {e}]", file=sys.stderr)
            continue

        print(f"\nassistant> {reply}")
        print_memory(cm)


if __name__ == "__main__":
    main()
