from __future__ import annotations

from enum import Enum
from typing import Callable, Iterator

import tiktoken

from controller import MemoryController
from functions.memory_tools import MemoryTools
from llm.interface import LLMBackend
from llm.tool_loop import run_tool_loop
from memory.novelty import NoveltyMode, build_novelty_fn

# Same encoder CacheBlock.token_cost uses, so the room check in PREP matches the
# token accounting the store reports.
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


class MemoryMode(str, Enum):
    ALGORITHMIC = "algorithmic"
    LLM = "llm"


_ALGO_INSTRUCTIONS = (
    "Your context window is actively managed. Memory blocks are compressed and evicted "
    "automatically based on novelty and recency — high-novelty, frequently-accessed "
    "information is preserved longest. The MEMORY STATUS block below shows the current "
    "state of your context. Focus on your task; memory management is handled for you."
)

# The LLM turn runs in three phases. PHASE 1 (PREP): surface relevant memory AND free
# budget — tool calls only, no reply — so there is room within budget before the reply
# is generated. PHASE 2 (REPLY): plain text against the now-fitted context, no tools.
# PHASE 3 (PERSIST): persist the exchange and re-score — tool calls only. Splitting this
# way frees budget BEFORE the reply (avoiding context overflow), keeps the reply isolated
# as phase-2's plain text (clean token streaming), and lets the model persist its OWN
# reply (phase 3 sees it). PREP/PERSIST are tool-only with no user-facing text, so the
# same blocking phase methods serve the blocking turn, the streaming turn, and the eval.

_LLM_PREP_INSTRUCTIONS = """\
You manage your own long-term memory. Before you reply to the user, prepare your context \
window — using ONLY tool calls, write no prose. The MEMORY STATUS block below shows your \
current memory blocks (novelty, fidelity, token cost, decay) and budget pressure.

1. SURFACE (only if the topic may have history): call query_lt(query) to search long-term
   memory, and promote(lt_block_id) to bring any genuinely relevant blocks into context so
   your answer can use them.
2. FREE BUDGET: if budget pressure is medium or higher (or surfacing pushed you over),
   compress(block_id, compressed_content) low-novelty/low-decay blocks, or evict(block_id)
   blocks unlikely to be needed for this turn — make room before you reply.

Do NOT reply to the user here and do NOT store/augment/re-score — that happens in later
phases. When your context is surfaced and within budget, stop (end your turn)."""

_LLM_PERSIST_INSTRUCTIONS = """\
You just replied to the user. The exchange above combines the user's message and your \
reply in one turn. Update your managed memory to reflect this exchange, using ONLY tool \
calls — write no prose.

1. PERSIST — default: ONE combined block. Call store(content, novelty) or augment(\
block_id, new_content) with content that synthesizes the full exchange (integrate what \
the user said and what you replied; drop superseded details). Prefer augment over store \
when a block already covers the topic. Write the FULL merged content — integrate, don't \
concatenate.

   Split into separate blocks ONLY when clearly justified:
   - The exchange contains multiple unrelated facts (one store/augment per distinct topic).
   - Only the user's side is worth remembering and your reply is generic boilerplate \
(store the durable user fact alone; skip the fluff).
   - User and assistant content need different novelty scores or lifecycles.

   Do NOT split by default — mirroring the raw user message and your reply as two blocks \
for the same topic is wrong.

2. RE-SCORE: call update_novelty(block_id, novelty) where significance has changed.

When memory reflects this exchange, stop (end your turn without writing text)."""

# Minimal user-role nudge that opens phase 3. A trailing user turn is REQUIRED — the
# messages array cannot end on the assistant reply (400 on Sonnet 4.6 / the 4.6+ family).
# This must NOT restate _LLM_PERSIST_INSTRUCTIONS; the detailed how-to lives there.
_PERSIST_NUDGE = "Now persist this exchange and re-score. Use tool calls only, then stop."


def _format_exchange(user_message: str, reply: str) -> str:
    """Single user-role payload for persist phase (user + assistant in one block)."""
    return f"User: {user_message}\n\nAssistant: {reply}"


class Agent:
    """Turn-by-turn conversation loop with pluggable memory management mode."""

    def __init__(
        self,
        controller: MemoryController,
        backend: LLMBackend,
        model: str = "claude-sonnet-4-6",
        mode: MemoryMode = MemoryMode.ALGORITHMIC,
        novelty_mode: NoveltyMode = NoveltyMode.EMBEDDING,
        novelty_model: str = "claude-haiku-4-5-20251001",
        system_prompt: str = "",
        max_tool_iterations: int = 10,
        max_prep_refires: int = 3,
        on_receive: Callable[[str, str], None] | None = None,
    ) -> None:
        self._controller = controller
        self._on_receive = on_receive
        self._backend = backend
        self._model = model
        self._mode = mode
        self._novelty_fn = build_novelty_fn(
            novelty_mode, controller._cm, backend, novelty_model
        )
        self._tools = MemoryTools(controller._cm)
        self._system_prompt = system_prompt
        self._max_tool_iterations = max_tool_iterations
        self._max_prep_refires = max_prep_refires
        self._messages: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the assistant's response."""
        self._messages.append({"role": "user", "content": user_message})
        if self._mode == MemoryMode.LLM:
            response_text = self._llm_turn(user_message)
        else:
            response_text = self._algorithmic_turn(user_message)
        self._messages.append({"role": "assistant", "content": response_text})
        return response_text

    def ingest(self, content: str) -> None:
        """Push content into memory without firing a conversational turn.

        Used for batch context ingestion (e.g. benchmark memorization phase) where
        we want the memory lifecycle (promote → insert/augment → fit budget) but
        not the cost of an LLM reply per chunk.
        """
        embedding = self._controller.embed(content)
        self._controller.receive(
            content, embedding, self._novelty_fn(content, embedding)
        )

    # ------------------------------------------------------------------
    # Streaming API (demo-facing; mirrors chat() turn logic, token-by-token)
    # ------------------------------------------------------------------

    def _stream_model_call(self, messages, system, tools=None):
        """Stream one model call. Yields ("token", delta) per text delta, then
        ("final", final_message) with the completed message."""
        for event in self._backend.stream_complete(
            model=self._model,
            messages=messages,
            system=system,
            max_tokens=4096,
            tools=tools,
        ):
            yield event

    def stream_chat(self, user_message: str) -> Iterator[dict]:
        """Streamed analogue of chat(): yields demo-agnostic events
        ({"type":"user"}, {"type":"token","delta":..}, {"type":"assistant","text":..}).
        Both memory modes stream token-by-token via the shared _stream_model_call."""
        self._messages.append({"role": "user", "content": user_message})
        if self._mode == MemoryMode.LLM:
            yield from self._stream_llm_turn(user_message)
        else:
            yield from self._stream_algorithmic_turn(user_message)

    def _stream_algorithmic_turn(self, user_message: str) -> Iterator[dict]:
        user_embedding = self._controller.embed(user_message)
        self._controller.receive(
            user_message, user_embedding, self._novelty_fn(user_message, user_embedding)
        )
        if self._on_receive:
            self._on_receive("user", user_message)
        yield {"type": "user"}

        text = ""
        # Context = memory blocks (system) + only the current turn; prior turns live in
        # the blocks, so the prompt stays bounded by the memory budget.
        for ev, val in self._stream_model_call(
            [{"role": "user", "content": user_message}], self._build_system_prompt()
        ):
            if ev == "token":
                text += val
                yield {"type": "token", "delta": val}

        text_embedding = self._controller.embed(text)
        self._controller.receive(
            text, text_embedding, self._novelty_fn(text, text_embedding)
        )
        if self._on_receive:
            self._on_receive("assistant", text)
        self._messages.append({"role": "assistant", "content": text})
        yield {"type": "assistant", "text": text}

    def _stream_llm_turn(self, user_message: str) -> Iterator[dict]:
        if self._on_receive:
            self._on_receive("user", user_message)
        yield {"type": "user"}

        # Phase 1 — PREP: surface relevant memory + free budget (tool-only, no streaming).
        # Shares the blocking phase method with _llm_turn and the eval path.
        self._llm_prep_phase(user_message)

        # Phase 2 — REPLY: stream plain text against the now-fitted context (no tools).
        # The reply is the only prose, so accumulated tokens == the reply (no pollution).
        reply = ""
        for ev, val in self._stream_model_call(
            [{"role": "user", "content": user_message}], self._build_system_prompt("")
        ):
            if ev == "token":
                reply += val
                yield {"type": "token", "delta": val}

        # Phase 3 — PERSIST: persist the exchange (incl. the reply) + re-score (tool-only).
        self._llm_persist_phase(user_message, reply)

        self._messages.append({"role": "assistant", "content": reply})
        if self._on_receive:
            self._on_receive("assistant", reply)  # snapshot reflects the persisted reply
        yield {"type": "assistant", "text": reply}

    # ------------------------------------------------------------------
    # Turn handlers
    # ------------------------------------------------------------------

    def _algorithmic_turn(self, user_message: str) -> str:
        user_embedding = self._controller.embed(user_message)
        self._controller.receive(
            user_message, user_embedding, self._novelty_fn(user_message, user_embedding)
        )
        if self._on_receive:
            self._on_receive("user", user_message)
        # Context = memory blocks (system) + only the current turn (see note above).
        response = self._backend.complete(
            model=self._model,
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.text
        text_embedding = self._controller.embed(text)
        self._controller.receive(
            text, text_embedding, self._novelty_fn(text, text_embedding)
        )
        if self._on_receive:
            self._on_receive("assistant", text)
        return text

    def _blocking_tool_loop(self, messages, instructions, tools) -> str:
        """Blocking tool loop until end_turn. Returns accumulated assistant text."""
        return run_tool_loop(
            self._backend,
            model=self._model,
            messages=messages,
            system=self._build_system_prompt(instructions),
            tools=tools,
            dispatch=self._tools.dispatch,
            max_iterations=self._max_tool_iterations,
        )

    def _llm_prep_phase(self, user_message: str) -> None:
        """Phase 1 — SURFACE + FREE BUDGET. Loop the LLM (query_lt/promote/compress/evict)
        until there's room within budget for the incoming message, then a hard algorithmic
        backstop. Tool-only (no user-facing text), so it is shared by the blocking turn,
        the streaming turn, and the eval path."""
        for _ in range(self._max_prep_refires):
            self._blocking_tool_loop(
                [{"role": "user", "content": user_message}],
                _LLM_PREP_INSTRUCTIONS, MemoryTools.PREP_TOOLS,
            )
            if self._has_room_for(user_message):
                return
        # Backstop: the model didn't free enough room in max_prep_refires passes.
        # Algorithmic fit_budget guarantees termination and that the reply is generated
        # within budget (no context overflow). Just-promoted blocks are freshly accessed
        # (high recency → low compression priority), so stale blocks are dropped first.
        self._controller.fit_budget()

    def _llm_persist_phase(self, user_message: str, reply: str) -> None:
        """Phase 3 — PERSIST + RESCORE."""
        self._blocking_tool_loop(
            [
                {"role": "user", "content": _format_exchange(user_message, reply)},
                {"role": "user", "content": _PERSIST_NUDGE},
            ],
            _LLM_PERSIST_INSTRUCTIONS,
            MemoryTools.PERSIST_TOOLS,
        )

    def _has_room_for(self, text: str) -> bool:
        """True when the current context plus the incoming message fits the token budget."""
        cm = self._controller._cm
        return cm.used_tokens + _count_tokens(text) <= cm.max_tokens

    def _llm_turn(self, user_message: str) -> str:
        # Three-phase (parity with the streaming path): PREP surfaces relevant memory and
        # frees budget so the reply is generated within budget; REPLY is the sole plain
        # text; PERSIST stores the exchange and re-scores. on_receive("user") captures
        # pre-turn state; on_receive("assistant") fires after PERSIST so the snapshot
        # reflects the persisted reply.
        if self._on_receive:
            self._on_receive("user", user_message)

        self._llm_prep_phase(user_message)

        response = self._backend.complete(
            model=self._model,
            max_tokens=4096,
            system=self._build_system_prompt(""),
            messages=[{"role": "user", "content": user_message}],
        )
        reply = response.text

        self._llm_persist_phase(user_message, reply)

        if self._on_receive:
            self._on_receive("assistant", reply)
        return reply

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, instructions: str | None = None) -> str:
        # LLM-mode phases always pass explicit instructions (PREP/PERSIST) or "" (REPLY),
        # so the default branch only supplies the algorithmic mode's standing instructions.
        if instructions is None:
            instructions = (
                _ALGO_INSTRUCTIONS if self._mode == MemoryMode.ALGORITHMIC else ""
            )
        parts = [p for p in [self._system_prompt, instructions] if p]
        parts.append(self._controller.build_memory_prompt())
        return "\n\n".join(parts)
