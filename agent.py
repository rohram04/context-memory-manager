from __future__ import annotations

from enum import Enum
from typing import Callable, Iterator

import anthropic

from controller import MemoryController
from functions.memory_tools import MemoryTools
from memory.novelty import NoveltyMode, build_novelty_fn


class MemoryMode(str, Enum):
    ALGORITHMIC = "algorithmic"
    LLM = "llm"


_ALGO_INSTRUCTIONS = (
    "Your context window is actively managed. Memory blocks are compressed and evicted "
    "automatically based on novelty and recency — high-novelty, frequently-accessed "
    "information is preserved longest. The MEMORY STATUS block below shows the current "
    "state of your context. Focus on your task; memory management is handled for you."
)

_LLM_INSTRUCTIONS = """\
You manage your own context window using structured function calls (tools). \
The MEMORY STATUS block below shows all active blocks with their novelty, decay, and token costs.

You may — and should — call multiple tools in a single turn before writing your reply. \
Follow this sequence each turn:

1. SURFACE (call first, if the topic may have LT history):
   - Call query_lt(query) to check long-term memory for relevant blocks.
   - If useful results come back, call promote(lt_block_id) to bring them into context.

2. PERSIST (call next, for any new information from this turn):
   - Scan MEMORY BLOCKS for a block that overlaps in topic.
   - If one exists, call augment(block_id, new_content) — write the FULL merged result. Combine existing facts and new information into one coherent block; drop anything superseded; do not concatenate.
   - If no existing block covers it, call store(content, novelty) with the new content.

3. FREE BUDGET (call next, when pressure is medium or higher):
   - Call compress(block_id, compressed_content) on low-novelty, low-decay blocks to shrink them.
   - Call evict(block_id) on blocks unlikely to be needed soon to remove them entirely.
   - You may compress or evict multiple blocks in one turn.

4. RE-SCORE (call if needed):
   - Call update_novelty(block_id, novelty) when new information changes how significant a block is.

5. REPLY:
   - After all tool calls, always write a conversational text response to the user.

Rules:
- Prefer augment over store whenever an existing block is topically related — duplicate blocks waste budget.
- When augmenting, the merged block must not repeat facts already in the block; integrate, don't concatenate.
- Use the structured tool call interface only — do not write tool calls as plain text or JSON.\
"""


def _extract_text(content: list) -> str:
    """Concatenate every text block in a response (text can be split across
    multiple blocks when interleaved with tool_use); empty string if none."""
    return "".join(block.text for block in content if hasattr(block, "text"))


class Agent:
    """Turn-by-turn conversation loop with pluggable memory management mode."""

    def __init__(
        self,
        controller: MemoryController,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        mode: MemoryMode = MemoryMode.ALGORITHMIC,
        novelty_mode: NoveltyMode = NoveltyMode.EMBEDDING,
        novelty_model: str = "claude-haiku-4-5-20251001",
        system_prompt: str = "",
        max_tool_iterations: int = 10,
        on_receive: Callable[[str, str], None] | None = None,
    ) -> None:
        self._controller = controller
        self._on_receive = on_receive
        self._client = client
        self._model = model
        self._mode = mode
        self._novelty_fn = build_novelty_fn(
            novelty_mode, controller._cm, client, novelty_model
        )
        self._tools = MemoryTools(controller._cm)
        self._system_prompt = system_prompt
        self._max_tool_iterations = max_tool_iterations
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
        ("final", final_message) with the completed message (stop_reason +
        content blocks)."""
        kwargs = dict(
            model=self._model, max_tokens=4096, system=system, messages=messages
        )
        if tools is not None:
            kwargs["tools"] = tools
        with self._client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                yield ("token", delta)
            final = stream.get_final_message()
        yield ("final", final)

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
        for ev, val in self._stream_model_call(
            self._messages, self._build_system_prompt()
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

        # Work on a local copy so intermediate tool-use turns never enter self._messages
        messages = list(self._messages)
        system = self._build_system_prompt()
        text = ""

        for _ in range(self._max_tool_iterations):
            final = None
            for ev, val in self._stream_model_call(
                messages, system, tools=MemoryTools.SCHEMAS
            ):
                if ev == "token":
                    text += val
                    yield {"type": "token", "delta": val}
                else:
                    final = val

            if final.stop_reason == "end_turn":
                break

            if final.stop_reason == "tool_use":
                tool_results = []
                for block in final.content:
                    if block.type == "tool_use":
                        result = self._tools.dispatch(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": final.content})
                messages.append({"role": "user", "content": tool_results})
                system = self._build_system_prompt()
                continue

            break

        self._messages.append({"role": "assistant", "content": text})
        if self._on_receive:
            self._on_receive("assistant", text)
        yield {"type": "assistant", "text": text}

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
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=self._messages,
        )
        text = _extract_text(response.content)
        text_embedding = self._controller.embed(text)
        self._controller.receive(
            text, text_embedding, self._novelty_fn(text, text_embedding)
        )
        if self._on_receive:
            self._on_receive("assistant", text)
        return text

    def _llm_turn(self, user_message: str) -> str:
        # Snapshot hook (parity with _algorithmic_turn): capture pre-turn state for the
        # user message (the model has not acted on memory yet), then post-turn state
        # after it finishes self-managing memory via tools. Gives the demo two
        # per-turn snapshots in LLM mode just like algorithmic mode.
        if self._on_receive:
            self._on_receive("user", user_message)
        # Work on a local copy so intermediate tool-use turns never enter self._messages
        messages = list(self._messages)
        system = self._build_system_prompt()

        for _ in range(self._max_tool_iterations):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=MemoryTools.SCHEMAS,
            )

            if response.stop_reason == "end_turn":
                text = _extract_text(response.content)
                if self._on_receive:
                    self._on_receive("assistant", text)
                return text

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._tools.dispatch(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                system = self._build_system_prompt()
                continue

            break

        raise RuntimeError(
            f"Tool-use loop exceeded {self._max_tool_iterations} iterations"
        )

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        instructions = (
            _ALGO_INSTRUCTIONS
            if self._mode == MemoryMode.ALGORITHMIC
            else _LLM_INSTRUCTIONS
        )
        parts = [p for p in [self._system_prompt, instructions] if p]
        parts.append(self._controller.build_memory_prompt())
        return "\n\n".join(parts)
