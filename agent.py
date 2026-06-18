from __future__ import annotations

from enum import Enum

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

Each turn, call tools as needed, then always end with a plain text reply to the user:
- Before persisting new information, scan MEMORY BLOCKS for any block that overlaps in topic.
  - If one exists, call augment(block_id, new_content) — write the FULL merged block as new_content. Combine the existing content and the new information into a single coherent block with no redundant or repeated facts. Drop anything the new content supersedes; keep what's still true.
  - If no existing block covers it, call store() with the new content.
- Call compress() or evict() to free token budget when pressure is medium or higher.
- Call query_lt() or promote() to surface relevant long-term memories before responding.
- Call update_novelty() when new information changes how significant an existing block is.

Rules:
- Prefer augment over store whenever an existing block is topically related — duplicate blocks waste budget.
- When augmenting, the merged block must not repeat facts that were already in the existing block; integrate, don't concatenate.
- After all tool calls, always write a conversational text response to the user.
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
    ) -> None:
        self._controller = controller
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
            response_text = self._llm_turn()
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
    # Turn handlers
    # ------------------------------------------------------------------

    def _algorithmic_turn(self, user_message: str) -> str:
        user_embedding = self._controller.embed(user_message)
        self._controller.receive(
            user_message, user_embedding, self._novelty_fn(user_message, user_embedding)
        )
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
        return text

    def _llm_turn(self) -> str:
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
                return _extract_text(response.content)

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
