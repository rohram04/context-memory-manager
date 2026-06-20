from __future__ import annotations

from typing import Callable

from llm.interface import LLMBackend
from llm.types import StopReason


def run_tool_loop(
    backend: LLMBackend,
    *,
    model: str,
    messages: list[dict],
    system: str,
    tools: list[dict],
    dispatch: Callable[[str, dict], str],
    max_iterations: int = 10,
    max_tokens: int = 4096,
) -> str:
    """Blocking tool loop until end_turn. Returns accumulated assistant text and
    mutates `messages` in place."""
    text = ""
    for _ in range(max_iterations):
        response = backend.complete(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            tools=tools,
        )
        text += response.text
        if response.stop_reason == StopReason.END_TURN:
            return text
        if response.stop_reason == StopReason.TOOL_USE and response.tool_calls:
            tool_results = []
            for call in response.tool_calls:
                result = dispatch(call.name, call.arguments)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                })
            messages.append({
                "role": "assistant",
                "content": _assistant_content(response),
            })
            messages.append({"role": "user", "content": tool_results})
            continue
        break
    return text


def _assistant_content(response) -> list | str:
    """Build canonical assistant message content for history append."""
    if response.raw is not None:
        raw = response.raw
        if hasattr(raw, "content"):
            return raw.content
    blocks: list[dict] = []
    if response.text:
        blocks.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        blocks.append({
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.arguments,
        })
    return blocks if blocks else ""
