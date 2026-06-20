from __future__ import annotations

from typing import Iterator, Literal

import anthropic

from llm.interface import LLMBackend
from llm.types import LLMResponse, StopReason, StreamEvent, ToolCall, extract_text


class AnthropicBackend:
    """Anthropic Messages API backend."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
        elif tool_choice == "auto":
            kwargs["tool_choice"] = {"type": "auto"}

        response = self._client.messages.create(**kwargs)
        return _normalize_response(response)

    def stream_complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools

        with self._client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                yield ("token", delta)
            final = stream.get_final_message()
        yield ("final", _normalize_response(final))


def _normalize_response(response) -> LLMResponse:
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input or {},
            ))

    stop = response.stop_reason
    if stop == "tool_use":
        stop_reason = StopReason.TOOL_USE
    elif stop == "end_turn":
        stop_reason = StopReason.END_TURN
    else:
        stop_reason = StopReason.OTHER

    return LLMResponse(
        text=extract_text(response.content),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        raw=response,
    )
