from __future__ import annotations

from typing import Iterator, Literal, Protocol

from llm.types import LLMResponse, StreamEvent


class LLMBackend(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
    ) -> LLMResponse: ...

    def stream_complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]: ...
