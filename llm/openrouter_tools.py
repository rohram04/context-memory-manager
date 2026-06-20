from __future__ import annotations

import json
from typing import Iterator, Literal

from openai import OpenAI

from llm.interface import LLMBackend
from llm.types import LLMResponse, StopReason, StreamEvent, ToolCall


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert canonical tool schemas (input_schema) to OpenAI function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["input_schema"],
            },
        }
        for schema in schemas
    ]


class OpenRouterBackend:
    """OpenAI chat-completions backend for OpenRouter (all models)."""

    def __init__(self, client: OpenAI) -> None:
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
        oai_messages = _to_openai_messages(messages, system)
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        if tools is not None:
            kwargs["tools"] = to_openai_tools(tools)
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"
            elif tool_choice == "auto":
                kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return _normalize_openai_response(response)

    def stream_complete(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        oai_messages = _to_openai_messages(messages, system)
        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            messages=oai_messages,
            stream=True,
        )
        if tools is not None:
            kwargs["tools"] = to_openai_tools(tools)

        stream = self._client.chat.completions.create(**kwargs)
        text_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None
        raw_chunks = []

        for chunk in stream:
            raw_chunks.append(chunk)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta.content:
                text_parts.append(delta.content)
                yield ("token", delta.content)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    acc = tool_calls_acc[idx]
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function.arguments:
                            acc["arguments"] += tc.function.arguments

        tool_calls = _parse_accumulated_tool_calls(tool_calls_acc)
        stop_reason = _map_finish_reason(finish_reason, tool_calls)
        yield ("final", LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            raw=raw_chunks,
        ))


def _to_openai_messages(messages: list[dict], system: str) -> list[dict]:
    """Translate canonical (Anthropic-shaped) messages to OpenAI format."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and isinstance(content, list):
            # Anthropic tool_result blocks -> OpenAI tool messages
            if content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                for block in content:
                    out.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": block.get("content", ""),
                    })
                continue

        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        })
                elif hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input or {}),
                            },
                        })
            assistant: dict = {"role": "assistant"}
            joined = "".join(text_parts)
            assistant["content"] = joined if joined else None
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            out.append(assistant)
            continue

        out.append({"role": role, "content": content})

    return out


def _normalize_openai_response(response) -> LLMResponse:
    choice = response.choices[0]
    message = choice.message
    tool_calls = _parse_openai_tool_calls(message.tool_calls)
    stop_reason = _map_finish_reason(choice.finish_reason, tool_calls)
    return LLMResponse(
        text=message.content or "",
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        raw=response,
    )


def _parse_openai_tool_calls(tool_calls) -> list[ToolCall]:
    if not tool_calls:
        return []
    out: list[ToolCall] = []
    for tc in tool_calls:
        args_raw = tc.function.arguments or "{}"
        try:
            arguments = json.loads(args_raw)
        except json.JSONDecodeError:
            arguments = {}
        out.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    return out


def _parse_accumulated_tool_calls(acc: dict[int, dict]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for idx in sorted(acc):
        entry = acc[idx]
        try:
            arguments = json.loads(entry["arguments"] or "{}")
        except json.JSONDecodeError:
            arguments = {}
        out.append(ToolCall(
            id=entry["id"],
            name=entry["name"],
            arguments=arguments,
        ))
    return out


def _map_finish_reason(finish_reason: str | None, tool_calls: list[ToolCall]) -> StopReason:
    # Tool calls take precedence over the textual finish_reason. Some OpenRouter
    # providers report finish_reason="stop" even when tool calls were emitted, so
    # checking "stop" first would map to END_TURN and silently drop the calls
    # without ever dispatching them.
    if tool_calls or finish_reason == "tool_calls":
        return StopReason.TOOL_USE
    if finish_reason == "stop":
        return StopReason.END_TURN
    return StopReason.OTHER
