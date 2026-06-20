from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    OTHER = "other"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    raw: Any = None


StreamEvent = tuple[Literal["token"], str] | tuple[Literal["final"], LLMResponse]


def extract_text(content: list | str) -> str:
    """Concatenate text from a response content list or plain string."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            continue
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)
