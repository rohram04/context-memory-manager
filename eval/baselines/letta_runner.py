"""Real Letta (MemGPT) agent runner for live needle-in-a-haystack eval.

Requires: pip install letta-client
Connect to Letta Cloud (LETTA_API_KEY) or self-hosted server (--letta-base-url).
"""

from __future__ import annotations

import os
from typing import Any

_INGEST_PROMPT = """\
Memorize the following fact for later recall. If your context is full, move it to \
archival memory. Reply with exactly: stored

FACT: {content}"""


def _extract_assistant_text(response: Any) -> str:
    """Pull assistant-visible text from a Letta messages.create response."""
    parts: list[str] = []
    for message in getattr(response, "messages", []) or []:
        msg_type = getattr(message, "message_type", None) or getattr(message, "type", None)
        if msg_type in ("assistant_message", "assistant"):
            content = getattr(message, "content", None) or getattr(message, "text", None)
            if content:
                parts.append(str(content))
    return "\n".join(parts)


class LettaRunner:
    """One agent per trial — ingest stream items, then answer probe queries."""

    def __init__(
        self,
        model: str,
        context_window_limit: int,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        embedding: str | None = None,
    ) -> None:
        try:
            from letta_client import Letta
        except ImportError as e:
            raise ImportError(
                "letta-client is required for live Letta eval. "
                "Install with: pip install letta-client"
            ) from e

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        key = api_key or os.environ.get("LETTA_API_KEY")
        if key:
            kwargs["api_key"] = key
        self._client = Letta(**kwargs)

        create_kwargs: dict[str, Any] = {
            "name": "niah_eval_agent",
            "model": model,
            "context_window_limit": context_window_limit,
            "memory_blocks": [
                {
                    "label": "human",
                    "value": "The user provides facts to memorize across many turns.",
                    "limit": 2000,
                },
                {
                    "label": "persona",
                    "value": (
                        "I am a memory evaluation agent. I store every fact the user "
                        "gives me, using archival memory when needed. I answer questions "
                        "from memory only."
                    ),
                    "limit": 1000,
                },
            ],
        }
        if embedding:
            create_kwargs["embedding"] = embedding

        self._agent = self._client.agents.create(**create_kwargs)
        self.agent_id = self._agent.id

    def ingest(self, content: str) -> None:
        self._client.agents.messages.create(
            agent_id=self.agent_id,
            input=_INGEST_PROMPT.format(content=content),
        )

    def query(self, question: str) -> str:
        response = self._client.agents.messages.create(
            agent_id=self.agent_id,
            input=question,
        )
        return _extract_assistant_text(response)

    def close(self) -> None:
        try:
            self._client.agents.delete(self.agent_id)
        except Exception:
            pass
