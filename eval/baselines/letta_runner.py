"""Real Letta (MemGPT) agent runner for live needle-in-a-haystack eval.

Requires: pip install letta-client
Connect to Letta Cloud (LETTA_API_KEY) or self-hosted server (--letta-base-url).
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

_AGENT_NAME_PREFIX = "niah_eval_"


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
        meter=None,
        system: str = "letta",
        timeout: float = 600.0,
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
        # A single big-context turn near overflow triggers Sonnet reasoning + memory
        # tool calls + summarization on tens of thousands of tokens, which blows the
        # client's ~60s default per-request timeout. Raise it so the run doesn't die
        # mid-stream (the model/server are fine; the client was giving up too early).
        kwargs["timeout"] = timeout
        self._client = Letta(**kwargs)

        self._sweep_stale_agents()

        # Drive Letta as the conversational agent it is designed to be: neutral persona
        # and human blocks with NO "memorize everything" directive, and let the agent
        # decide autonomously what to keep in core memory vs. page out to archival
        # (its intended overflow mechanism). Earlier configs commanded total retention
        # (persona "I store every fact", per-message "Memorize this... move to archival
        # if full"), which hoarded everything into the un-pageable core memory and
        # overflowed the system prompt → CONTEXT_WINDOW_EXCEEDED. Default block limits;
        # with conversational ingest core memory stays lean, so no window-fitting cap.
        create_kwargs: dict[str, Any] = {
            "name": f"{_AGENT_NAME_PREFIX}{uuid4().hex[:12]}",
            "model": model,
            "context_window_limit": context_window_limit,
            "memory_blocks": [
                {
                    "label": "human",
                    "value": "The user chats about a variety of topics over many turns.",
                    "limit": 2000,
                },
                {
                    "label": "persona",
                    "value": (
                        "I am a helpful conversational assistant. I keep track of "
                        "important details the user shares and recall them when asked."
                    ),
                    "limit": 1000,
                },
            ],
        }
        if embedding:
            create_kwargs["embedding"] = embedding

        self._agent = self._client.agents.create(**create_kwargs)
        self.agent_id = self._agent.id
        self._meter = meter
        self._system = system
        self._model = model

    def _sweep_stale_agents(self) -> None:
        """Best-effort delete of leftover agents from crashed/killed prior runs.

        Any agent whose name starts with _AGENT_NAME_PREFIX is a leftover from a
        previous trial that failed to clean up after itself. Deleting them before
        creating a fresh agent prevents archival memory from one trial leaking into
        the next. All failures are swallowed so this never aborts a real eval run.
        """
        try:
            agents = self._client.agents.list()
        except Exception:
            return
        try:
            iterator = iter(agents)
        except TypeError:
            return
        for agent in iterator:
            try:
                name = getattr(agent, "name", None)
                if not name or not str(name).startswith(_AGENT_NAME_PREFIX):
                    continue
                agent_id = getattr(agent, "id", None)
                if agent_id is None:
                    continue
                self._client.agents.delete(agent_id)
            except Exception:
                continue

    def _meter_response(self, response) -> None:
        if self._meter is not None:
            self._meter.add_letta_response(self._system, self._model, response)

    def ingest(self, content: str) -> None:
        # Feed each stream item as a plain conversational turn (no "memorize this /
        # reply stored" directive) so Letta decides for itself what is salient enough
        # to remember and pages the rest out — exercising its overflow design instead
        # of being forced to hoard everything into core memory.
        response = self._client.agents.messages.create(
            agent_id=self.agent_id,
            input=content,
        )
        self._meter_response(response)

    def query(self, question: str) -> str:
        response = self._client.agents.messages.create(
            agent_id=self.agent_id,
            input=question,
        )
        self._meter_response(response)
        return _extract_assistant_text(response)

    def close(self) -> None:
        try:
            self._client.agents.delete(self.agent_id)
        except Exception:
            pass
