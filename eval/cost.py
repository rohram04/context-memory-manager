"""Cost & token metering for the eval harnesses.

Cost is a first-class metric here, not just a budget guard: the algorithmic
manager makes far fewer LLM calls per turn than an LLM-agent system like Letta,
so we report tokens + USD per system and recall-per-dollar alongside accuracy.

Pricing is USD per *single* token (input, output), approximate list price.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# (input_per_token, output_per_token) USD. Update if list prices change.
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3e-6, 15e-6),
    "claude-haiku-4-5-20251001": (1e-6, 5e-6),
    # OpenRouter ids (Anthropic pass-through; same list price).
    "claude-sonnet-4.6": (3e-6, 15e-6),
    "claude-haiku-4.5": (1e-6, 5e-6),
}
# Provider-prefixed ids (e.g. "openrouter/anthropic/claude-sonnet-4.6"); strip to match.
# Order matters: outer provider prefixes first, then anthropic/.
_PREFIXES = ("openrouter/", "anthropic/", "openai/", "google_ai/")


class BudgetExceeded(Exception):
    """Raised when accumulated spend crosses the configured --max-spend cap."""


class _MeteredMessages:
    def __init__(self, inner, meter: "CostMeter", system: str) -> None:
        self._inner, self._meter, self._system = inner, meter, system

    def create(self, **kwargs):
        response = self._inner.create(**kwargs)
        self._meter.add_response(self._system, kwargs.get("model", "unknown"), response)
        return response


class _MeteredClient:
    """Wraps an Anthropic client so every messages.create() is metered.

    Routes all of a system's LLM calls (compress, merge, novelty-llm, probe)
    through the CostMeter, so --max-spend bounds them and the reported cost is
    complete — not just the probe calls.
    """

    def __init__(self, client, meter: "CostMeter", system: str) -> None:
        self._client = client
        self.messages = _MeteredMessages(client.messages, meter, system)

    def __getattr__(self, name):
        return getattr(self._client, name)


def _price(model: str) -> tuple[float, float]:
    m = model
    for p in _PREFIXES:
        if m.startswith(p):
            m = m[len(p):]
    try:
        return PRICING[m]
    except KeyError:
        # No silent Sonnet fallback: an unknown id means cost would be fabricated,
        # not measured. Fail loudly so the real list price is added to PRICING.
        raise ValueError(
            f"No pricing for model {model!r} (normalized to {m!r}). "
            f"Add its list price to PRICING in eval/cost.py. "
            f"Known: {sorted(PRICING)}."
        ) from None


@dataclass
class _Tally:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    usd: float = 0.0


@dataclass
class CostMeter:
    """Accumulates token usage and USD per system, with an optional hard cap."""

    max_spend: float | None = None
    tallies: dict[str, _Tally] = field(default_factory=dict)

    def _tally(self, system: str) -> _Tally:
        return self.tallies.setdefault(system, _Tally())

    def add(self, system: str, model: str, input_tokens: int, output_tokens: int) -> None:
        in_price, out_price = _price(model)
        t = self._tally(system)
        t.input_tokens += input_tokens
        t.output_tokens += output_tokens
        t.calls += 1
        t.usd += input_tokens * in_price + output_tokens * out_price
        self._enforce()

    def add_response(self, system: str, model: str, response: Any) -> None:
        """Record usage from an Anthropic messages.create response (response.usage)."""
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        self.add(system, model, in_tok, out_tok)

    def add_letta_response(self, system: str, model: str, response: Any) -> None:
        """Record usage from a Letta messages.create response.

        Letta exposes a `usage` object whose field names vary across client
        versions; probe the common ones and fall back to 0 (recorded as a call
        with no measurable tokens) so totals never silently double-count.
        """
        usage = getattr(response, "usage", None)
        in_tok = int(
            getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "input_tokens", 0)
            or 0
        )
        out_tok = int(
            getattr(usage, "completion_tokens", 0)
            or getattr(usage, "output_tokens", 0)
            or 0
        )
        self.add(system, model, in_tok, out_tok)

    def total_usd(self) -> float:
        return sum(t.usd for t in self.tallies.values())

    def _enforce(self) -> None:
        if self.max_spend is not None and self.total_usd() > self.max_spend:
            raise BudgetExceeded(
                f"spend ${self.total_usd():.2f} exceeded cap ${self.max_spend:.2f}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd(), 4),
            "by_system": {
                s: {
                    "input_tokens": t.input_tokens,
                    "output_tokens": t.output_tokens,
                    "calls": t.calls,
                    "usd": round(t.usd, 4),
                }
                for s, t in self.tallies.items()
            },
        }

    def wrap_client(self, client, system: str):
        """Return a drop-in client whose messages.create() is metered + budget-capped."""
        return _MeteredClient(client, self, system)

    def print_tally(self, prefix: str = "  [cost]") -> None:
        for s, t in self.tallies.items():
            print(
                f"{prefix} {s}: {t.calls} calls, "
                f"{t.input_tokens}+{t.output_tokens} tok, ${t.usd:.4f}",
                flush=True,
            )
        print(f"{prefix} TOTAL ${self.total_usd():.4f}", flush=True)
