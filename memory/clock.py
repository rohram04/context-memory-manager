from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# Fixed epoch for simulated mode so runs are deterministic and reproducible.
_SIM_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)


class Clock:
    """Per-agent injectable clock used by memory blocks for decay timing.

    Default mode is wall-clock: now() returns real time, so production behaviour
    is unchanged. In simulated mode (seconds_per_turn > 0) the clock starts at a
    fixed epoch and only advances when tick() is called — once per turn
    (MemoryController.receive() in algorithmic mode, or _llm_persist_phase in
    LLM mode). This makes the recency/decay signal meaningful and deterministic
    during fast batch evaluations, where real elapsed time between ingested items
    is ~0 and every block's decay_score would otherwise be ~1.0.

    Each ContextManager owns one Clock instance, shared with its ContextStore and
    LongTermStore, so concurrent agents never stomp each other's simulated time.
    """

    def __init__(self, seconds_per_turn: float = 0.0) -> None:
        self._simulated = seconds_per_turn > 0.0
        self._seconds_per_turn = seconds_per_turn
        self._t = _SIM_EPOCH

    def tick(self) -> None:
        """Advance simulated time by one turn. No-op in wall-clock mode."""
        if self._simulated:
            self._t = self._t + timedelta(seconds=self._seconds_per_turn)

    def now(self) -> datetime:
        return self._t if self._simulated else datetime.now(UTC)
