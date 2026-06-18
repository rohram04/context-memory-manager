"""Pre-baked demo scripts.

Each script is a sequence of chat messages that run through the normal
``agent.chat`` path (so they produce the same recorded per-message snapshots as
a manual chat). They reuse the needle/distractor generators from the stress
test so the seeded "needle" facts are memorable and the distractors realistically
flood the budget.
"""

from __future__ import annotations

from typing import Callable

from eval.stress_test import generate_needles

_SEED = 7

# A single fixed needle so the recall probe is deterministic.
_NEEDLES = generate_needles(2, seed=_SEED)
_NEEDLE = _NEEDLES[0]
_MID_NEEDLE = _NEEDLES[1]

# Curated, semantically DISTINCT distractors. The controller augment-merges
# topically similar blocks (cosine >= 0.85) into one compact block, so generic
# filler never grows the context past budget. These span unrelated domains so
# each becomes its own block — that is what reliably drives budget pressure and
# fires the compression/eviction lifecycle under a small (400-token) budget.
_DISTRACTORS = [
    "I'm planning a trip to Lisbon in October and can't decide between staying in Alfama or Bairro Alto.",
    "My sourdough starter is finally active — I named it Bartholomew and feed it rye flour every morning.",
    "At work we migrated our billing service from REST to gRPC last sprint and tail latency dropped about 40%.",
    "I adopted a three-legged greyhound this weekend; her name is Comet and she sleeps roughly 18 hours a day.",
    "I'm reading 'The Left Hand of Darkness' and the worldbuilding around ambisexual physiology is fascinating.",
    "I'm thinking about converting my road bike to tubeless tires before the gravel race next month.",
    "The community garden gave me plot 14, so I'm putting in heirloom tomatoes and Thai basil this spring.",
    "I started learning the cello at 34 and I'm slowly fighting my way through Suzuki book one.",
    "We repainted the kitchen a deep forest green and it makes the old brass fixtures really pop.",
    "I'm trying to cut down to one coffee a day and switch to matcha in the afternoons.",
]


def _needle_msg(needle: dict) -> str:
    return needle["content"]


def _recall_msg(needle: dict) -> str:
    return needle["query"]


def _flood(n: int) -> list[str]:
    """First n distinct distractors (cycles if n exceeds the pool)."""
    return [_DISTRACTORS[i % len(_DISTRACTORS)] for i in range(n)]


SCRIPTS: dict[str, dict] = {
    "early_fact": {
        "description": "Seed one memorable needle fact early.",
        "messages": [_needle_msg(_NEEDLE)],
    },
    "flood": {
        "description": "Send several distractor messages to drive budget pressure.",
        "messages": _flood(6),
    },
    "recall": {
        "description": "Probe for the early seeded fact.",
        "messages": [_recall_msg(_NEEDLE)],
    },
    "full_demo": {
        "description": (
            "End-to-end: seed an early fact, flood with distractors, seed a mid "
            "fact, flood again, then probe recall of the early fact."
        ),
        "messages": (
            [_needle_msg(_NEEDLE)]
            + _flood(4)
            + [_needle_msg(_MID_NEEDLE)]
            + _flood(4)
            + [_recall_msg(_NEEDLE)]
        ),
    },
}


def list_scripts() -> list[dict[str, str]]:
    return [
        {"name": name, "description": spec["description"]}
        for name, spec in SCRIPTS.items()
    ]


def script_messages(name: str) -> list[str] | None:
    spec = SCRIPTS.get(name)
    return list(spec["messages"]) if spec else None
