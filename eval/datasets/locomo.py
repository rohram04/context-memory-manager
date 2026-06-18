"""Loader for the LoCoMo benchmark (snap-research/locomo).

LoCoMo is 10 very-long multi-session conversations between two speakers, each
with ~200 hand-annotated QA pairs across five categories. This loader downloads
locomo10.json once, flattens each conversation to chronological turns, and
extracts QA pairs for use in the memory eval.

Confirmed schema (inspected from the real locomo10.json, June 2026):
  sample {
    sample_id: str                      # e.g. "conv-26"
    conversation: {
      speaker_a: str, speaker_b: str
      session_<N>: [ {speaker, dia_id, text}, ... ]   # N = chronological order
      session_<N>_date_time: str
    }
    qa: [ {question, answer, evidence, category}, ... ]
    observation, session_summary, event_summary       # unused here
  }

Schema surprises baked into the parsing below:
  - `category` is a STRING ("1".."5"), not an int.
  - Adversarial items (category 5) have NO `answer` field — the gold is under
    `adversarial_answer` instead. We normalize both into the "answer" key.
  - Sessions are dict keys (session_1, session_2, ...), not a list; order is by
    the integer suffix. Each has a sibling session_<N>_date_time string.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Confirmed integer->name mapping for the `category` field (stored as strings in
# the JSON). 5 is the adversarial/"no answer" category.
CATEGORY_NAMES: dict[int, str] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_SESSION_RE = re.compile(r"^session_(\d+)$")


@dataclass
class Conversation:
    conv_id: str
    turns: list[str] = field(default_factory=list)  # "<speaker>: <text>", chronological
    qa: list[dict] = field(default_factory=list)     # {question, answer, category}


def _download_if_needed(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, path)


def _flatten_turns(conversation: dict) -> list[str]:
    """Flatten all sessions into chronological "<speaker>: <text>" turns."""
    # Sessions live as keys session_1, session_2, ... — order by integer suffix.
    session_keys = sorted(
        (k for k in conversation if _SESSION_RE.match(k)),
        key=lambda k: int(_SESSION_RE.match(k).group(1)),
    )
    turns: list[str] = []
    for key in session_keys:
        session = conversation.get(key) or []
        if not isinstance(session, list):
            continue
        for turn in session:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            turns.append(f"{speaker}: {text}")
    return turns


def _normalize_qa(raw_qa: list[dict]) -> list[dict]:
    """Normalize QA items, folding adversarial_answer into the answer key."""
    out: list[dict] = []
    for item in raw_qa or []:
        # category is a string in the file; coerce to int defensively.
        raw_cat = item.get("category")
        try:
            category: int | str = int(raw_cat)
        except (TypeError, ValueError):
            category = raw_cat
        # Adversarial (cat 5) items store gold under adversarial_answer; be
        # defensive about either field being present.
        answer = item.get("answer")
        if answer is None:
            answer = item.get("adversarial_answer", "")
        out.append({
            "question": item.get("question", ""),
            "answer": "" if answer is None else str(answer),
            "category": category,
        })
    return out


def load_locomo(
    n_convs: int,
    turn_cap: int | None = None,
    cache_dir: str = "eval/data",
) -> list[Conversation]:
    """Load the first n_convs LoCoMo conversations as Conversation objects.

    Downloads locomo10.json into cache_dir once, then loads from disk. If
    turn_cap is set, each conversation's turns are truncated to the first
    turn_cap entries (QA pairs are left intact).
    """
    path = os.path.join(cache_dir, "locomo10.json")
    _download_if_needed(path)
    with open(path) as f:
        samples = json.load(f)

    conversations: list[Conversation] = []
    for sample in samples[:n_convs]:
        turns = _flatten_turns(sample.get("conversation", {}))
        if turn_cap is not None:
            turns = turns[:turn_cap]
        conversations.append(Conversation(
            conv_id=sample.get("sample_id", ""),
            turns=turns,
            qa=_normalize_qa(sample.get("qa", [])),
        ))
    return conversations


if __name__ == "__main__":
    convs = load_locomo(n_convs=1, turn_cap=10)
    c = convs[0]
    print(f"conv_id: {c.conv_id}")
    print(f"n_turns: {len(c.turns)}")
    print("first 3 turns:")
    for t in c.turns[:3]:
        print(f"  {t}")
    print(f"n_qa: {len(c.qa)}")
