from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Callable

import anthropic
import numpy as np

if TYPE_CHECKING:
    from ContextManager import ContextManager

_NOVELTY_PROMPT = """\
You are assessing how novel new information is relative to an agent's existing memory.

EXISTING MEMORY:
{memory_summary}

NEW CONTENT:
"{content}"

Rate the novelty of the new content from 0.0 to 1.0:
- 1.0 = entirely new, surprising, or contradicts / significantly extends what is known
- 0.5 = partially new — adds some detail to existing knowledge
- 0.0 = fully redundant — already well-covered in existing memory

Consider: does this contradict existing facts? Does it introduce new entities, events,
or preferences not yet captured? Is it a rephrasing of something already stored?

Respond with a single float only.\
"""


class NoveltyMode(str, Enum):
    EMBEDDING = "embedding"  # fast, no extra API call
    LLM = "llm"              # one Haiku call per block, highest quality
    HYBRID = "hybrid"        # embedding first, LLM only in ambiguous band


def _parse_float(text: str, fallback: float = 0.5) -> float:
    text = text.strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        match = re.search(r"(0\.\d+|1\.0*|[01])", text)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return fallback


def _extract_text(content: list) -> str:
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return ""


def score_novelty_embedding(
    embedding: list[float],
    cm: ContextManager,
    top_k: int = 5,
) -> float:
    """Embedding-based novelty: 1.0 - mean_top_k_cosine_similarity vs context + LT."""
    if not embedding:
        return 1.0
    q = np.array(embedding, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0:
        return 1.0

    sims: list[float] = []
    for block in cm._store.all_blocks():
        if not block.original_embedding:
            continue
        v = np.array(block.original_embedding, dtype=np.float32)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0:
            continue
        sims.append(float(np.dot(q, v) / (q_norm * v_norm)))
    for _, score in cm._lt.similarity_search(embedding, top_k):
        sims.append(score)

    if not sims:
        return 1.0
    sims.sort(reverse=True)
    mean_sim = sum(sims[:top_k]) / len(sims[:top_k])
    return max(0.0, 1.0 - mean_sim)


def score_novelty_llm(
    content: str,
    memory_summary: str,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> float:
    """Score novelty via LLM call.

    Captures contradiction, surprise, and significance that embedding similarity
    cannot detect. Uses a small model (Haiku by default) to keep latency low.
    Falls back to 0.5 on any API or parse failure.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{
                "role": "user",
                "content": _NOVELTY_PROMPT.format(
                    memory_summary=memory_summary,
                    content=content,
                ),
            }],
        )
        return _parse_float(_extract_text(response.content))
    except Exception:
        return 0.5


def score_novelty_hybrid(
    content: str,
    embedding: list[float],
    cm: ContextManager,
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
    ambiguity_range: tuple[float, float] = (0.35, 0.65),
) -> float:
    """Score novelty using embedding similarity, with LLM fallback for ambiguous cases.

    Most blocks are clearly novel or clearly redundant — the LLM is only called
    for the ambiguous middle band, keeping costs low while capturing edge cases
    embeddings miss (contradiction, significance, surprise).
    """
    emb_score = score_novelty_embedding(embedding, cm)
    lo, hi = ambiguity_range
    if lo <= emb_score <= hi:
        return score_novelty_llm(content, cm.build_memory_prompt(), client, model)
    return emb_score


def build_novelty_fn(
    mode: NoveltyMode,
    cm: ContextManager,
    client: anthropic.Anthropic,
    novelty_model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, list[float]], float]:
    """Factory: returns a novelty_fn closure for the given mode.

    The closure takes (content, embedding) — embedding is precomputed by the
    caller so it isn't re-derived inside novelty scoring and downstream receive().
    LLM mode ignores the embedding argument.
    """
    if mode == NoveltyMode.EMBEDDING:
        return lambda content, embedding: score_novelty_embedding(embedding, cm)
    if mode == NoveltyMode.LLM:
        return lambda content, embedding: score_novelty_llm(
            content, cm.build_memory_prompt(), client, novelty_model
        )
    # HYBRID
    return lambda content, embedding: score_novelty_hybrid(
        content, embedding, cm, client, novelty_model
    )
