from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

from llm.interface import LLMBackend

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


_RETRIEVAL_NOVELTY_PROMPT = """\
You are assessing how novel new information is relative to an agent's existing memory.

These are the NEAREST existing memories (most semantically similar to the new content),
with their similarity scores — they are the most likely sources of redundancy or conflict:
{neighbors}

BROADER CONVERSATION CONTEXT:
{memory_summary}

NEW CONTENT:
"{content}"

Rate the novelty of the new content from 0.0 to 1.0:
- 1.0 = entirely new, surprising, or it CONTRADICTS / UPDATES / significantly extends one of
  the nearest memories above (an update or correction is highly novel even if it shares wording)
- 0.5 = partially new — adds some detail to what the nearest memories already cover
- 0.0 = fully redundant — already well-covered by the nearest memories

Judge primarily against the nearest memories. A near-duplicate is low novelty; a fact that
revises or conflicts with a near memory is high novelty.

Respond with a single float only.\
"""


class NoveltyMode(str, Enum):
    EMBEDDING = "embedding"        # fast, no extra API call
    LLM = "llm"                    # one call per block, whole-memory prompt
    HYBRID = "hybrid"              # embedding first, LLM only in ambiguous band
    RETRIEVAL_LLM = "retrieval"    # retrieve top-k neighbors, LLM grades against them


def _parse_float(text: str, fallback: float = 0.5) -> float:
    text = text.strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        match = re.search(r"(0\.\d+|1\.0*|[01])", text)
        if match:
            return max(0.0, min(1.0, float(match.group())))
        return fallback


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
    return min(1.0, max(0.0, 1.0 - mean_sim))


def _top_k_neighbors(
    embedding: list[float],
    cm: ContextManager,
    top_k: int,
) -> list[tuple[float, str]]:
    """Return up to top_k (similarity, content) pairs nearest to `embedding`,
    drawn from both the context store and the long-term store, highest sim first."""
    if not embedding:
        return []
    q = np.array(embedding, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0:
        return []

    scored: list[tuple[float, str]] = []
    for block in cm._store.all_blocks():
        if not block.original_embedding:
            continue
        v = np.array(block.original_embedding, dtype=np.float32)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0:
            continue
        scored.append((float(np.dot(q, v) / (q_norm * v_norm)), block.content))
    for lt_block, sim in cm._lt.similarity_search(embedding, top_k):
        scored.append((sim, lt_block.content))

    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[:top_k]


def score_novelty_retrieval_llm(
    content: str,
    embedding: list[float],
    cm: ContextManager,
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
    top_k: int = 5,
) -> float:
    """Retrieval-grounded LLM novelty scoring."""
    neighbors = _top_k_neighbors(embedding, cm, top_k)
    if neighbors:
        neighbor_text = "\n".join(
            f"  [sim {sim:.2f}] {text}" for sim, text in neighbors
        )
    else:
        neighbor_text = "  (memory is empty — nothing to compare against)"
    try:
        response = backend.complete(
            model=model,
            max_tokens=16,
            messages=[{
                "role": "user",
                "content": _RETRIEVAL_NOVELTY_PROMPT.format(
                    neighbors=neighbor_text,
                    memory_summary=cm.build_memory_prompt(),
                    content=content,
                ),
            }],
        )
        return _parse_float(response.text)
    except Exception:
        return score_novelty_embedding(embedding, cm, top_k)


def score_novelty_llm(
    content: str,
    memory_summary: str,
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
) -> float:
    """Score novelty via LLM call."""
    try:
        response = backend.complete(
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
        return _parse_float(response.text)
    except Exception:
        return 0.5


def score_novelty_hybrid(
    content: str,
    embedding: list[float],
    cm: ContextManager,
    backend: LLMBackend,
    model: str = "claude-haiku-4-5-20251001",
    ambiguity_range: tuple[float, float] = (0.35, 0.65),
) -> float:
    """Score novelty using embedding similarity, with LLM fallback for ambiguous cases."""
    emb_score = score_novelty_embedding(embedding, cm)
    lo, hi = ambiguity_range
    if lo <= emb_score <= hi:
        return score_novelty_llm(content, cm.build_memory_prompt(), backend, model)
    return emb_score


def build_novelty_fn(
    mode: NoveltyMode,
    cm: ContextManager,
    backend: LLMBackend,
    novelty_model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, list[float]], float]:
    """Factory: returns a novelty_fn closure for the given mode."""
    cfg = cm.config
    if mode == NoveltyMode.EMBEDDING:
        return lambda content, embedding: score_novelty_embedding(
            embedding, cm, cfg.novelty_top_k
        )
    if mode == NoveltyMode.LLM:
        return lambda content, embedding: score_novelty_llm(
            content, cm.build_memory_prompt(), backend, novelty_model
        )
    if mode == NoveltyMode.RETRIEVAL_LLM:
        return lambda content, embedding: score_novelty_retrieval_llm(
            content, embedding, cm, backend, novelty_model, cfg.novelty_retrieval_top_k
        )
    return lambda content, embedding: score_novelty_hybrid(
        content, embedding, cm, backend, novelty_model, cfg.novelty_hybrid_band
    )
