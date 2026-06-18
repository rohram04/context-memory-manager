from __future__ import annotations

from dataclasses import dataclass, field


def _default_pressure_thresholds() -> dict[str, float]:
    return {"critical": 0.90, "high": 0.75, "medium": 0.50, "low": 0.0}


@dataclass
class MemoryConfig:
    """Central, tunable configuration for the memory manager.

    Every value here was previously a hardcoded constant scattered across
    store.py / ContextManager.py / controller.py. Collecting them into one object
    lets an evaluation sweep them without editing source. The defaults reproduce
    the original hardcoded behaviour exactly, so callers that pass no config are
    unaffected.
    """

    # --- compression priority (memory/store.py:_priority) -------------------
    # compressibility = w_novelty * (1 - novelty) + w_recency * (1 - decay)
    compress_novelty_weight: float = 0.6
    compress_recency_weight: float = 0.4

    # --- fidelity-triggered removal (ContextManager.compress) ---------------
    fidelity_removal_threshold: float = 0.50

    # --- pre-prompt promotion (controller.pre_prompt_promote) ---------------
    # score = alpha * similarity + beta * decay_score ; promote if >= threshold
    promote_alpha: float = 0.7
    promote_beta: float = 0.3
    promote_threshold: float = 0.6

    # --- augmentation / similarity (controller + store.find_similar) --------
    # Fallback cosine cutoff used only when LLM-judged augmentation is disabled.
    augment_similarity_threshold: float = 0.85
    # LLM-judged augmentation: surface the top-k most similar in-context blocks to a
    # cheap util-model that decides whether the incoming content is about the SAME
    # subject (augment into one) or a distinct topic (insert new). Robust where a
    # single cosine threshold is not: same-vs-different entity with similar wording.
    # Default ON; requires controller to be given an augment_decision_fn.
    llm_augment_decision: bool = True
    # Low cosine floor that forms the candidate set for the LLM judge (and lets us
    # skip the LLM call entirely when nothing is even plausibly related ~0.0).
    augment_candidate_floor: float = 0.4
    augment_candidate_top_k: int = 3

    # --- novelty scoring (memory/novelty.py) --------------------------------
    novelty_top_k: int = 5
    novelty_hybrid_band: tuple[float, float] = (0.35, 0.65)
    novelty_retrieval_top_k: int = 5

    # --- pre-prompt promotion fan-out (controller.receive) ------------------
    promote_top_k: int = 5

    # --- logical clock (decay signal) ---------------------------------------
    # 0.0 => wall-clock time (production default; decay measured in real seconds).
    # > 0  => simulated turn clock: each MemoryController.receive() advances time
    #         by this many seconds. Lets decay differentiate blocks during fast
    #         batch evals, where wall-clock dt is ~0 and every decay_score ~= 1.0.
    # Relative to CacheBlock.DECAY_CONSTANT_S (3600s), a value of 600 means decay
    # reaches ~e^-1 after 6 turns.
    clock_seconds_per_turn: float = 0.0

    # --- budget pressure thresholds (memory/store.py) -----------------------
    pressure_thresholds: dict[str, float] = field(
        default_factory=_default_pressure_thresholds
    )
