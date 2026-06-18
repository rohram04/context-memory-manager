"""Cheap LLM judge for grading predicted vs. gold answers on LoCoMo probes.

Used only on the final probes of the LoCoMo eval — not in the hot loop. The
judge handles abstention/adversarial items: when the gold answer indicates the
fact is not in the conversation (or is empty), a prediction that abstains is
correct and a confident wrong fact is incorrect.

No retrieval, no scoring heuristics — one small Haiku call per probe, parsed to
a single CORRECT/INCORRECT token. Defaults to False (incorrect) on any error.
"""

from __future__ import annotations

from typing import Callable

import anthropic

from eval.cost import BudgetExceeded

_JUDGE_PROMPT = """\
You are grading whether a predicted answer to a question is correct, given the \
gold (reference) answer.

QUESTION:
{question}

GOLD ANSWER:
{gold}

PREDICTED ANSWER:
{predicted}

Grading rules:
- If the prediction conveys the same fact as the gold answer (allowing for \
paraphrase, extra detail, or different wording), it is CORRECT.
- If the prediction states a different or contradictory fact, it is INCORRECT.
- ABSTENTION CASE: if the gold answer indicates there is no answer (e.g. it is \
empty, or says "not mentioned", "no answer", "not in the conversation", or \
similar), then a prediction that abstains or says it does not know is CORRECT, \
and a prediction that confidently states a specific fact is INCORRECT.

Respond with exactly one word: CORRECT or INCORRECT."""


def _extract_text(content: list) -> str:
    for block in content:
        if hasattr(block, "text"):
            return block.text
    return ""


def make_judge_fn(
    client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5-20251001",
) -> Callable[[str, str, str], bool]:
    """Return a closure (question, gold, predicted) -> bool.

    True when the prediction is judged correct. Parses CORRECT/INCORRECT
    case-insensitively; returns False on any API or parse error.
    """
    def judge(question: str, gold_answer: str, predicted_answer: str) -> bool:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                    question=question,
                    gold=gold_answer,
                    predicted=predicted_answer,
                )}],
            )
            verdict = _extract_text(response.content).strip().upper()
            return verdict.startswith("CORRECT")
        except BudgetExceeded:
            raise  # let --max-spend aborts reach the harness; don't grade as wrong
        except Exception:
            return False

    return judge
