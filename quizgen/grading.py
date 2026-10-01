"""Checking a learner's answer against the key (lenient for typed answers)."""
from __future__ import annotations

import re

from .evaluate import token_f1
from .keywords import normalize
from .models import Question


def check_answer(q: Question, response: str | None) -> bool:
    if not response or not response.strip():
        return False
    if q.qtype in ("mcq", "true_false"):
        return response.strip().lower() == q.answer.strip().lower()
    given, gold = normalize(response), normalize(q.answer)
    if given == gold:
        return True
    # Short answers: accept paraphrases with high word overlap.
    threshold = 0.85 if q.qtype == "fill_blank" else 0.6
    return token_f1(re.sub(r"\b(the|a|an)\b", " ", given), re.sub(r"\b(the|a|an)\b", " ", gold)) >= threshold
