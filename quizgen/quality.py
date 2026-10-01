"""Quality control: grounding / answerability checks, scoring and de-duplication."""
from __future__ import annotations

import re
from typing import List

from .models import Flashcard, Question
from .retrieval import TfidfRetriever

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


_TEMPLATE_WORDS = {"fill", "blank", "true", "false", "choose", "correct", "option", "complete", "statement",
                   "what", "meant", "by", "is", "the", "a", "an", "or", "of", "to", "in", "and", "are"}


def content_jaccard(a: str, b: str) -> float:
    """Jaccard similarity ignoring question-template and function words (for de-duplication)."""
    ta, tb = tokens(a) - _TEMPLATE_WORDS, tokens(b) - _TEMPLATE_WORDS
    return len(ta & tb) / max(len(ta | tb), 1)


def is_grounded(quote: str, source_text: str, threshold: float = 0.8) -> bool:
    """The supporting quote must (almost) appear in the source document."""
    if not quote:
        return False
    qt = tokens(quote)
    if not qt:
        return False
    src = source_text.lower()
    if quote.lower().strip(" .") in src:
        return True
    return len(qt & tokens(source_text)) / len(qt) >= threshold


def is_answerable(q: Question, retriever: TfidfRetriever, sentences: List[str]) -> bool:
    """Round-trip check: retrieve the best sentences for the question and verify the
    correct answer is supported by them (a lightweight extractive-QA stand-in)."""
    if q.qtype == "true_false":
        # The original (true) sentence must be retrievable from the statement.
        hits = retriever.search(q.question, k=3)
        return any(jaccard(sentences[i], q.source_sentence) > 0.8 for i, _ in hits)
    query = q.question.replace("_____", " ")
    hits = retriever.search(query, k=3)
    ans = tokens(q.answer)
    if not ans:
        return False
    for i, _ in hits:
        if len(ans & tokens(sentences[i])) / len(ans) >= 0.6:
            return True
    return False


def score_question(q: Question) -> float:
    s = 1.0
    n = len(q.question.split())
    if n < 6 or n > 70:
        s -= 0.3
    if q.qtype == "mcq":
        opts = [o.lower().strip() for o in q.options]
        if len(set(opts)) != len(opts) or q.answer.lower().strip() not in opts:
            return 0.0
        if any(o and o in q.question.lower().replace("_____", "") for o in opts if o != q.answer.lower()):
            s -= 0.2  # a distractor that appears in the stem is a give-away
    if q.qtype in ("fill_blank", "mcq") and q.answer.lower() in q.question.lower().replace("_____", ""):
        s -= 0.5  # answer leaks into the stem
    if q.qtype == "short_answer" and len(q.answer.split()) > 40:
        s -= 0.2
    return max(s, 0.0)


def filter_questions(questions: List[Question], source_text: str, retriever: TfidfRetriever,
                     sentences: List[str], dedup_threshold: float = 0.6) -> List[Question]:
    kept: List[Question] = []
    for q in questions:
        q.score = score_question(q)
        if q.score < 0.5:
            continue
        if q.engine == "llm" and not is_grounded(q.source_sentence, source_text):
            continue
        if not is_answerable(q, retriever, sentences):
            q.score -= 0.2
            if q.score < 0.5:
                continue
        if any(content_jaccard(q.question, k.question) > dedup_threshold or
               (q.source_sentence and q.source_sentence == k.source_sentence) for k in kept):
            continue
        kept.append(q)
    return kept


def filter_flashcards(cards: List[Flashcard], source_text: str) -> List[Flashcard]:
    kept: List[Flashcard] = []
    for c in cards:
        if c.engine == "llm" and not is_grounded(c.source_sentence, source_text):
            continue
        if any(jaccard(c.front, k.front) > 0.8 for k in kept):
            continue
        kept.append(c)
    return kept
