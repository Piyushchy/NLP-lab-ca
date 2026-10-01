"""Optional LLM engine: retrieval-grounded question/flashcard generation with Claude.

Enabled when an Anthropic credential is configured (e.g. ANTHROPIC_API_KEY).
Each request receives only one retrieved chunk of the textbook and must quote
the supporting sentence, so every item can be verified against the source.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

from .models import Flashcard, Question, Sentence

MODEL = os.environ.get("QUIZGEN_MODEL", "claude-opus-5-5")

SYSTEM_PROMPT = (
    "You are an experienced teacher who writes exam questions and revision flashcards from "
    "textbook passages. Use only facts stated in the passage you are given. For every item, copy "
    "the supporting sentence from the passage verbatim into `source_quote`; it is used to verify "
    "the item. For multiple-choice questions give exactly four options, one correct, with wrong "
    "options that are plausible to a student who skimmed the chapter. Vary Bloom's levels "
    "(remember, understand, apply, analyze) and difficulty."
)

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "qtype": {"type": "string", "enum": ["mcq", "true_false", "fill_blank", "short_answer"]},
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
        "source_quote": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "bloom": {"type": "string", "enum": ["remember", "understand", "apply", "analyze"]},
    },
    "required": ["qtype", "question", "options", "answer", "explanation", "source_quote", "difficulty", "bloom"],
    "additionalProperties": False,
}

_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "front": {"type": "string"},
        "back": {"type": "string"},
        "source_quote": {"type": "string"},
    },
    "required": ["front", "back", "source_quote"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": _QUESTION_SCHEMA},
        "flashcards": {"type": "array", "items": _CARD_SCHEMA},
    },
    "required": ["questions", "flashcards"],
    "additionalProperties": False,
}


def llm_available(api_key: str | None = None) -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _build_prompt(passage: str, n_q: int, n_cards: int, types: List[str]) -> str:
    type_names = {
        "mcq": "multiple choice (mcq)", "true_false": "true/false (options ['True','False'])",
        "fill_blank": "fill in the blank (use _____ in the question; options empty)",
        "short_answer": "short answer (options empty)",
    }
    wanted = ", ".join(type_names[t] for t in types)
    return (
        f"<passage>\n{passage}\n</passage>\n\n"
        f"Write {n_q} questions using these types: {wanted}. "
        f"Also write {n_cards} flashcards (front: a term or a question; back: a concise answer). "
        "Skip anything the passage does not support rather than inventing facts."
    )


def generate_from_chunk(chunk: List[Sentence], n_q: int, n_cards: int, types: List[str],
                        api_key: str | None = None) -> Tuple[List[Question], List[Flashcard]]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    passage = " ".join(s.text for s in chunk)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": _build_prompt(passage, n_q, n_cards, types)}],
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return [], []
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)

    def page_of(quote: str) -> int:
        q = quote.lower()[:60]
        for s in chunk:
            if q and q in s.text.lower():
                return s.page
        return chunk[0].page

    questions = [
        Question(
            qtype=q["qtype"], question=q["question"], answer=q["answer"], options=q["options"],
            explanation=q["explanation"], source_sentence=q["source_quote"], page=page_of(q["source_quote"]),
            difficulty=q["difficulty"], bloom=q["bloom"], engine="llm",
        )
        for q in data["questions"] if q["qtype"] in types
    ]
    cards = [
        Flashcard(front=c["front"], back=c["back"], source_sentence=c["source_quote"],
                  page=page_of(c["source_quote"]), engine="llm")
        for c in data["flashcards"]
    ]
    return questions, cards
