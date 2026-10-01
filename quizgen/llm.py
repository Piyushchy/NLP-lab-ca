"""Optional LLM engine: retrieval-grounded question/flashcard generation with Ollama.

Two ways to reach a model:

* Ollama Cloud API (no install, free tier): create a key at https://ollama.com/settings/keys
  and set OLLAMA_API_KEY. Requests go to https://ollama.com with a Bearer token and use
  cloud model names such as "gpt-oss:120b".
* A local Ollama install (OLLAMA_HOST, default http://localhost:11434): either a local model
  ("llama3.2") or a cloud model through the local app after `ollama signin`
  ("gpt-oss:120b-cloud").

Each request receives only one retrieved chunk of the textbook and must quote the supporting
sentence, so every item can be verified against the source. The output is constrained with
a JSON schema (`format=`) and validated again here, because open models do not always follow it.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from .models import Flashcard, Question, Sentence

CLOUD_HOST = "https://ollama.com"
DEFAULT_CLOUD_MODEL = "gpt-oss:120b"
DEFAULT_LOCAL_MODEL = "gpt-oss:120b-cloud"
# Cloud models listed in the Ollama Python library README; any other model name also works.
SUGGESTED_MODELS = ["gpt-oss:120b", "gpt-oss:20b", "deepseek-v3.1:671b", "qwen3-coder:480b", "kimi-k2:1t"]

SYSTEM_PROMPT = (
    "You are an experienced teacher who writes exam questions and revision flashcards from "
    "textbook passages. Use only facts stated in the passage you are given. For every item, copy "
    "the supporting sentence from the passage verbatim into `source_quote`; it is used to verify "
    "the item. For multiple-choice questions give exactly four options, one correct, with wrong "
    "options that are plausible to a student who skimmed the chapter; the answer must be copied "
    "exactly from the options. Vary Bloom's levels (remember, understand, apply, analyze) and "
    "difficulty. Reply with JSON only."
)

QTYPES = ["mcq", "true_false", "fill_blank", "short_answer"]
DIFFICULTIES = ["easy", "medium", "hard"]
BLOOM_LEVELS = ["remember", "understand", "apply", "analyze"]

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "qtype": {"type": "string", "enum": QTYPES},
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
        "explanation": {"type": "string"},
        "source_quote": {"type": "string"},
        "difficulty": {"type": "string", "enum": DIFFICULTIES},
        "bloom": {"type": "string", "enum": BLOOM_LEVELS},
    },
    "required": ["qtype", "question", "options", "answer", "explanation", "source_quote", "difficulty", "bloom"],
}

_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "front": {"type": "string"},
        "back": {"type": "string"},
        "source_quote": {"type": "string"},
    },
    "required": ["front", "back", "source_quote"],
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": _QUESTION_SCHEMA},
        "flashcards": {"type": "array", "items": _CARD_SCHEMA},
    },
    "required": ["questions", "flashcards"],
}


def _api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.environ.get("OLLAMA_API_KEY") or None


def _local_host() -> Optional[str]:
    return os.environ.get("OLLAMA_HOST") or None


def llm_available(api_key: Optional[str] = None) -> bool:
    """True when the `ollama` package is installed and a cloud key or a local host is configured."""
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return bool(_api_key(api_key) or _local_host())


def default_model(api_key: Optional[str] = None) -> str:
    if os.environ.get("QUIZGEN_MODEL"):
        return os.environ["QUIZGEN_MODEL"]
    return DEFAULT_CLOUD_MODEL if _api_key(api_key) else DEFAULT_LOCAL_MODEL


def make_client(api_key: Optional[str] = None):
    from ollama import Client

    key = _api_key(api_key)
    if key:  # Ollama Cloud API
        return Client(host=CLOUD_HOST, headers={"Authorization": "Bearer " + key}, timeout=180)
    return Client(host=_local_host(), timeout=180)  # local Ollama app


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
        "Skip anything the passage does not support rather than inventing facts. "
        'Return a JSON object: {"questions": [...], "flashcards": [...]}.'
    )


def parse_json(text: str) -> dict:
    """Parse the model reply, tolerating ```json fences or text around the object."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _clean_question(q: dict, types: List[str]) -> Optional[dict]:
    """Validate one question from the model; return None to drop it."""
    if not isinstance(q, dict):
        return None
    qtype = str(q.get("qtype", "")).strip().lower()
    question, answer = str(q.get("question", "")).strip(), str(q.get("answer", "")).strip()
    if qtype not in types or not question or not answer:
        return None
    options = [str(o).strip() for o in q.get("options") or [] if str(o).strip()]
    if qtype == "true_false":
        if answer.lower() not in ("true", "false"):
            return None
        answer, options = answer.capitalize(), ["True", "False"]
    elif qtype == "mcq":
        match = [o for o in options if o.lower() == answer.lower()]
        if len(options) != 4 or len({o.lower() for o in options}) != 4 or not match:
            return None
        answer = match[0]
    else:
        options = []
    difficulty = str(q.get("difficulty", "medium")).lower()
    bloom = str(q.get("bloom", "understand")).lower()
    return {
        "qtype": qtype, "question": question, "answer": answer, "options": options,
        "explanation": str(q.get("explanation", "")).strip(), "source_quote": str(q.get("source_quote", "")).strip(),
        "difficulty": difficulty if difficulty in DIFFICULTIES else "medium",
        "bloom": bloom if bloom in BLOOM_LEVELS else "understand",
    }


def generate_from_chunk(chunk: List[Sentence], n_q: int, n_cards: int, types: List[str],
                        api_key: Optional[str] = None, model: Optional[str] = None,
                        client=None) -> Tuple[List[Question], List[Flashcard]]:
    client = client or make_client(api_key)
    passage = " ".join(s.text for s in chunk)
    response = client.chat(
        model=model or default_model(api_key),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(passage, n_q, n_cards, types)},
        ],
        format=OUTPUT_SCHEMA,
        options={"temperature": 0.3},
    )
    data = parse_json(response.message.content or "")

    def page_of(quote: str) -> int:
        q = quote.lower()[:60]
        for s in chunk:
            if q and q in s.text.lower():
                return s.page
        return chunk[0].page

    questions = []
    for raw in data.get("questions") or []:
        q = _clean_question(raw, types)
        if q:
            questions.append(Question(
                qtype=q["qtype"], question=q["question"], answer=q["answer"], options=q["options"],
                explanation=q["explanation"], source_sentence=q["source_quote"], page=page_of(q["source_quote"]),
                difficulty=q["difficulty"], bloom=q["bloom"], engine="llm",
            ))
    cards = []
    for c in data.get("flashcards") or []:
        if isinstance(c, dict) and str(c.get("front", "")).strip() and str(c.get("back", "")).strip():
            quote = str(c.get("source_quote", "")).strip()
            cards.append(Flashcard(front=str(c["front"]).strip(), back=str(c["back"]).strip(),
                                   source_sentence=quote, page=page_of(quote), engine="llm"))
    return questions, cards
