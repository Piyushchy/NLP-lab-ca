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


_EXAMPLE = {
    "questions": [
        {"qtype": "mcq", "question": "Which pigment absorbs light energy in plants?",
         "options": ["Chlorophyll", "Haemoglobin", "Melanin", "Keratin"], "answer": "Chlorophyll",
         "explanation": "The passage says chlorophyll absorbs light energy.",
         "source_quote": "Chlorophyll is a green pigment that absorbs light energy.",
         "difficulty": "easy", "bloom": "remember"},
    ],
    "flashcards": [{"front": "Chlorophyll", "back": "A green pigment that absorbs light energy.",
                    "source_quote": "Chlorophyll is a green pigment that absorbs light energy."}],
}


def _build_prompt(passage: str, n_q: int, n_cards: int, types: List[str], avoid: List[str]) -> str:
    type_names = {
        "mcq": '"mcq" (exactly 4 options; "answer" is the full text of the correct option, not a letter)',
        "true_false": '"true_false" (options ["True", "False"]; "answer" is "True" or "False")',
        "fill_blank": '"fill_blank" (put _____ in the question where the answer goes; options [])',
        "short_answer": '"short_answer" (options [])',
    }
    wanted = "; ".join(type_names[t] for t in types)
    avoid_text = ""
    if avoid:
        avoid_text = "Do not repeat or rephrase these earlier questions:\n" + "\n".join(f"- {q}" for q in avoid[:15]) + "\n\n"
    return (
        f"<passage>\n{passage}\n</passage>\n\n"
        f"Write {n_q} questions. Allowed values for \"qtype\": {wanted}. Mix the types.\n"
        f"Also write {n_cards} flashcards (front: a term or a question; back: a concise answer).\n"
        "Every item needs \"source_quote\": one sentence copied word for word from the passage.\n"
        "Skip anything the passage does not support rather than inventing facts.\n\n"
        f"{avoid_text}"
        "Reply with JSON only, in exactly this shape (this example is about a different passage):\n"
        f"{json.dumps(_EXAMPLE)}"
    )


def parse_json(text: str):
    """Parse the model reply, tolerating ```json fences or text around the JSON."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_, close in (("{", "}"), ("[", "]")):
            start, end = text.find(open_), text.rfind(close)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        raise


# ---------- tolerant normalisation of what open models actually return ----------

_ALIASES = {
    "qtype": ("qtype", "type", "question_type", "questionType", "kind", "format", "category"),
    "question": ("question", "question_text", "questionText", "stem", "prompt", "statement", "text", "q"),
    "options": ("options", "choices", "alternatives", "answers", "option"),
    "answer": ("answer", "correct_answer", "correctAnswer", "correct_option", "correctOption", "correct",
               "solution", "answer_text", "a"),
    "explanation": ("explanation", "rationale", "reason", "reasoning", "justification"),
    "source_quote": ("source_quote", "sourceQuote", "source", "quote", "evidence", "source_sentence",
                     "supporting_sentence", "reference", "context"),
    "front": ("front", "term", "question", "prompt", "concept", "q"),
    "back": ("back", "definition", "answer", "explanation", "a"),
}

_TYPE_SYNONYMS = {
    "mcq": "mcq", "multiple_choice": "mcq", "multiplechoice": "mcq", "multiple_choice_question": "mcq",
    "mc": "mcq", "choice": "mcq", "multiple": "mcq",
    "true_false": "true_false", "truefalse": "true_false", "true_or_false": "true_false", "tf": "true_false",
    "boolean": "true_false", "t_f": "true_false",
    "fill_blank": "fill_blank", "fill_in_the_blank": "fill_blank", "fill_in_blank": "fill_blank",
    "fill_in_the_blanks": "fill_blank", "fill_blanks": "fill_blank", "cloze": "fill_blank", "fill": "fill_blank",
    "blank": "fill_blank", "fitb": "fill_blank",
    "short_answer": "short_answer", "short": "short_answer", "open": "short_answer", "open_ended": "short_answer",
    "descriptive": "short_answer", "short_answer_question": "short_answer", "saq": "short_answer",
}

_LETTER_PREFIX = re.compile(r"^\s*(?:option\s*)?\(?([A-Ha-h])[\).:\-]\s+")


def _get(d: dict, field: str):
    for k in _ALIASES[field]:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _find_list(data, key: str, probe: tuple) -> list:
    """Locate the list of question / flashcard dicts wherever the model put it."""
    if isinstance(data, list):
        return data if key == "questions" else []
    if not isinstance(data, dict):
        return []
    for k, v in data.items():
        if key in k.lower() and isinstance(v, list):
            return v
    for v in data.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(p in v[0] for p in probe):
            return v
        if isinstance(v, dict):
            found = _find_list(v, key, probe)
            if found:
                return found
    return []


def _normalise_type(raw, question: str, options: list, answer: str) -> str:
    key = re.sub(r"[^a-z]+", "_", str(raw or "").lower()).strip("_")
    if key in _TYPE_SYNONYMS:
        return _TYPE_SYNONYMS[key]
    for k, v in _TYPE_SYNONYMS.items():
        if k in key and len(k) > 3:
            return v
    # no usable type: infer it from the shape of the item
    if answer.lower() in ("true", "false"):
        return "true_false"
    if len(options) >= 3:
        return "mcq"
    if "___" in question:
        return "fill_blank"
    return "short_answer"


def _resolve_answer(answer: str, options: List[str]) -> Optional[str]:
    """Map 'B', 'Option B', 'B) Rubisco', 'rubisco' or an index to the matching option text."""
    a = answer.strip()
    letter = re.fullmatch(r"(?:option\s*)?\(?([A-Ha-h])\)?[.:]?", a, flags=re.I)
    if letter:
        i = ord(letter.group(1).upper()) - ord("A")
        return options[i] if i < len(options) else None
    a = _LETTER_PREFIX.sub("", a).strip()
    for o in options:
        if o.lower() == a.lower():
            return o
    close = [o for o in options if a and (a.lower() in o.lower() or o.lower() in a.lower())]
    return close[0] if len(close) == 1 else None


def _clean_question(q: dict, types: List[str]) -> Optional[dict]:
    """Normalise one question from the model; return None to drop it."""
    if not isinstance(q, dict):
        return None
    question = str(_get(q, "question") or "").strip()
    raw_answer = _get(q, "answer")
    if isinstance(raw_answer, bool):
        raw_answer = "True" if raw_answer else "False"
    if isinstance(raw_answer, list):
        raw_answer = raw_answer[0] if raw_answer else ""
    answer = str(raw_answer or "").strip()
    raw_opts = _get(q, "options") or []
    if isinstance(raw_opts, dict):
        raw_opts = [raw_opts[k] for k in sorted(raw_opts)]
    if not isinstance(raw_opts, list):
        raw_opts = []
    options = [_LETTER_PREFIX.sub("", str(o)).strip() for o in raw_opts if str(o).strip()]
    if not question or not answer:
        return None
    qtype = _normalise_type(_get(q, "qtype"), question, options, answer)
    if qtype not in types:
        return None
    question = re.sub(r"_{3,}", "_____", question)
    if qtype == "true_false":
        if answer.lower() not in ("true", "false", "t", "f"):
            return None
        answer, options = ("True" if answer.lower().startswith("t") else "False"), ["True", "False"]
    elif qtype == "mcq":
        options = list(dict.fromkeys(options))  # drop duplicate options
        resolved = _resolve_answer(answer, options)
        if resolved is None and len(options) == 3 and answer:
            options, resolved = options + [_LETTER_PREFIX.sub("", answer).strip()], _LETTER_PREFIX.sub("", answer).strip()
        if resolved is None or not (3 <= len(options) <= 6):
            return None
        answer = resolved
    else:
        options = []
        answer = _LETTER_PREFIX.sub("", answer).strip()
    difficulty = str(q.get("difficulty", "medium")).lower()
    bloom = str(q.get("bloom", q.get("bloom_level", "understand"))).lower()
    return {
        "qtype": qtype, "question": question, "answer": answer, "options": options,
        "explanation": str(_get(q, "explanation") or "").strip(),
        "source_quote": str(_get(q, "source_quote") or "").strip(),
        "difficulty": difficulty if difficulty in DIFFICULTIES else "medium",
        "bloom": bloom if bloom in BLOOM_LEVELS else "understand",
    }


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _ground(quote: str, question: str, answer: str, qtype: str, chunk: List[Sentence],
            document: Optional[List[Sentence]] = None) -> Tuple[str, int]:
    """Return the passage sentence that supports an item, and its page.

    Uses the model's quote when it matches a passage sentence; otherwise finds the sentence
    that best supports the question and answer. If nothing matches, the raw quote is kept and
    the quality filter later drops the item as ungrounded.
    """
    best, best_score = None, 0.0
    qw = _words(quote)
    if qw:
        for s in list(chunk) + list(document or []):  # the passage first, then the whole document
            sw = _words(s.text)
            score = len(qw & sw) / len(qw)
            if score > best_score:
                best, best_score = s, score
        if best is not None and best_score >= 0.8:
            return best.text, best.page
    target = _words(question) | (_words(answer) if qtype != "true_false" else set())
    target -= {"the", "a", "an", "of", "is", "are", "in", "to", "and", "what", "which", "true", "false", "or"}
    aw = _words(answer) - {"the", "a", "an", "of"}
    best, best_score = None, 0.0
    for s in chunk:
        sw = _words(s.text)
        if qtype not in ("true_false",) and aw and len(aw & sw) / len(aw) < 0.6:
            continue  # the sentence must contain the answer
        score = len(target & sw) / max(len(target), 1)
        if score > best_score:
            best, best_score = s, score
    if best is not None and best_score >= 0.4:
        return best.text, best.page
    return quote, chunk[0].page


def generate_from_chunk(chunk: List[Sentence], n_q: int, n_cards: int, types: List[str],
                        api_key: Optional[str] = None, model: Optional[str] = None, client=None,
                        avoid: Optional[List[str]] = None, seed: Optional[int] = None,
                        raw_replies: Optional[list] = None,
                        document: Optional[List[Sentence]] = None) -> Tuple[List[Question], List[Flashcard]]:
    client = client or make_client(api_key)
    passage = " ".join(s.text for s in chunk)
    options = {"temperature": 0.7}
    if seed is not None:
        options["seed"] = seed
    response = client.chat(
        model=model or default_model(api_key),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(passage, n_q, n_cards, types, avoid or [])},
        ],
        format=OUTPUT_SCHEMA,
        options=options,
    )
    content = response.message.content or ""
    if raw_replies is not None:
        raw_replies.append(content)
    data = parse_json(content)

    questions = []
    for raw in _find_list(data, "questions", ("question", "qtype", "type", "stem")):
        q = _clean_question(raw, types)
        if q:
            src, page = _ground(q["source_quote"], q["question"], q["answer"], q["qtype"], chunk, document)
            questions.append(Question(
                qtype=q["qtype"], question=q["question"], answer=q["answer"], options=q["options"],
                explanation=q["explanation"], source_sentence=src, page=page,
                difficulty=q["difficulty"], bloom=q["bloom"], engine="llm",
            ))
    cards = []
    for c in _find_list(data, "flashcards", ("front", "back", "term")):
        if not isinstance(c, dict):
            continue
        front, back = str(_get(c, "front") or "").strip(), str(_get(c, "back") or "").strip()
        if front and back and front != back:
            src, page = _ground(str(_get(c, "source_quote") or ""), front, back, "flashcard", chunk, document)
            cards.append(Flashcard(front=front, back=back, source_sentence=src, page=page, engine="llm"))
    return questions, cards
