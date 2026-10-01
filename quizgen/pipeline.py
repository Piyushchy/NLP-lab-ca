"""End-to-end pipeline: PDF -> text -> concepts -> questions & flashcards -> quality filter."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import llm
from .classic import ClassicGenerator
from .keywords import extract_keyphrases
from .models import QUESTION_TYPES, Flashcard, Page, Question, Sentence
from .pdf_extract import PdfSource, extract_pages
from .preprocess import chunk_sentences, split_sentences
from .quality import filter_flashcards, filter_questions
from .retrieval import TfidfRetriever


@dataclass
class QuizResult:
    questions: List[Question]
    flashcards: List[Flashcard]
    keyphrases: List[dict]
    sentences: List[Sentence]
    pages: List[Page]
    engine: str
    stats: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _select_chunks(chunks: List[List[Sentence]], keyphrases: List[dict], k: int, rng: random.Random,
                   avoid: set) -> List[List[Sentence]]:
    """Pick k chunks, weighted by how many important concepts they cover, so each run can use
    different passages. Chunks already used in earlier quizzes are down-weighted."""
    weights = {}
    for kp in keyphrases:
        for idx in kp["sentences"]:
            weights[idx] = weights.get(idx, 0) + kp["score"]

    def weight(ch):
        w = sum(weights.get(s.index, 0) for s in ch) + 1e-6
        seen = sum(s.text in avoid for s in ch) / len(ch)
        return w * (1 - 0.8 * seen)

    order = sorted(range(len(chunks)), key=lambda i: rng.random() ** (1.0 / weight(chunks[i])), reverse=True)
    return [chunks[i] for i in sorted(order[:k])]


def _snippet(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def generate(
    source: PdfSource,
    n_questions: int = 10,
    n_flashcards: int = 10,
    types: Optional[List[str]] = None,
    engine: str = "auto",
    page_range: Optional[tuple] = None,
    seed: Optional[int] = None,
    use_filter: bool = True,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    avoid: Optional[set] = None,
    avoid_questions: Optional[List[str]] = None,
) -> QuizResult:
    """Generate a quiz and flashcards from a PDF.

    engine: "classic" (NLP only), "llm" (Ollama model, retrieval-grounded) or "auto"
    (LLM if a key is configured, otherwise classic). `use_filter=False` skips the
    quality filter (used for the ablation study). `api_key` overrides the
    OLLAMA_API_KEY environment variable for this call only; `model` picks the Ollama model.

    `seed=None` gives a different quiz on every call (pass an int to reproduce one).
    `avoid` is a set of source sentences used in earlier quizzes and `avoid_questions`
    the earlier question texts; both steer generation towards new material.
    """
    t0 = time.time()
    types = [t for t in (types or list(QUESTION_TYPES)) if t in QUESTION_TYPES]
    warnings: List[str] = []
    rng = random.Random(seed)
    avoid = avoid or set()

    pages = extract_pages(source, page_range)
    sentences = split_sentences(pages)
    if len(sentences) < 5:
        raise ValueError("Too little usable text was found in the PDF to generate questions.")
    keyphrases = extract_keyphrases(sentences, top_k=max(40, n_questions * 3))
    full_text = " ".join(s.text for s in sentences)
    sent_texts = [s.text for s in sentences]
    retriever = TfidfRetriever(sent_texts)

    def classic(n_q: int, n_c: int):
        gen = ClassicGenerator(sentences, keyphrases, seed=rng.randrange(2**31), avoid=avoid)
        return gen.generate_questions(int(n_q * 1.6) + 2, types), gen.generate_flashcards(n_c + 3)

    if engine == "auto":
        engine = "llm" if llm.llm_available(api_key) else "classic"
    if engine == "llm" and not llm.llm_available(api_key):
        warnings.append("No Ollama API key (OLLAMA_API_KEY) or local OLLAMA_HOST found; "
                        "fell back to the classic NLP engine.")
        engine = "classic"

    questions: List[Question] = []
    cards: List[Flashcard] = []
    raw_replies: List[str] = []
    if engine == "llm":
        chunks = chunk_sentences(sentences)
        n_chunks = min(len(chunks), max(1, (n_questions + 2) // 3))
        selected = _select_chunks(chunks, keyphrases, n_chunks, rng, avoid)
        per_q = -(-n_questions // len(selected)) + 1  # over-generate a little for the filter
        per_c = -(-n_flashcards // len(selected)) + 1
        try:
            client = llm.make_client(api_key)
            for ch in selected:
                q, c = llm.generate_from_chunk(ch, per_q, per_c, types, api_key=api_key, model=model, client=client,
                                               avoid=avoid_questions, seed=rng.randrange(2**31),
                                               raw_replies=raw_replies, document=sentences)
                questions += q
                cards += c
        except Exception as e:  # network / auth / model / parse errors: degrade gracefully
            reply = f' Model reply started with: "{_snippet(raw_replies[-1])}"' if raw_replies else ""
            warnings.append(f"LLM call failed ({type(e).__name__}: {e}); using the classic engine.{reply}")
            engine = "classic"
            questions, cards = [], []
        if engine == "llm" and not questions:
            reply = f' Model reply started with: "{_snippet(raw_replies[0])}"' if raw_replies else ""
            warnings.append(f"The LLM returned no usable questions; using the classic engine.{reply}")
            engine = "classic"
            cards = []

    if engine == "classic":
        questions, cards = classic(n_questions, n_flashcards)

    raw_q, raw_c = len(questions), len(cards)
    if use_filter:
        questions = filter_questions(questions, full_text, retriever, sent_texts)
        cards = filter_flashcards(cards, full_text)
        if engine == "llm" and len(questions) < n_questions:
            # Top up with classic questions on sentences the LLM did not use.
            if not questions:
                warnings.append("None of the LLM's questions could be matched to a sentence in the PDF "
                                "(possible hallucinations), so the classic engine was used.")
                engine = "classic"
            else:
                warnings.append(f"The LLM produced {len(questions)} usable questions; the rest come from the "
                                "classic engine.")
            extra_q, extra_c = classic(n_questions - len(questions), max(0, n_flashcards - len(cards)))
            raw_q, raw_c = raw_q + len(extra_q), raw_c + len(extra_c)
            questions = filter_questions(questions + extra_q, full_text, retriever, sent_texts)
            cards = filter_flashcards(cards + extra_c, full_text)
    questions, cards = questions[:n_questions], cards[:n_flashcards]
    if len(questions) < n_questions:
        warnings.append(f"Only {len(questions)} good questions could be generated from this text.")

    stats = {
        "pages": len(pages), "sentences": len(sentences), "keyphrases": len(keyphrases),
        "raw_questions": raw_q, "kept_questions": len(questions),
        "raw_flashcards": raw_c, "kept_flashcards": len(cards),
        "seconds": round(time.time() - t0, 2),
    }
    return QuizResult(questions, cards, keyphrases, sentences, pages, engine, stats, warnings)
