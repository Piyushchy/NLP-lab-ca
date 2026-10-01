"""End-to-end pipeline: PDF -> text -> concepts -> questions & flashcards -> quality filter."""
from __future__ import annotations

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


def _select_chunks(chunks: List[List[Sentence]], keyphrases: List[dict], k: int) -> List[List[Sentence]]:
    """Pick the k chunks covering the most important concepts (keeps document order)."""
    weights = {}
    for kp in keyphrases:
        for idx in kp["sentences"]:
            weights[idx] = weights.get(idx, 0) + kp["score"]
    scored = [(sum(weights.get(s.index, 0) for s in ch), i) for i, ch in enumerate(chunks)]
    best = sorted(i for _, i in sorted(scored, reverse=True)[:k])
    return [chunks[i] for i in best]


def generate(
    source: PdfSource,
    n_questions: int = 10,
    n_flashcards: int = 10,
    types: Optional[List[str]] = None,
    engine: str = "auto",
    page_range: Optional[tuple] = None,
    seed: int = 13,
    use_filter: bool = True,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> QuizResult:
    """Generate a quiz and flashcards from a PDF.

    engine: "classic" (NLP only), "llm" (Ollama model, retrieval-grounded) or "auto"
    (LLM if a key is configured, otherwise classic). `use_filter=False` skips the
    quality filter (used for the ablation study). `api_key` overrides the
    OLLAMA_API_KEY environment variable for this call only; `model` picks the Ollama model.
    """
    t0 = time.time()
    types = [t for t in (types or list(QUESTION_TYPES)) if t in QUESTION_TYPES]
    warnings: List[str] = []

    pages = extract_pages(source, page_range)
    sentences = split_sentences(pages)
    if len(sentences) < 5:
        raise ValueError("Too little usable text was found in the PDF to generate questions.")
    keyphrases = extract_keyphrases(sentences, top_k=max(40, n_questions * 3))
    full_text = " ".join(s.text for s in sentences)
    sent_texts = [s.text for s in sentences]
    retriever = TfidfRetriever(sent_texts)

    if engine == "auto":
        engine = "llm" if llm.llm_available(api_key) else "classic"
    if engine == "llm" and not llm.llm_available(api_key):
        warnings.append("No Ollama API key (OLLAMA_API_KEY) or local OLLAMA_HOST found; "
                        "fell back to the classic NLP engine.")
        engine = "classic"

    questions: List[Question] = []
    cards: List[Flashcard] = []
    if engine == "llm":
        chunks = chunk_sentences(sentences)
        n_chunks = min(len(chunks), max(1, (n_questions + 2) // 3))
        selected = _select_chunks(chunks, keyphrases, n_chunks)
        per_q = -(-n_questions // len(selected)) + 1  # over-generate a little for the filter
        per_c = -(-n_flashcards // len(selected)) + 1
        try:
            client = llm.make_client(api_key)
            for ch in selected:
                q, c = llm.generate_from_chunk(ch, per_q, per_c, types, api_key=api_key, model=model, client=client)
                questions += q
                cards += c
        except Exception as e:  # network / auth / model / parse errors: degrade gracefully
            warnings.append(f"LLM call failed ({type(e).__name__}: {e}); using the classic engine.")
            engine = "classic"
            questions, cards = [], []
        if engine == "llm" and not questions:
            warnings.append("The LLM returned no usable questions; using the classic engine.")
            engine = "classic"
            cards = []

    if engine == "classic":
        gen = ClassicGenerator(sentences, keyphrases, seed=seed)
        questions = gen.generate_questions(int(n_questions * 1.6) + 2, types)
        cards = gen.generate_flashcards(n_flashcards + 3)

    raw_q, raw_c = len(questions), len(cards)
    if use_filter:
        questions = filter_questions(questions, full_text, retriever, sent_texts)
        cards = filter_flashcards(cards, full_text)
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
