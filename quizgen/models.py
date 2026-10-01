"""Data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

QUESTION_TYPES = ("mcq", "true_false", "fill_blank", "short_answer")


@dataclass
class Page:
    number: int  # 1-indexed page number in the source PDF
    text: str


@dataclass
class Sentence:
    text: str
    page: int
    index: int  # position in the document


@dataclass
class Question:
    qtype: str  # one of QUESTION_TYPES
    question: str
    answer: str
    options: List[str] = field(default_factory=list)  # MCQ / True-False choices
    explanation: str = ""
    source_sentence: str = ""
    page: Optional[int] = None
    keyword: str = ""
    difficulty: str = "medium"  # easy / medium / hard
    bloom: str = "remember"  # Bloom's taxonomy level
    engine: str = "classic"  # classic / llm
    score: float = 0.0  # quality score from the filter

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Flashcard:
    front: str
    back: str
    page: Optional[int] = None
    source_sentence: str = ""
    engine: str = "classic"
    # SM-2 spaced-repetition state
    easiness: float = 2.5
    interval: int = 0
    repetitions: int = 0
    due: int = 0  # day index when the card is next due

    def to_dict(self) -> dict:
        return asdict(self)
