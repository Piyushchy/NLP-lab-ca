"""Educational Quiz & Flashcard Generator from PDF textbooks."""
from .models import Flashcard, Question
from .pipeline import QuizResult, generate

__all__ = ["generate", "QuizResult", "Question", "Flashcard"]
__version__ = "1.0.0"
