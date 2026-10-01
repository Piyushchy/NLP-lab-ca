"""SM-2 spaced-repetition scheduling for flashcards."""
from __future__ import annotations

from typing import List

from .models import Flashcard


def review(card: Flashcard, quality: int, today: int) -> Flashcard:
    """Update a card after a review. `quality` is 0 (blackout) .. 5 (perfect recall)."""
    quality = max(0, min(5, quality))
    if quality < 3:
        card.repetitions = 0
        card.interval = 1
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = round(card.interval * card.easiness)
        card.repetitions += 1
    card.easiness = max(1.3, card.easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    card.due = today + card.interval
    return card


def due_cards(cards: List[Flashcard], today: int) -> List[Flashcard]:
    return sorted((c for c in cards if c.due <= today), key=lambda c: (c.due, c.easiness))
