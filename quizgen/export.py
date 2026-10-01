"""Export quizzes and flashcards: JSON, CSV, Markdown, printable PDF and Anki decks."""
from __future__ import annotations

import csv
import io
import json
import zlib
from typing import List

from .models import Flashcard, Question

_LETTERS = "ABCDEFGH"


def to_json(questions: List[Question], cards: List[Flashcard]) -> str:
    return json.dumps({"questions": [q.to_dict() for q in questions],
                       "flashcards": [c.to_dict() for c in cards]}, indent=2, ensure_ascii=False)


def questions_csv(questions: List[Question]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["type", "question", "options", "answer", "explanation", "page", "difficulty", "bloom", "engine"])
    for q in questions:
        w.writerow([q.qtype, q.question, " | ".join(q.options), q.answer, q.explanation, q.page,
                    q.difficulty, q.bloom, q.engine])
    return buf.getvalue()


def flashcards_csv(cards: List[Flashcard]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["front", "back", "page"])
    for c in cards:
        w.writerow([c.front, c.back, c.page])
    return buf.getvalue()


def to_markdown(questions: List[Question], cards: List[Flashcard], title: str = "Quiz") -> str:
    lines = [f"# {title}", "", "## Questions", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"**Q{i}.** ({q.qtype}, {q.difficulty}) {q.question}")
        for j, o in enumerate(q.options):
            lines.append(f"   - {_LETTERS[j]}. {o}")
        lines.append("")
    lines += ["## Answer key", ""]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. **{q.answer}** — {q.explanation}")
    lines += ["", "## Flashcards", "", "| Front | Back | Page |", "|---|---|---|"]
    for c in cards:
        lines.append(f"| {c.front} | {c.back.replace('|', '/')} | {c.page or ''} |")
    return "\n".join(lines) + "\n"


def to_pdf(questions: List[Question], cards: List[Flashcard], title: str = "Quiz") -> bytes:
    """Printable worksheet: questions, then answer key, then flashcards."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title)
    st = getSampleStyleSheet()
    story = [Paragraph(escape(title), st["Title"]), Spacer(1, 12)]
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Q{i}.</b> {escape(q.question)}", st["BodyText"]))
        for j, o in enumerate(q.options):
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;{_LETTERS[j]}. {escape(o)}", st["BodyText"]))
        story.append(Spacer(1, 8))
    story += [PageBreak(), Paragraph("Answer key", st["Heading2"])]
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"{i}. <b>{escape(q.answer)}</b> (p. {q.page})", st["BodyText"]))
    if cards:
        story += [PageBreak(), Paragraph("Flashcards", st["Heading2"])]
        for c in cards:
            story.append(Paragraph(f"<b>{escape(c.front)}</b><br/>{escape(c.back)}", st["BodyText"]))
            story.append(Spacer(1, 6))
    doc.build(story)
    return buf.getvalue()


def to_anki(cards: List[Flashcard], deck_name: str = "Textbook Flashcards") -> bytes:
    """Build an Anki .apkg deck (requires `genanki`)."""
    import genanki
    import os
    import tempfile

    deck_id = zlib.crc32(deck_name.encode()) | 1 << 30
    model = genanki.Model(
        1607392319, "QuizGen Basic",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Page"}],
        templates=[{"name": "Card 1", "qfmt": "{{Front}}",
                    "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><small>p. {{Page}}</small>"}],
    )
    deck = genanki.Deck(deck_id, deck_name)
    for c in cards:
        deck.add_note(genanki.Note(model=model, fields=[c.front, c.back, str(c.page or "")]))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deck.apkg")
        genanki.Package(deck).write_to_file(path)
        with open(path, "rb") as f:
            return f.read()
