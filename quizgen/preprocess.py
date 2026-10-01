"""Cleaning, sentence segmentation and chunking."""
from __future__ import annotations

import re
from collections import Counter
from typing import List

from .models import Page, Sentence
from .nlp import sent_tokenize, word_tokenize

_MIN_WORDS, _MAX_WORDS = 6, 60


def _repeated_lines(pages: List[Page]) -> set[str]:
    """Lines that appear on most pages are running headers/footers."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for p in pages:
        counts.update({ln.strip() for ln in p.text.splitlines() if ln.strip()})
    return {ln for ln, c in counts.items() if c >= 0.6 * len(pages)}


def _is_heading(line: str, prev: str) -> bool:
    """Short line with no closing punctuation that follows a finished sentence (or nothing)."""
    if re.match(r"^(chapter|section|unit)\s+\d+", line, re.I) or re.match(r"^\d+(\.\d+)+\s+[A-Z]", line):
        return True
    words = line.split()
    if len(words) > 8 or line[-1] in ".,;:!?-" or (prev and prev[-1] not in ".!?:"):
        return False
    # Headings are short or Title Case; this keeps short wrapped body lines.
    content = [w for w in words if len(w) > 3]
    return len(words) <= 2 or (bool(content) and sum(w[0].isupper() for w in content) / len(content) >= 0.6)


def clean_text(text: str, drop_lines: set[str] | None = None) -> str:
    lines = []
    prev = ""
    for ln in text.splitlines():
        s = ln.strip()
        if not s or (drop_lines and s in drop_lines):
            continue
        if re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", s, flags=re.I):  # page numbers
            continue
        if _is_heading(s, prev):
            prev = ""
            continue
        lines.append(s)
        prev = s
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[•●▪–]\s*", " ", text)  # bullets
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def is_good_sentence(s: str) -> bool:
    words = s.split()
    if not (_MIN_WORDS <= len(words) <= _MAX_WORDS):
        return False
    if not s[0].isupper() or s[-1] not in ".!":
        return False
    if s.endswith("?") or re.search(r"\b(figure|fig\.|table|exercise|see page)\b", s, re.I):
        return False
    alpha = sum(ch.isalpha() for ch in s)
    return alpha / max(len(s), 1) > 0.6


def split_sentences(pages: List[Page]) -> List[Sentence]:
    drop = _repeated_lines(pages)
    out: List[Sentence] = []
    for p in pages:
        for s in sent_tokenize(clean_text(p.text, drop)):
            s = s.strip()
            if is_good_sentence(s):
                out.append(Sentence(text=s, page=p.number, index=len(out)))
    return out


def chunk_sentences(sentences: List[Sentence], max_words: int = 180) -> List[List[Sentence]]:
    """Group consecutive sentences into ~max_words chunks for retrieval / LLM context."""
    chunks, cur, n = [], [], 0
    for s in sentences:
        w = len(word_tokenize(s.text))
        if cur and n + w > max_words:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(cur)
    return chunks
