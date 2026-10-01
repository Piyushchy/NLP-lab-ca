"""Rule-based / statistical question and flashcard generation (no LLM required)."""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

from .distractors import DistractorGenerator
from .models import Flashcard, Question, Sentence

BLANK = "_____"

# Definition patterns: (regex, which group is the term, which is the definition)
_DEF_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(?P<term>[A-Z][\w\s\-()']{2,60}?) (?:is|are) (?:defined as|known as) (?P<defn>.+)$"), "is"),
    (re.compile(r"^(?P<term>[A-Z][\w\s\-()']{2,60}?) refers? to (?P<defn>.+)$"), "refers"),
    (re.compile(r"^(?P<defn>.+?),? (?:is|are) (?:called|known as|termed) (?P<term>[\w\s\-']{2,50})$"), "called"),
    (re.compile(r"^(?P<term>[A-Z][\w\s\-()']{2,60}?) (?:is|are) (?P<defn>(?:a|an|the) .+)$"), "is"),
    (re.compile(r"^(?P<term>[A-Z][\w\s\-()']{2,60}?) means (?P<defn>.+)$"), "means"),
]

_PRONOUNS = {"it", "this", "that", "these", "those", "they", "there", "which", "such", "each"}

_ANAPHORA = re.compile(r"^(?:In |At |During |For )?(?:this|these|that|those|it|they|such|its|their)\b", re.I)

_NEGATIONS = [
    (r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bcan\b", "cannot"),
    (r"\bwas\b", "was not"), (r"\bwere\b", "were not"), (r"\bhas\b", "does not have"),
]


def _replace_once(text: str, phrase: str, repl: str) -> Optional[str]:
    pat = re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])", re.I)
    m = pat.search(text)
    if not m:
        return None
    if m.start() == 0 and repl[:1].islower():
        repl = repl[0].upper() + repl[1:]
    return text[: m.start()] + repl + text[m.end():]


def find_definition(sentence: str) -> Optional[Tuple[str, str]]:
    s = sentence.rstrip(".! ")
    for pat, _ in _DEF_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        term, defn = m.group("term").strip(), m.group("defn").strip()
        term = re.sub(r"^(?:The|A|An)\s+", "", term)
        if term.split()[0].lower() in _PRONOUNS or not (1 <= len(term.split()) <= 5) or len(defn.split()) < 3:
            continue
        if not defn[:2].isupper():  # keep acronyms like "NLP ..." as they are
            defn = defn[0].lower() + defn[1:]
        return term, defn
    return None


class ClassicGenerator:
    def __init__(self, sentences: List[Sentence], keyphrases: List[dict], seed: int = 13):
        self.sentences = sentences
        self.by_idx: Dict[int, Sentence] = {s.index: s for s in sentences}
        self.keyphrases = keyphrases
        self.rng = random.Random(seed)
        self.distractors = DistractorGenerator(keyphrases, sentences, seed=seed)
        self._rank = {kp["key"]: i for i, kp in enumerate(keyphrases)}

    # ---------- helpers ----------
    def _difficulty(self, kp: dict, sentence: str) -> str:
        rank = self._rank.get(kp["key"], len(self.keyphrases)) / max(len(self.keyphrases), 1)
        long = len(sentence.split()) > 24
        if rank < 0.12 and not long:
            return "easy"
        if rank > 0.35 or long:
            return "hard"
        return "medium"

    def _best_sentence(self, kp: dict, used: set) -> Optional[Sentence]:
        """Pick the most informative unused sentence containing the key phrase."""
        best, best_score = None, -1.0
        for idx in kp["sentences"]:
            s = self.by_idx.get(idx)
            if s is None or idx in used:
                continue
            if _replace_once(s.text, kp["phrase"], BLANK) is None:
                continue
            n = len(s.text.split())
            # prefer mid-length sentences and ones with other key concepts (more context)
            score = 1.0 - abs(n - 20) / 40
            score += 0.1 * sum(1 for o in self.keyphrases[:20] if o["key"] != kp["key"] and o["phrase"].lower() in s.text.lower())
            if s.text.lower().startswith(kp["phrase"].lower()):
                score -= 0.15  # blank at the very start reads poorly
            if _ANAPHORA.match(s.text):
                score -= 0.4  # "These reactions ..." needs the previous sentence for context
            if score > best_score:
                best, best_score = s, score
        return best

    # ---------- question types ----------
    def fill_blank(self, kp: dict, s: Sentence) -> Optional[Question]:
        stem = _replace_once(s.text, kp["phrase"], BLANK)
        if not stem:
            return None
        return Question(
            qtype="fill_blank", question=f"Fill in the blank: {stem}", answer=kp["phrase"],
            explanation=f'From the text (p. {s.page}): "{s.text}"', source_sentence=s.text,
            page=s.page, keyword=kp["phrase"], difficulty=self._difficulty(kp, s.text), bloom="remember",
        )

    def mcq(self, kp: dict, s: Sentence) -> Optional[Question]:
        stem = _replace_once(s.text, kp["phrase"], BLANK)
        if not stem:
            return None
        wrong = self.distractors.generate(kp["phrase"], context=s.text, k=3)
        if len(wrong) < 3:
            return None
        options = wrong + [kp["phrase"]]
        self.rng.shuffle(options)
        return Question(
            qtype="mcq", question=f"Choose the correct option to complete the statement: {stem}",
            answer=kp["phrase"], options=options,
            explanation=f'From the text (p. {s.page}): "{s.text}"', source_sentence=s.text, page=s.page,
            keyword=kp["phrase"], difficulty=self._difficulty(kp, s.text), bloom="understand",
        )

    def true_false(self, kp: dict, s: Sentence, make_false: bool) -> Optional[Question]:
        statement, how = s.text, "original"
        if make_false:
            wrong = self.distractors.generate(kp["phrase"], context=s.text, k=1)
            swapped = _replace_once(s.text, kp["phrase"], wrong[0]) if wrong else None
            if swapped:
                statement, how = swapped, f'"{kp["phrase"]}" was replaced by "{wrong[0]}"'
            else:
                for pat, rep in _NEGATIONS:
                    if re.search(pat, s.text):
                        statement, how = re.sub(pat, rep, s.text, count=1), "the statement was negated"
                        break
                else:
                    return None
        answer = "False" if make_false else "True"
        expl = f'The text (p. {s.page}) says: "{s.text}"'
        if make_false:
            expl += f" — {how}."
        return Question(
            qtype="true_false", question=f"True or False: {statement}", answer=answer, options=["True", "False"],
            explanation=expl, source_sentence=s.text, page=s.page, keyword=kp["phrase"],
            difficulty=self._difficulty(kp, s.text), bloom="understand",
        )

    def short_answer_from_definition(self, s: Sentence) -> Optional[Question]:
        d = find_definition(s.text)
        if not d:
            return None
        term, defn = d
        return Question(
            qtype="short_answer", question=f"What is meant by '{term}'?", answer=defn.rstrip("."),
            explanation=f'Definition from the text (p. {s.page}): "{s.text}"', source_sentence=s.text,
            page=s.page, keyword=term, difficulty="medium", bloom="remember",
        )

    # ---------- orchestration ----------
    def generate_questions(self, n: int, types: List[str]) -> List[Question]:
        """Round-robin over the requested types, walking concepts from most to least important."""
        if not types:
            return []
        out: List[Question] = []
        used_sentences: set = set()
        tf_toggle = self.rng.random() < 0.5
        type_cycle = list(types)
        t_i = 0

        # Definition-based short answers come from definition sentences, not keyphrases.
        defs = [s for s in self.sentences if find_definition(s.text)] if "short_answer" in types else []

        for kp in self.keyphrases:
            if len(out) >= n:
                break
            s = self._best_sentence(kp, used_sentences)
            if s is None:
                continue
            for attempt in range(len(type_cycle)):
                qtype = type_cycle[(t_i + attempt) % len(type_cycle)]
                q = None
                if qtype == "fill_blank":
                    q = self.fill_blank(kp, s)
                elif qtype == "mcq":
                    q = self.mcq(kp, s)
                elif qtype == "true_false":
                    q = self.true_false(kp, s, make_false=tf_toggle)
                    if q:
                        tf_toggle = not tf_toggle
                elif qtype == "short_answer" and defs:
                    d = defs.pop(0)
                    if d.index not in used_sentences:
                        q = self.short_answer_from_definition(d)
                        if q:
                            used_sentences.add(d.index)
                if q:
                    out.append(q)
                    used_sentences.add(s.index)
                    t_i = (t_i + attempt + 1) % len(type_cycle)
                    break
        # Top up with any remaining definitions if we are short.
        while len(out) < n and defs:
            d = defs.pop(0)
            if d.index in used_sentences:
                continue
            q = self.short_answer_from_definition(d)
            if q:
                out.append(q)
                used_sentences.add(d.index)
        return out

    def generate_flashcards(self, n: int) -> List[Flashcard]:
        cards: List[Flashcard] = []
        seen = set()
        # 1) explicit definitions -> term / definition cards
        for s in self.sentences:
            d = find_definition(s.text)
            if d and d[0].lower() not in seen:
                term, defn = d
                seen.add(term.lower())
                cards.append(Flashcard(front=term[0].upper() + term[1:], back=defn[0].upper() + defn[1:], page=s.page, source_sentence=s.text))
        # 2) top concepts -> concept / context-sentence cards
        used = set()
        for kp in self.keyphrases:
            if len(cards) >= n:
                break
            if kp["phrase"].lower() in seen:
                continue
            s = self._best_sentence(kp, used)
            if s is None:
                continue
            used.add(s.index)
            seen.add(kp["phrase"].lower())
            cards.append(Flashcard(front=f"Explain: {kp['phrase']}", back=s.text, page=s.page, source_sentence=s.text))
        return cards[:n]
