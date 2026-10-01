"""Naive cloze baseline used in the evaluation: random sentence, random noun blanked,
random nouns as distractors. No keyphrase ranking, no context-aware distractors, no filter."""
from __future__ import annotations

import random
from typing import List

from .classic import BLANK, _replace_once
from .models import Question, Sentence
from .nlp import pos_tag, word_tokenize


def naive_questions(sentences: List[Sentence], n: int, seed: int = 13) -> List[Question]:
    rng = random.Random(seed)
    nouns_by_sent = {
        s.index: [w for w, t in pos_tag(word_tokenize(s.text)) if t.startswith("NN") and w.isalpha() and len(w) > 2]
        for s in sentences
    }
    vocab = sorted({w.lower() for ws in nouns_by_sent.values() for w in ws})
    pool = [s for s in sentences if nouns_by_sent[s.index]]
    rng.shuffle(pool)
    out: List[Question] = []
    for i, s in enumerate(pool[:n]):
        answer = rng.choice(nouns_by_sent[s.index])
        stem = _replace_once(s.text, answer, BLANK)
        if not stem:
            continue
        if i % 2 == 0:
            wrong = rng.sample([w for w in vocab if w != answer.lower()], 3)
            opts = wrong + [answer]
            rng.shuffle(opts)
            out.append(Question("mcq", f"Choose the correct option to complete the statement: {stem}", answer,
                                opts, source_sentence=s.text, page=s.page, keyword=answer, engine="baseline"))
        else:
            out.append(Question("fill_blank", f"Fill in the blank: {stem}", answer, source_sentence=s.text,
                                page=s.page, keyword=answer, engine="baseline"))
    return out
