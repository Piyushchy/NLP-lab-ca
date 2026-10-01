"""Distractor (wrong option) generation for multiple-choice questions.

Two sources are combined:
1. Document-internal concepts that appear in similar contexts (TF-IDF cosine
   similarity of their context sentences) - plausible "hard" negatives.
2. WordNet co-hyponyms (sister terms) of the answer's head noun.
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .keywords import normalize
from .models import Sentence
from .nlp import lemmatize, wordnet_available


class DistractorGenerator:
    def __init__(self, keyphrases: List[dict], sentences: List[Sentence], seed: int = 13):
        self.keyphrases = keyphrases
        self.rng = random.Random(seed)
        by_idx = {s.index: s.text for s in sentences}
        contexts = [" ".join(by_idx[i] for i in kp["sentences"] if i in by_idx) or kp["phrase"] for kp in keyphrases]
        self.index: Dict[str, int] = {kp["key"]: i for i, kp in enumerate(keyphrases)}
        if keyphrases:
            vec = TfidfVectorizer(stop_words="english")
            try:
                self.sim = cosine_similarity(vec.fit_transform(contexts))
            except ValueError:  # empty vocabulary
                self.sim = np.zeros((len(keyphrases), len(keyphrases)))
        else:
            self.sim = np.zeros((0, 0))

    @staticmethod
    def _overlaps(a: str, b: str) -> bool:
        wa, wb = set(normalize(a).split()), set(normalize(b).split())
        return bool(wa & wb) or normalize(a) in normalize(b) or normalize(b) in normalize(a)

    @staticmethod
    def _is_plural(word: str) -> bool:
        return word.isalpha() and word.islower() and lemmatize(word) != word

    def _shape_ok(self, answer: str, cand: str) -> bool:
        # Keep options of a similar "shape" so the answer does not stand out.
        la, lc = len(answer.split()), len(cand.split())
        if abs(la - lc) > 1:
            return False
        if answer[:1].isupper() != cand[:1].isupper():
            return False
        if self._is_plural(answer.split()[-1]) != self._is_plural(cand.split()[-1]):
            return False  # keep subject-verb agreement when options are swapped in
        return any(ch.isdigit() for ch in answer) == any(ch.isdigit() for ch in cand)

    def _from_document(self, answer: str, context: str) -> List[str]:
        key = normalize(answer)
        ctx = normalize(context)
        if key in self.index:
            row = self.sim[self.index[key]]
            order = np.argsort(-row)
        else:
            order = np.arange(len(self.keyphrases))
        out = []
        for j in order:
            kp = self.keyphrases[j]
            cand = kp["phrase"]
            if kp["key"] == key or self._overlaps(answer, cand) or kp["key"] in ctx:
                continue
            if not self._shape_ok(answer, cand):
                continue
            out.append(cand)
        return out

    def _from_wordnet(self, answer: str) -> List[str]:
        if not wordnet_available():
            return []
        from nltk.corpus import wordnet as wn

        words = answer.split()
        if len(words) != 1:  # sister terms only make sense for single-word concepts
            return []
        out = []
        for syn in wn.synsets(words[0].lower(), pos=wn.NOUN)[:2]:
            for hyper in syn.hypernyms()[:2]:
                for sister in hyper.hyponyms():
                    if sister == syn or sister.lemmas()[0].count() == 0:
                        continue  # skip rare words: they make implausible distractors
                    name = sister.lemmas()[0].name().replace("_", " ")
                    if answer[:1].isupper():
                        name = name[:1].upper() + name[1:]
                    if not self._overlaps(name, answer):
                        out.append(name)
        return out

    def generate(self, answer: str, context: str = "", k: int = 3) -> List[str]:
        """Return up to k distractors for `answer` (excluding words used in `context`)."""
        doc = self._from_document(answer, context)
        wn_cands = self._from_wordnet(answer)
        picked: List[str] = []
        # Prefer the 2 most context-similar document concepts, then mix in WordNet terms.
        pools = [doc[:2], wn_cands[:4], doc[2:8]]
        for pool in pools:
            for c in pool:
                if len(picked) >= k:
                    break
                if all(not self._overlaps(c, p) for p in picked) and c.lower() != answer.lower():
                    picked.append(c)
        return picked[:k]
