"""Key-concept extraction: TF-IDF + TextRank over noun-phrase candidates."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Sentence
from .nlp import lemmatize, noun_phrases, stopwords

_GENERIC = {
    "example", "examples", "chapter", "section", "way", "ways", "type", "types", "kind", "number",
    "part", "parts", "use", "form", "thing", "things", "result", "results", "case", "cases", "lot",
    "fact", "set", "order", "time", "times", "term", "terms", "process", "amount", "level", "end",
    "one", "first", "second", "others", "many", "most", "same", "figure", "table", "page",
    "factor", "factors", "place", "region", "regions", "means", "step", "steps", "stage", "role",
}

_QUANTIFIERS = {"many", "most", "several", "few", "fewer", "tiny", "small", "large", "other", "another",
                "various", "different", "some", "each", "every", "certain", "main", "important", "new"}


def normalize(phrase: str) -> str:
    words = [lemmatize(w) for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", phrase)]
    return " ".join(words)


def _clean_candidate(phrase: str) -> str | None:
    words = phrase.split()
    sw = stopwords()
    while words and (words[0].lower() in sw or words[0].lower() in _QUANTIFIERS or not words[0][0].isalnum()):
        words = words[1:]
    while words and (words[-1].lower() in sw or not words[-1][0].isalnum()):
        words = words[:-1]
    if not words or len(words) > 4:
        return None
    lowered = [w.lower() for w in words]
    if any(w in sw and w != "of" for w in lowered[1:-1]):
        return None  # "runoff because fewer roots": spans a clause boundary
    phrase = " ".join(words)
    if len(phrase) < 3 or phrase.lower() in _GENERIC or not re.search(r"[A-Za-z]{3}", phrase):
        return None
    if all(w.lower() in sw or w.lower() in _GENERIC for w in words):
        return None
    return phrase


def candidate_phrases(sentences: List[Sentence]) -> Dict[str, dict]:
    """Map normalized phrase -> {surface form, sentence indices, count}."""
    cands: Dict[str, dict] = {}
    surfaces: Dict[str, Counter] = defaultdict(Counter)
    for s in sentences:
        for np_ in noun_phrases(s.text):
            c = _clean_candidate(np_)
            if not c:
                continue
            key = normalize(c)
            if not key:
                continue
            initial = s.text.startswith(c)
            surfaces[key][(c, initial)] += 1
            entry = cands.setdefault(key, {"sentences": set(), "count": 0})
            entry["sentences"].add(s.index)
            entry["count"] += 1
    for key, entry in cands.items():
        entry["surface"] = _surface_form(surfaces[key])
    return cands


def _surface_form(forms: Counter) -> str:
    """Prefer how the phrase is written mid-sentence; otherwise undo sentence-initial capitals
    (but keep acronyms / mixed-case words such as ATP or NADPH)."""
    mid = Counter({f: n for (f, initial), n in forms.items() if not initial})
    if mid:
        return mid.most_common(1)[0][0]
    form = forms.most_common(1)[0][0][0]
    first = form.split()[0]
    if first[1:] != first[1:].lower() or len(first) == 1:
        return form
    return form[0].lower() + form[1:]


def _textrank(sentences: List[Sentence], cands: Dict[str, dict]) -> Dict[str, float]:
    """PageRank over a co-occurrence graph: phrases sharing a sentence are linked."""
    g = nx.Graph()
    by_sentence: Dict[int, List[str]] = defaultdict(list)
    for key, e in cands.items():
        g.add_node(key)
        for idx in e["sentences"]:
            by_sentence[idx].append(key)
    for keys in by_sentence.values():
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                w = g.get_edge_data(a, b, {"weight": 0})["weight"]
                g.add_edge(a, b, weight=w + 1)
    if g.number_of_edges() == 0:
        return {k: 1.0 / max(len(cands), 1) for k in cands}
    return nx.pagerank(g, weight="weight")


def _tfidf(sentences: List[Sentence], cands: Dict[str, dict]) -> Dict[str, float]:
    """Max TF-IDF of each candidate across sentence-level documents."""
    docs = [normalize(s.text) for s in sentences]
    vocab = sorted(cands)
    if not docs or not vocab:
        return {}
    vec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 4), token_pattern=r"[A-Za-z0-9][A-Za-z0-9'-]*")
    mat = vec.fit_transform(docs)
    maxes = mat.max(axis=0).toarray().ravel()
    return dict(zip(vec.get_feature_names_out(), maxes))


def extract_keyphrases(sentences: List[Sentence], top_k: int = 40) -> List[dict]:
    """Return ranked key concepts: [{phrase, key, score, count, sentences}]."""
    cands = candidate_phrases(sentences)
    if not cands:
        return []
    tr = _textrank(sentences, cands)
    tf = _tfidf(sentences, cands)

    def norm(d: Dict[str, float]) -> Dict[str, float]:
        m = max(d.values(), default=0) or 1.0
        return {k: v / m for k, v in d.items()}

    tr, tf = norm(tr), norm(tf)
    ranked = []
    for key, e in cands.items():
        freq_bonus = min(e["count"], 5) / 5
        multiword = 0.15 if " " in key else 0.0
        score = 0.45 * tr.get(key, 0) + 0.35 * tf.get(key, 0) + 0.2 * freq_bonus + multiword
        ranked.append({
            "phrase": e["surface"], "key": key, "score": round(score, 4),
            "count": e["count"], "sentences": sorted(e["sentences"]),
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)

    # Drop phrases that are sub-strings of a higher-ranked phrase with the same sentences.
    out: List[dict] = []
    for r in ranked:
        if any(set(r["key"].split()) < set(o["key"].split()) and set(r["sentences"]) <= set(o["sentences"])
               for o in out):
            continue
        out.append(r)
        if len(out) >= top_k:
            break
    return out
