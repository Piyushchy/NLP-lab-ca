"""Thin wrappers around NLTK with graceful regex fallbacks when corpora are missing.

Covers the classic NLP steps used by the pipeline: tokenization, sentence
segmentation, POS tagging, noun-phrase chunking, stop-word removal,
lemmatization and WordNet lookups.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Tuple

import nltk

_FALLBACK_STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can did do does doing down during each few for from further had has
    have having he her here hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when where which while who whom
    why will with you your yours yourself yourselves also may might must shall would could""".split()
)

_NAMING_VERBS = {"called", "known", "termed", "named", "defined", "referred"}

_NP_GRAMMAR = r"""
NP: {<JJ.*|VBN|VBG>*<NN.*>+(<IN><DT>?<JJ.*>*<NN.*>+)?}
"""


def _has(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


@lru_cache(maxsize=None)
def _resources() -> dict:
    return {
        "punkt": _has("tokenizers/punkt_tab") or _has("tokenizers/punkt"),
        "tagger": _has("taggers/averaged_perceptron_tagger_eng") or _has("taggers/averaged_perceptron_tagger"),
        "stopwords": _has("corpora/stopwords"),
        "wordnet": _has("corpora/wordnet") or _has("corpora/wordnet.zip"),
    }


def ensure_nltk_data(quiet: bool = True) -> None:
    """Download the NLTK resources used by the project (call once after install)."""
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger_eng", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=quiet)
    _resources.cache_clear()


def sent_tokenize(text: str) -> List[str]:
    if _resources()["punkt"]:
        return nltk.sent_tokenize(text)
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s]


def word_tokenize(text: str) -> List[str]:
    if _resources()["punkt"]:
        return nltk.word_tokenize(text)
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    if _resources()["stopwords"]:
        from nltk.corpus import stopwords as sw

        return frozenset(sw.words("english")) | frozenset(_FALLBACK_STOPWORDS)
    return frozenset(_FALLBACK_STOPWORDS)


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    if _resources()["tagger"]:
        return nltk.pos_tag(tokens)
    # Crude fallback: capitalised/long words as nouns, -ly adverbs, rest nouns unless stop-word.
    out = []
    for t in tokens:
        if not t[0].isalnum():
            out.append((t, "."))
        elif t.lower() in stopwords():
            out.append((t, "DT"))
        elif t.endswith("ly"):
            out.append((t, "RB"))
        elif t.isdigit():
            out.append((t, "CD"))
        else:
            out.append((t, "NN"))
    return out


_chunker = nltk.RegexpParser(_NP_GRAMMAR)


def _verb_dominant(word: str) -> bool:
    """True if WordNet sees `word` far more often as a verb than as a noun (e.g. "increases", "form")."""
    if not _resources()["wordnet"] or not word.isalpha():
        return False
    from nltk.corpus import wordnet as wn

    w = word.lower()
    verb = sum(l.count() for syn in wn.synsets(w, pos=wn.VERB) for l in syn.lemmas() if l.name() == wn.morphy(w, wn.VERB))
    noun = sum(l.count() for syn in wn.synsets(w, pos=wn.NOUN) for l in syn.lemmas() if l.name() == wn.morphy(w, wn.NOUN))
    return verb > 0 and verb >= 2 * noun


def _has_verb_reading(word: str) -> bool:
    if not _resources()["wordnet"]:
        return False
    from nltk.corpus import wordnet as wn

    return wn.morphy(word.lower(), wn.VERB) is not None


def noun_phrases(sentence: str) -> List[str]:
    """Extract noun phrases with a POS-pattern chunker (shallow parsing).

    Prepositional attachments are only kept for "of" ("rate of photosynthesis").
    A common tagger error is reading "Noun Verb" as one noun phrase ("Clouds form
    when ...", "photosynthesis increases until ..."); when such a phrase is not
    followed by a verb and its last word is most likely a verb, that word is dropped.
    """
    tagged = pos_tag(word_tokenize(sentence))
    if not tagged:
        return []
    tree = _chunker.parse(tagged)
    phrases = []
    pos = 0
    for node in tree:
        if not hasattr(node, "label"):
            pos += 1
            continue
        leaves = node.leaves()
        start, pos = pos, pos + len(leaves)
        words = [w for w, _ in leaves]
        tags = [t for _, t in leaves]
        if "IN" in tags:
            i = tags.index("IN")
            if words[i].lower() != "of":
                words, tags = words[:i], tags[:i]
        nxt = tagged[pos][1] if pos < len(tagged) else ""
        if len(words) > 1 and not nxt.startswith(("VB", "MD")):
            last = words[-1]
            if tags[-1] == "NNS" and _verb_dominant(last):
                words = words[:-1]  # "rate of photosynthesis increases until ..."
            elif start == 0 and nxt == "WRB" and _has_verb_reading(last):
                words = words[:-1]  # "Clouds form when ..."
        if len(words) > 1 and (words[0].lower() in _NAMING_VERBS or _verb_dominant(words[0])):
            words = words[1:]  # "called evapotranspiration", "transmits groundwater"
        phrase = " ".join(words).strip()
        if phrase and any(ch.isalpha() for ch in phrase):
            phrases.append(phrase)
    return phrases


@lru_cache(maxsize=4096)
def lemmatize(word: str) -> str:
    if _resources()["wordnet"]:
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer().lemmatize(word.lower())
    w = word.lower()
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("s", "")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)] + rep
    return w


def wordnet_available() -> bool:
    return _resources()["wordnet"]
