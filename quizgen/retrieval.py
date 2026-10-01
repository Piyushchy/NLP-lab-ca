"""TF-IDF vector-space retriever over text chunks (the "R" in RAG)."""
from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform(docs) if docs else None

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        if self.mat is None:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.mat).ravel()
        order = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in order]
