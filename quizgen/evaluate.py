"""Evaluation of generated quizzes.

* Reference-based: BLEU-4 and ROUGE-L between each generated question and its
  best-matching teacher-written question, plus answer-concept coverage (how many reference answers are
  tested by at least one generated question or flashcard).
* Reference-free (intrinsic): grounding, answerability, MCQ validity,
  diversity, type / difficulty / Bloom distribution.
* Human evaluation: a CSV rating sheet (relevance, fluency, answerability,
  distractor quality, difficulty) for team members to fill in.
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from typing import Dict, List

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from .models import Flashcard, Question
from .quality import is_answerable, is_grounded, jaccard, tokens
from .retrieval import TfidfRetriever

_PREFIXES = re.compile(
    r"^(fill in the blank:|true or false:|choose the correct option to complete the statement:)\s*", re.I
)


def _stem(q: str) -> str:
    return _PREFIXES.sub("", q).replace("_____", "").strip()


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 based on the longest common subsequence of words."""
    c, r = _words(candidate), _words(reference)
    if not c or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
    for i, cw in enumerate(c, 1):
        for j, rw in enumerate(r, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if cw == rw else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def bleu4(candidate: str, reference: str) -> float:
    return sentence_bleu([_words(reference)], _words(candidate),
                         smoothing_function=SmoothingFunction().method1)


def token_f1(pred: str, gold: str) -> float:
    p, g = _words(pred), _words(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def reference_metrics(questions: List[Question], cards: List[Flashcard], references: List[dict]) -> Dict:
    if not questions or not references:
        return {}
    gen_text = [_stem(q.question) + " " + q.answer for q in questions]
    ref_text = [r["question"] + " " + r["answer"] for r in references]

    # For each generated question: best-matching reference (precision-style).
    best_rouge, best_bleu = [], []
    for q, g in zip(questions, gen_text):
        scores = [rouge_l(g, r) for r in ref_text]
        j = max(range(len(ref_text)), key=scores.__getitem__)
        best_rouge.append(scores[j])
        best_bleu.append(bleu4(_stem(q.question), references[j]["question"]))

    # Concept coverage: a reference answer is "covered" if a generated answer / card overlaps it.
    tested = [q.answer + " " + q.source_sentence for q in questions if q.qtype != "true_false"]
    tested += [q.source_sentence for q in questions if q.qtype == "true_false"]
    tested += [c.front + " " + c.back for c in cards]
    covered = 0
    for r in references:
        ans = tokens(r["answer"]) - {"the", "a", "an", "of", "and", "by", "is"}
        if ans and any(len(ans & tokens(t)) / len(ans) >= 0.75 for t in tested):
            covered += 1

    n = len(questions)
    return {
        "rougeL_f1": round(sum(best_rouge) / n, 4),
        "bleu4": round(sum(best_bleu) / n, 4),
        "reference_concept_coverage": round(covered / len(references), 4),
    }


def concept_metrics(questions: List[Question], keyphrases: List[dict], top_k: int = 30) -> Dict:
    """How often answers / distractors are among the document's top key concepts.
    High distractor relevance means wrong options come from the same topic (plausible)."""
    from .keywords import normalize

    top = {k["key"] for k in keyphrases[:top_k]}
    answers = [normalize(q.answer) for q in questions if q.qtype in ("mcq", "fill_blank")]
    distractors = [normalize(o) for q in questions if q.qtype == "mcq" for o in q.options if o != q.answer]
    return {
        "answer_is_key_concept": round(sum(a in top for a in answers) / max(len(answers), 1), 4),
        "distractor_is_key_concept": round(sum(d in top for d in distractors) / max(len(distractors), 1), 4)
        if distractors else None,
    }


def intrinsic_metrics(questions: List[Question], source_sentences: List[str]) -> Dict:
    if not questions:
        return {}
    full = " ".join(source_sentences)
    retriever = TfidfRetriever(source_sentences)
    n = len(questions)
    grounded = sum(is_grounded(q.source_sentence, full) for q in questions)
    answerable = sum(is_answerable(q, retriever, source_sentences) for q in questions)
    mcqs = [q for q in questions if q.qtype == "mcq"]
    valid_mcq = sum(
        len(set(o.lower() for o in q.options)) == len(q.options) == 4 and q.answer in q.options for q in mcqs
    )
    pairs = [(a, b) for i, a in enumerate(questions) for b in questions[i + 1:]]
    mean_sim = sum(jaccard(a.question, b.question) for a, b in pairs) / max(len(pairs), 1)
    return {
        "n_questions": n,
        "grounded_rate": round(grounded / n, 4),
        "answerable_rate": round(answerable / n, 4),
        "mcq_validity": round(valid_mcq / len(mcqs), 4) if mcqs else None,
        "diversity_1_minus_mean_jaccard": round(1 - mean_sim, 4),
        "type_distribution": dict(Counter(q.qtype for q in questions)),
        "difficulty_distribution": dict(Counter(q.difficulty for q in questions)),
        "bloom_distribution": dict(Counter(q.bloom for q in questions)),
    }


def human_eval_sheet(questions: List[Question]) -> str:
    """CSV for manual rating (1-5) by evaluators; averaged in the report."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "type", "question", "options", "answer", "relevance_1to5", "fluency_1to5",
                "answerable_1to5", "distractor_quality_1to5", "difficulty_appropriate_1to5", "comments"])
    for i, q in enumerate(questions, 1):
        w.writerow([i, q.qtype, q.question, " | ".join(q.options), q.answer, "", "", "", "", "", ""])
    return buf.getvalue()


def summarize_human_ratings(csv_text: str) -> Dict[str, float]:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    out = {}
    for col in ("relevance_1to5", "fluency_1to5", "answerable_1to5", "distractor_quality_1to5",
                "difficulty_appropriate_1to5"):
        vals = [float(r[col]) for r in rows if r.get(col, "").strip()]
        if vals:
            out[col] = round(sum(vals) / len(vals), 3)
    return out
