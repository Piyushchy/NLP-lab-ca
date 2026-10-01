from pathlib import Path

import pytest

from quizgen import generate
from quizgen.classic import _replace_once, find_definition
from quizgen.evaluate import bleu4, reference_metrics, rouge_l, token_f1
from quizgen.export import to_anki, to_json, to_markdown, to_pdf
from quizgen.grading import check_answer
from quizgen.keywords import extract_keyphrases
from quizgen.models import Flashcard, Page, Question
from quizgen.preprocess import clean_text, split_sentences
from quizgen.quality import is_grounded, score_question
from quizgen.srs import due_cards, review

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample_textbook.pdf"


@pytest.fixture(scope="module")
def result():
    if not SAMPLE.exists():
        from scripts.make_sample_pdf import build
        build(SAMPLE)
    return generate(str(SAMPLE), n_questions=12, n_flashcards=10, engine="classic")


def test_clean_text_removes_headings_and_hyphenation():
    raw = "5.2 The Leaf\nThe leaf is the main site of photo-\nsynthesis in plants.\n12\n"
    assert clean_text(raw) == "The leaf is the main site of photosynthesis in plants."


def test_split_sentences_filters_fragments():
    pages = [Page(1, "Short one. Chlorophyll is a green pigment that absorbs light energy. See Figure 2 for the diagram.")]
    texts = [s.text for s in split_sentences(pages)]
    assert texts == ["Chlorophyll is a green pigment that absorbs light energy."]


def test_find_definition_patterns():
    assert find_definition("Photolysis is the splitting of water molecules by light energy.") == (
        "Photolysis", "the splitting of water molecules by light energy")
    assert find_definition("Transpiration refers to the loss of water vapour from the leaves.")[0] == "Transpiration"
    assert find_definition("It is a very important process for life.") is None


def test_replace_once_respects_word_boundaries_and_case():
    assert _replace_once("Leaves look green.", "leaves", "_____") == "_____ look green."
    assert _replace_once("Chloroplasts contain chlorophyll.", "chlorophyll", "X") == "Chloroplasts contain X."
    assert _replace_once("carbon dioxide", "oxide", "X") is None


def test_keyphrases_rank_domain_terms(result):
    top = {k["key"] for k in result.keyphrases[:20]}
    assert {"carbon fixation", "calvin cycle", "compensation point"} & top


def test_generates_requested_counts_and_types(result):
    assert len(result.questions) == 12
    assert len(result.flashcards) == 10
    assert {q.qtype for q in result.questions} == {"mcq", "true_false", "fill_blank", "short_answer"}


def test_mcq_well_formed(result):
    for q in (q for q in result.questions if q.qtype == "mcq"):
        assert len(q.options) == 4 and len(set(o.lower() for o in q.options)) == 4
        assert q.answer in q.options
        assert "_____" in q.question


def test_questions_are_grounded_in_source(result):
    full = " ".join(s.text for s in result.sentences)
    assert all(is_grounded(q.source_sentence, full) for q in result.questions)
    assert all(q.page is not None for q in result.questions)


def test_no_duplicate_source_sentences(result):
    srcs = [q.source_sentence for q in result.questions]
    assert len(srcs) == len(set(srcs))


def test_type_filter():
    r = generate(str(SAMPLE), n_questions=5, n_flashcards=3, types=["true_false"], engine="classic")
    assert r.questions and all(q.qtype == "true_false" for q in r.questions)
    assert {q.answer for q in r.questions} <= {"True", "False"}


def test_llm_engine_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = generate(str(SAMPLE), n_questions=3, n_flashcards=3, engine="llm")
    assert r.engine == "classic" and r.warnings


def test_score_question_rejects_broken_mcq():
    q = Question(qtype="mcq", question="Choose: _____ is green.", answer="leaf", options=["leaf", "leaf", "root", "stem"])
    assert score_question(q) == 0.0


def test_check_answer():
    fb = Question(qtype="fill_blank", question="_____ is ...", answer="carbon fixation")
    assert check_answer(fb, "Carbon Fixation")
    assert not check_answer(fb, "photolysis")
    sa = Question(qtype="short_answer", question="What is photolysis?", answer="the splitting of water molecules by light energy")
    assert check_answer(sa, "splitting of water molecules using light energy")
    tf = Question(qtype="true_false", question="True or False: ...", answer="True", options=["True", "False"])
    assert check_answer(tf, "True") and not check_answer(tf, "False")


def test_sm2_schedule():
    c = Flashcard(front="ATP", back="energy currency")
    review(c, 4, today=0)
    assert c.interval == 1 and c.due == 1
    review(c, 4, today=1)
    assert c.interval == 6 and c.due == 7
    review(c, 1, today=7)
    assert c.interval == 1 and c.repetitions == 0
    assert due_cards([c], today=8) == [c]


def test_metrics():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("dog", "cat") == 0.0
    assert token_f1("carbon dioxide", "carbon dioxide and water") == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert 0 <= bleu4("what is atp", "what is the main energy currency") <= 1


def test_reference_metrics(result):
    import json
    refs = json.loads((ROOT / "data" / "reference_questions.json").read_text())["questions"]
    m = reference_metrics(result.questions, result.flashcards, refs)
    assert m["reference_concept_coverage"] > 0.5


def test_exports(result):
    q, c = result.questions, result.flashcards
    assert to_json(q, c).startswith("{")
    assert "## Answer key" in to_markdown(q, c)
    assert to_pdf(q, c).startswith(b"%PDF")
    assert to_anki(c)[:2] == b"PK"  # .apkg is a zip file


def test_keyphrases_empty_input():
    assert extract_keyphrases([]) == []


def test_llm_path_parses_and_drops_ungrounded_items(monkeypatch):
    """The LLM engine with a fake Anthropic client: grounded items are kept, invented ones dropped."""
    import json
    import types as pytypes

    import anthropic

    payload = {
        "questions": [
            {"qtype": "mcq", "question": "Which enzyme catalyses carbon fixation?", "options": ["Rubisco", "Amylase", "Lipase", "Catalase"],
             "answer": "Rubisco", "explanation": "Stated in 5.4.", "source_quote": "Rubisco is the enzyme that catalyses carbon fixation.",
             "difficulty": "easy", "bloom": "remember"},
            {"qtype": "true_false", "question": "True or False: Photosynthesis happens in mitochondria.", "options": ["True", "False"],
             "answer": "False", "explanation": "Invented.", "source_quote": "Mitochondria are the powerhouse where photosynthesis occurs.",
             "difficulty": "easy", "bloom": "remember"},
        ],
        "flashcards": [{"front": "Photolysis", "back": "Splitting of water by light energy",
                        "source_quote": "Photolysis is the splitting of water molecules by light energy."}],
    }

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["output_config"]["format"]["type"] == "json_schema"
            return pytypes.SimpleNamespace(stop_reason="end_turn",
                                           content=[pytypes.SimpleNamespace(type="text", text=json.dumps(payload))])

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = pytypes.SimpleNamespace(messages=FakeMessages())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    r = generate(str(SAMPLE), n_questions=3, n_flashcards=3, engine="llm")
    assert r.engine == "llm"
    assert [q.answer for q in r.questions] == ["Rubisco"]  # the mitochondria item is not grounded
    assert r.questions[0].page == 2
    assert r.flashcards and r.flashcards[0].front == "Photolysis"


def test_baseline_and_concept_metrics(result):
    from quizgen.baseline import naive_questions
    from quizgen.evaluate import concept_metrics

    base = naive_questions(result.sentences, 10)
    assert base and all(q.engine == "baseline" for q in base)
    ours = concept_metrics(result.questions, result.keyphrases)
    theirs = concept_metrics(base, result.keyphrases)
    assert ours["answer_is_key_concept"] > theirs["answer_is_key_concept"]


def test_cli_generate_and_evaluate(tmp_path, monkeypatch):
    import json

    from quizgen.__main__ import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["generate", str(SAMPLE), "-n", "5", "-f", "4", "--engine", "classic", "--out", str(tmp_path)]) == 0
    for name in ("quiz.json", "quiz.md", "quiz.pdf", "questions.csv", "flashcards.csv", "flashcards.apkg"):
        assert (tmp_path / name).stat().st_size > 0
    out = tmp_path / "eval.json"
    assert main(["evaluate", str(SAMPLE), "--refs", str(ROOT / "data" / "reference_questions.json"),
                 "-n", "6", "--out", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert [r["system"] for r in rows][0] == "naive cloze baseline"
