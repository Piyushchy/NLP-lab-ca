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
    return generate(str(SAMPLE), n_questions=12, n_flashcards=10, engine="classic", seed=13)


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
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
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


class FakeOllamaClient:
    """Stands in for ollama.Client; records how it was constructed and called."""
    reply = ""
    instances = []

    def __init__(self, host=None, headers=None, **kwargs):
        self.host, self.headers, self.calls = host, headers or {}, []
        FakeOllamaClient.instances.append(self)

    def chat(self, model, messages, format=None, options=None, **kwargs):
        import types as pytypes

        self.calls.append({"model": model, "format": format, "messages": messages})
        return pytypes.SimpleNamespace(message=pytypes.SimpleNamespace(content=FakeOllamaClient.reply))


@pytest.fixture
def fake_ollama(monkeypatch):
    import ollama

    for var in ("OLLAMA_API_KEY", "OLLAMA_HOST", "QUIZGEN_MODEL"):
        monkeypatch.delenv(var, raising=False)
    FakeOllamaClient.instances = []
    monkeypatch.setattr(ollama, "Client", FakeOllamaClient)
    return FakeOllamaClient


def test_llm_path_parses_and_drops_ungrounded_items(fake_ollama, monkeypatch):
    """Ollama Cloud engine with a fake client: grounded items are kept, invented / malformed ones dropped."""
    import json

    payload = {
        "questions": [
            {"qtype": "mcq", "question": "Which enzyme catalyses carbon fixation?", "options": ["Rubisco", "Amylase", "Lipase", "Catalase"],
             "answer": "rubisco", "explanation": "Stated in 5.4.", "source_quote": "Rubisco is the enzyme that catalyses carbon fixation.",
             "difficulty": "easy", "bloom": "remember"},
            {"qtype": "true_false", "question": "True or False: Photosynthesis happens in mitochondria.", "options": ["True", "False"],
             "answer": "False", "explanation": "Invented.", "source_quote": "Mitochondria are the powerhouse where photosynthesis occurs.",
             "difficulty": "easy", "bloom": "remember"},
            {"qtype": "mcq", "question": "Where is chlorophyll found?", "options": ["Stroma", "Nucleus"],
             "answer": "Chloroplast", "explanation": "", "source_quote": "Chlorophyll is a green pigment that absorbs light energy.",
             "difficulty": "easy", "bloom": "remember"},
        ],
        "flashcards": [{"front": "Photolysis", "back": "Splitting of water by light energy",
                        "source_quote": "Photolysis is the splitting of water molecules by light energy."}],
    }
    fake_ollama.reply = "```json\n" + json.dumps(payload) + "\n```"  # fenced reply must still parse
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    r = generate(str(SAMPLE), n_questions=3, n_flashcards=3, engine="llm")
    assert r.engine == "llm", r.warnings
    llm_qs = [q for q in r.questions if q.engine == "llm"]
    assert [q.answer for q in llm_qs] == ["Rubisco"]  # ungrounded T/F and broken MCQ are dropped
    assert llm_qs[0].page == 2
    # too few LLM questions survived, so the quiz is topped up from the classic engine
    assert len(r.questions) == 3 and any("rest come from the classic engine" in w for w in r.warnings)
    assert r.flashcards and r.flashcards[0].front == "Photolysis"
    client = fake_ollama.instances[0]
    assert client.host == "https://ollama.com" and client.headers["Authorization"] == "Bearer test-key"
    assert client.calls[0]["model"] == "gpt-oss:120b"
    assert client.calls[0]["format"]["required"] == ["questions", "flashcards"]


def test_llm_local_host_and_model_override(fake_ollama, monkeypatch):
    fake_ollama.reply = '{"questions": [], "flashcards": []}'
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    r = generate(str(SAMPLE), n_questions=3, n_flashcards=3, engine="llm", model="llama3.2")
    client = fake_ollama.instances[0]
    assert client.host == "http://localhost:11434" and "Authorization" not in client.headers
    assert client.calls[0]["model"] == "llama3.2"
    # nothing usable came back, so the app falls back to the classic engine
    assert r.engine == "classic" and r.questions and any("no usable" in w for w in r.warnings)


def test_llm_garbage_reply_falls_back(fake_ollama, monkeypatch):
    fake_ollama.reply = "Sorry, I cannot help with that."
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    r = generate(str(SAMPLE), n_questions=3, n_flashcards=3, engine="llm")
    assert r.engine == "classic" and r.questions and any("LLM call failed" in w for w in r.warnings)


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

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert main(["generate", str(SAMPLE), "-n", "5", "-f", "4", "--engine", "classic", "--out", str(tmp_path)]) == 0
    for name in ("quiz.json", "quiz.md", "quiz.pdf", "questions.csv", "flashcards.csv", "flashcards.apkg"):
        assert (tmp_path / name).stat().st_size > 0
    out = tmp_path / "eval.json"
    assert main(["evaluate", str(SAMPLE), "--refs", str(ROOT / "data" / "reference_questions.json"),
                 "-n", "6", "--runs", "2", "--out", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert [r["system"] for r in rows][0] == "naive cloze baseline"


def test_real_ollama_client_against_local_fake_server(monkeypatch):
    """Exercise the real `ollama` HTTP client end to end against a local stand-in for ollama.com."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from quizgen import llm

    seen = {}
    content = json.dumps({
        "questions": [{"qtype": "fill_blank", "question": "_____ is the enzyme that catalyses carbon fixation.",
                       "options": [], "answer": "Rubisco", "explanation": "Section 5.4",
                       "source_quote": "Rubisco is the enzyme that catalyses carbon fixation.",
                       "difficulty": "easy", "bloom": "remember"}],
        "flashcards": [],
    })

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(path=self.path, auth=self.headers.get("Authorization"), body=body)
            reply = json.dumps({"model": body["model"], "created_at": "2026-10-01T00:00:00Z",
                                "message": {"role": "assistant", "content": content},
                                "done": True, "done_reason": "stop"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(llm, "CLOUD_HOST", f"http://127.0.0.1:{server.server_port}")
        for var in ("OLLAMA_HOST", "QUIZGEN_MODEL"):
            monkeypatch.delenv(var, raising=False)
        r = generate(str(SAMPLE), n_questions=2, n_flashcards=2, types=["fill_blank"], engine="llm",
                     api_key="sk-test", model="gpt-oss:20b")
    finally:
        server.shutdown()
    assert seen["path"] == "/api/chat" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-oss:20b" and seen["body"]["stream"] is False
    assert seen["body"]["format"]["required"] == ["questions", "flashcards"]
    assert r.engine == "llm", r.warnings
    assert [q.answer for q in r.questions if q.engine == "llm"] == ["Rubisco"]


MESSY_REPLY = """Here is your quiz:
```json
{"quiz": {"questions": [
  {"type": "Multiple Choice", "question": "Which enzyme catalyses carbon fixation?",
   "choices": {"A": "Amylase", "B": "Rubisco", "C": "Lipase", "D": "Catalase"}, "correct_answer": "B",
   "source": "Rubisco is the enzyme that catalyses carbon fixation."},
  {"question_type": "True/False", "statement": "Photolysis releases oxygen, protons and electrons.",
   "answer": true, "evidence": "Photolysis releases oxygen, protons and electrons."},
  {"type": "fill in the blank", "question": "The Calvin cycle takes place in the ___.",
   "answer": "stroma"},
  {"type": "multiple-choice", "question": "What is the main energy currency of the cell?",
   "options": ["A) Glucose", "B) ATP", "C) Starch", "D) NADPH"], "answer": "B) ATP"}
 ],
 "flashcards": [{"term": "Rubisco", "definition": "The enzyme that catalyses carbon fixation."}]}}
```"""


def test_llm_accepts_messy_real_world_reply(fake_ollama, monkeypatch):
    """Open models rename fields, use letters for answers and wrap JSON in prose; all of it should parse."""
    fake_ollama.reply = MESSY_REPLY
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    r = generate(str(SAMPLE), n_questions=4, n_flashcards=1, engine="llm", seed=1)
    assert r.engine == "llm", r.warnings
    got = {q.answer: q for q in r.questions if q.engine == "llm"}
    assert set(got) == {"Rubisco", "True", "stroma", "ATP"}
    assert got["Rubisco"].qtype == "mcq" and got["Rubisco"].options == ["Amylase", "Rubisco", "Lipase", "Catalase"]
    assert got["ATP"].options == ["Glucose", "ATP", "Starch", "NADPH"]
    assert got["True"].qtype == "true_false" and got["stroma"].qtype == "fill_blank"
    assert "_____" in got["stroma"].question
    # no quote was given for the fill-in-the-blank, so it is grounded on the sentence containing the answer
    assert "Calvin cycle" in got["stroma"].source_sentence and got["stroma"].page == 2
    assert r.flashcards[0].front == "Rubisco"


def test_each_generation_gives_different_questions():
    first = generate(str(SAMPLE), n_questions=8, n_flashcards=5, engine="classic")
    second = generate(str(SAMPLE), n_questions=8, n_flashcards=5, engine="classic",
                      avoid={q.source_sentence for q in first.questions})
    a = {q.source_sentence for q in first.questions}
    b = {q.source_sentence for q in second.questions}
    assert len(a & b) <= 1, (a & b)  # the second quiz moves on to new sentences
    # a fixed seed still reproduces the same quiz (used by the evaluation)
    x = generate(str(SAMPLE), n_questions=6, engine="classic", seed=7)
    y = generate(str(SAMPLE), n_questions=6, engine="classic", seed=7)
    assert [q.question for q in x.questions] == [q.question for q in y.questions]
