"""Streamlit front-end:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from quizgen import export
from quizgen.evaluate import human_eval_sheet, intrinsic_metrics
from quizgen.grading import check_answer
from quizgen.llm import SUGGESTED_MODELS, default_model, llm_available
from quizgen.models import QUESTION_TYPES
from quizgen.pipeline import generate
from quizgen.srs import due_cards, review

SAMPLE = Path(__file__).parent / "data" / "sample_textbook.pdf"
TYPE_LABELS = {"mcq": "Multiple choice", "true_false": "True / False",
               "fill_blank": "Fill in the blank", "short_answer": "Short answer"}

st.set_page_config(page_title="Textbook Quiz & Flashcard Generator", page_icon="📚", layout="wide")
st.title("📚 Educational Quiz & Flashcard Generator")
st.caption("Upload a textbook chapter (PDF) and get an auto-generated quiz and spaced-repetition flashcards.")

# ---------------- sidebar: inputs ----------------
with st.sidebar:
    st.header("1. Source")
    upload = st.file_uploader("Textbook PDF", type=["pdf"])
    use_sample = st.checkbox("Use the sample chapter (Photosynthesis)", value=upload is None)
    pages = st.text_input("Page range (optional, e.g. 2-5)")

    st.header("2. Options")
    n_q = st.slider("Number of questions", 3, 40, 10)
    n_c = st.slider("Number of flashcards", 3, 40, 10)
    types = st.multiselect("Question types", list(QUESTION_TYPES), default=list(QUESTION_TYPES),
                           format_func=TYPE_LABELS.get)
    engine = st.radio("Generation engine", ["auto", "classic", "llm"], horizontal=True,
                      help="classic = NLP pipeline only; llm = Ollama model, grounded on retrieved passages; "
                           "auto = llm when Ollama is configured.")
    key = st.text_input("Ollama API key (optional)", type="password",
                        help="Free key from ollama.com/settings/keys. Used only for this browser session; "
                             "never written to the server environment.") or None
    local_only = os.environ.get("OLLAMA_HOST") and not (key or os.environ.get("OLLAMA_API_KEY"))
    choices = ([default_model()] if local_only or os.environ.get("QUIZGEN_MODEL") else []) + SUGGESTED_MODELS
    model = st.selectbox("Ollama model", list(dict.fromkeys(choices)), accept_new_options=True,
                         help="Cloud models via the Ollama API, e.g. gpt-oss:120b. Type any other model name.")
    st.caption("LLM available ✅" if llm_available(key) else "Ollama not configured — classic engine will be used.")
    go = st.button("Generate", type="primary", width="stretch")
    st.caption("Each Generate avoids questions already asked from the same PDF.")
    if st.button("Forget previous questions", width="stretch"):
        st.session_state.history = {}
        st.toast("Question history cleared.")


def _range(text: str):
    if not text.strip():
        return None
    a, _, b = text.partition("-")
    return int(a), int(b or a)


if go:
    source = upload.getvalue() if upload is not None and not use_sample else (SAMPLE.read_bytes() if use_sample else None)
    if source is None:
        st.error("Upload a PDF or tick 'Use the sample chapter'.")
    elif not types:
        st.error("Select at least one question type.")
    else:
        # Remember what each PDF has already been asked, so every Generate gives new questions.
        doc_id = hashlib.sha1(source).hexdigest()
        history = st.session_state.setdefault("history", {}).setdefault(doc_id, {"sources": set(), "questions": []})
        with st.spinner("Reading the PDF and generating questions..."):
            try:
                result = generate(source, n_questions=n_q, n_flashcards=n_c, types=types,
                                  engine=engine, page_range=_range(pages), api_key=key, model=model,
                                  avoid=history["sources"], avoid_questions=history["questions"][-15:])
                history["sources"] |= {q.source_sentence for q in result.questions}
                history["sources"] |= {c.source_sentence for c in result.flashcards}
                history["questions"] += [q.question for q in result.questions]
                st.session_state.result = result
                st.session_state.quiz_no = st.session_state.get("quiz_no", 0) + 1
                st.session_state.title = (upload.name.rsplit(".", 1)[0] if upload is not None and not use_sample
                                          else "Photosynthesis and Plant Nutrition")
                st.session_state.submitted = False
                st.session_state.day = 0
            except ValueError as e:
                st.error(str(e))

result = st.session_state.get("result")
if result is None:
    st.info("Choose a PDF and press **Generate** in the sidebar.")
    st.stop()

for w in result.warnings:
    st.warning(w)
s = result.stats
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Engine", result.engine)
c2.metric("Pages", s["pages"])
c3.metric("Sentences", s["sentences"])
c4.metric("Questions", s["kept_questions"], help=f"{s['raw_questions']} generated before filtering")
c5.metric("Flashcards", s["kept_flashcards"])

tab_quiz, tab_cards, tab_concepts, tab_export, tab_eval = st.tabs(
    ["📝 Quiz", "🃏 Flashcards", "🔑 Key concepts", "⬇️ Export", "📊 Evaluation"])

# ---------------- quiz ----------------
with tab_quiz:
    quiz_no = st.session_state.get("quiz_no", 0)
    with st.form("quiz"):
        responses = {}
        for i, q in enumerate(result.questions):
            st.markdown(f"**Q{i + 1}.** {q.question}  \n"
                        f"<small>{TYPE_LABELS[q.qtype]} · {q.difficulty} · Bloom: {q.bloom} · p. {q.page}</small>",
                        unsafe_allow_html=True)
            if q.qtype in ("mcq", "true_false"):
                responses[i] = st.radio("Answer", q.options, index=None, key=f"q{quiz_no}-{i}", label_visibility="collapsed")
            else:
                responses[i] = st.text_input("Answer", key=f"q{quiz_no}-{i}", label_visibility="collapsed")
        if st.form_submit_button("Submit answers"):
            st.session_state.submitted = True
            st.session_state.responses = responses

    if st.session_state.get("submitted"):
        resp = st.session_state.responses
        correct = [check_answer(q, resp.get(i)) for i, q in enumerate(result.questions)]
        st.subheader(f"Score: {sum(correct)} / {len(correct)}")
        st.progress(sum(correct) / max(len(correct), 1))
        for i, (q, ok) in enumerate(zip(result.questions, correct)):
            with st.expander(f"{'✅' if ok else '❌'} Q{i + 1}: correct answer — {q.answer}"):
                st.write(f"Your answer: {resp.get(i) or '—'}")
                st.write(q.explanation)

# ---------------- flashcards (SM-2) ----------------
with tab_cards:
    cards = result.flashcards
    day = st.session_state.get("day", 0)
    due = due_cards(cards, day)
    st.caption(f"Study day {day} · {len(due)} card(s) due · spaced repetition with the SM-2 algorithm")
    if not due:
        nxt = min((c.due for c in cards), default=day)
        st.success(f"No cards due. Next review on day {nxt}.")
        if st.button("Advance to next study day"):
            st.session_state.day = nxt
            st.rerun()
    else:
        card = due[0]
        st.markdown(f"### {card.front}")
        if st.toggle("Show answer", key=f"flip-{id(card)}-{card.repetitions}-{card.due}"):
            st.info(card.back)
            st.caption(f"Source: p. {card.page}")
            cols = st.columns(4)
            for col, (label, quality) in zip(cols, [("Again", 1), ("Hard", 3), ("Good", 4), ("Easy", 5)]):
                if col.button(label, width="stretch"):
                    review(card, quality, day)
                    st.rerun()
    with st.expander("All flashcards"):
        st.dataframe(pd.DataFrame([{"front": c.front, "back": c.back, "page": c.page, "due day": c.due,
                                    "interval": c.interval} for c in cards]), width="stretch")

# ---------------- key concepts ----------------
with tab_concepts:
    st.write("Key concepts ranked by a combination of TextRank and TF-IDF over noun-phrase candidates.")
    df = pd.DataFrame([{"concept": k["phrase"], "score": k["score"], "occurrences": k["count"]}
                       for k in result.keyphrases])
    st.dataframe(df, width="stretch")
    if not df.empty:
        st.bar_chart(df.head(15).set_index("concept")["score"])

# ---------------- export ----------------
with tab_export:
    title = st.session_state.get("title", "Quiz")
    q, c = result.questions, result.flashcards
    st.download_button("Quiz + answer key (PDF)", export.to_pdf(q, c, title), "quiz.pdf", "application/pdf")
    st.download_button("Quiz (Markdown)", export.to_markdown(q, c, title), "quiz.md", "text/markdown")
    st.download_button("Questions (CSV)", export.questions_csv(q), "questions.csv", "text/csv")
    st.download_button("Flashcards (CSV)", export.flashcards_csv(c), "flashcards.csv", "text/csv")
    st.download_button("Everything (JSON)", export.to_json(q, c), "quiz.json", "application/json")
    try:
        st.download_button("Anki deck (.apkg)", export.to_anki(c, title), "flashcards.apkg",
                           "application/octet-stream")
    except ImportError:
        st.caption("Install `genanki` to enable Anki export.")

# ---------------- evaluation ----------------
with tab_eval:
    st.write("Reference-free quality metrics for this quiz (see `python -m quizgen evaluate` for the full study).")
    m = intrinsic_metrics(result.questions, [x.text for x in result.sentences])
    cols = st.columns(4)
    cols[0].metric("Grounded in text", f"{m.get('grounded_rate', 0):.0%}")
    cols[1].metric("Answerable (round-trip)", f"{m.get('answerable_rate', 0):.0%}")
    cols[2].metric("Valid MCQs", f"{m['mcq_validity']:.0%}" if m.get("mcq_validity") is not None else "—")
    cols[3].metric("Diversity", f"{m.get('diversity_1_minus_mean_jaccard', 0):.2f}")
    st.json({k: m[k] for k in ("type_distribution", "difficulty_distribution", "bloom_distribution") if k in m})
    st.download_button("Human evaluation sheet (CSV)", human_eval_sheet(result.questions), "human_eval_sheet.csv",
                       "text/csv")
