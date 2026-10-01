# Presentation outline (about 10 minutes plus a live demo)

Rubric weights are in brackets. Speaker notes are in italics.

1. **Title.** Educational Quiz & Flashcard Generator from PDF Textbooks: group members, course, faculty.
2. **Problem** [3]. Students lack chapter-specific practice questions, and teachers spend hours writing quizzes and distractors. Retrieval practice and spaced repetition work, but need question banks. *Mention: Educational Tutor domain.*
3. **Objectives.** PDF → clean text → key concepts → 4 question types plus flashcards → quality filter → interactive app and exports → evaluation.
4. **Architecture** [7]. Show the diagram from the README. Two engines (classic NLP, LLM + RAG) and one shared quality filter.
5. **Preprocessing.** Header/footer removal (lines repeated on ≥60% of pages), heading detection, de-hyphenation, Punkt sentence split, sentence filtering. *Show a before/after snippet.*
6. **Key-concept extraction.** POS tagging, then a regexp NP chunker with tagger-error fixes from WordNet sense frequencies, then TF-IDF and TextRank fusion. *Show the Key concepts tab.*
7. **Question generation.** Cloze, MCQ, true/false (concept swap or negation), definition patterns, flashcards. Bloom level and difficulty tags.
8. **Distractors.** TF-IDF context similarity between document concepts, plus WordNet co-hyponyms, plus shape and plural-agreement constraints. *Example: "leaf" → root, blue light, nectary.*
9. **LLM + RAG engine** [4]. Retrieval of key passages, Claude with JSON-schema output, mandatory source quote, grounding check that drops hallucinations.
10. **Quality filter.** Rule checks, grounding, round-trip answerability with the retriever, de-duplication.
11. **Live demo** [3]. See the script below.
12. **Evaluation** [5]. Results table from REPORT §3.3: concept coverage 81% vs 35%, answers that are key concepts 100% vs 27%, distractors that are key concepts 50% vs 8%. Then the human-rating table.
13. **Limitations and future work.** OCR, Bloom levels above "understand" need the LLM, POS-tagger errors, English only. Next: T5 question generation, DeBERTa QA, embeddings, multilingual, adaptive quizzes.
14. **Conclusion and Q&A.**

## Demo script (about 4 minutes)

1. `streamlit run app.py` (start it before the presentation; the first load takes about 5 s).
2. Tick **Use the sample chapter** and press **Generate**. Point out the counters: pages, sentences, and questions kept after filtering.
3. **Quiz tab:** answer 3–4 questions (get one wrong on purpose), press **Submit**, then open the explanation with its page number.
4. **Flashcards tab:** reveal a card and press **Good**; the due count drops (SM-2).
5. **Key concepts tab:** the ranked table and bar chart.
6. **Export tab:** download the PDF worksheet and the Anki deck.
7. **Evaluation tab:** grounded, answerable and valid-MCQ rates.
8. Upload a chapter from your own textbook (keep one ready) to show it generalizes.
9. If an API key is available, switch the engine to **llm** and regenerate to show the varied, higher-Bloom questions.

## Suggested work split (group of 3)

| Member | Owns | Presents |
|---|---|---|
| 1 | `pdf_extract`, `preprocess`, `nlp`, `keywords` | slides 2–6 |
| 2 | `classic`, `distractors`, `llm`, `quality`, `retrieval` | slides 7–10 |
| 3 | `app.py`, `srs`, `export`, `evaluate`, report | slides 11–13 and demo |
