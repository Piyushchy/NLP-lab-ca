# Educational Quiz & Flashcard Generator from PDF Textbooks

**Course:** Natural Language Processing (216U01E745), LY B.Tech Computer Engineering, Sem VII, 2026–27
**Assessment:** LAB CA, Task 2: Mini Project
**Faculty in-charge:** Dr. Grishma Sharma
**Group members:** _(names / roll numbers)_

---

## 1. Problem identification and objectives

Students revise from textbooks, but practice questions are scarce, generic, or not tied to the chapter they are studying. Teachers spend hours writing quizzes, and writing good multiple-choice distractors is especially slow. Self-testing (retrieval practice) and spaced repetition are two of the best-supported study techniques, but they need question banks and flashcards that most students never make.

**Problem statement.** Given a textbook chapter as a PDF, automatically produce a quiz and a set of flashcards. The output must be correct, grounded in the chapter (traceable to a page), varied in question type, and usable straight away for self-study.

**Objectives**

1. Extract and clean text from real-world PDFs: running headers and footers, page numbers, headings and hyphenation.
2. Identify the chapter's key concepts using classic NLP (POS tagging, chunking, TF-IDF, TextRank).
3. Generate four question types (MCQ, true/false, fill-in-the-blank, short answer) and flashcards, with plausible distractors.
4. Optionally use an LLM with retrieval-augmented generation (RAG), with a check that every item is grounded in the source.
5. Filter out low-quality items automatically.
6. Deliver an interactive app: quiz with scoring and explanations, SM-2 flashcards, and exports to PDF and Anki.
7. Evaluate against a naive baseline and teacher-written reference questions.

**Relevance.** This is an educational-tutor application, one of the suggested domains. It is useful to students for self-assessment and to teachers for preparing quizzes quickly.

## 2. System design and methodology

### 2.1 Architecture

```
PDF ─► Extraction ─► Preprocessing ─► Concept mining ─┬─► Classic generator ─┐
       (PyMuPDF)     (clean, split,    (POS, NP chunk, │                      ├─► Quality filter ─► App / exports
                      chunk)           TF-IDF+TextRank)└─► Retriever ─► LLM ──┘
```

### 2.2 Text extraction and preprocessing (`pdf_extract.py`, `preprocess.py`)

- **Extraction:** page-by-page text from PyMuPDF, with pypdf as a fallback. Scanned PDFs, which have no text layer, are detected and the user is told to run OCR first.
- **Running headers and footers:** lines that repeat on at least 60% of pages are removed, and so are page-number lines.
- **Headings:** a line is treated as a heading and dropped if it has at most 8 words, no closing punctuation, follows a finished sentence, and is Title Case or numbered ("5.2 The Leaf…"). This stops headings from merging into the next sentence.
- **Normalization:** words split across lines are rejoined, and bullets, ligatures and smart quotes are normalized.
- **Sentence segmentation:** NLTK Punkt. Unusable sentences are filtered out: fewer than 6 or more than 60 words, questions, references to figures or tables, or mostly non-alphabetic text.
- **Chunking:** about 180-word passages for retrieval and as LLM context.

### 2.3 Key-concept extraction (`nlp.py`, `keywords.py`)

1. **Candidates:** a POS-tag regular-expression chunker, `NP: {<JJ.*|VBN|VBG>*<NN.*>+(<IN><DT>?<JJ.*>*<NN.*>+)?}`, keeps only "of" attachments ("rate of photosynthesis"). Two corrections handle common tagger errors:
   - WordNet verb/noun sense frequencies drop a phrase-final word that is really a verb ("the rate of photosynthesis *increases* until…").
   - The same check, plus naming verbs, removes leading verbs ("*called* evapotranspiration").

   Stop words, quantifiers and generic nouns are stripped. Candidates are grouped by lemma so that "leaf" and "leaves" count as one.
2. **Ranking:** score = 0.45 × TextRank + 0.35 × TF-IDF + 0.2 × frequency + a 0.15 bonus for multi-word phrases.
   - TextRank is PageRank over a graph where two phrases are linked when they occur in the same sentence.
   - TF-IDF is the maximum score of the phrase over sentence-level documents.

   A phrase is dropped if it is contained in a higher-ranked phrase from the same sentences.
3. **Surface form:** each concept is displayed the way it is written mid-sentence, so sentence-initial capitals are not copied into options.

### 2.4 Question generation, classic engine (`classic.py`, `distractors.py`)

For each concept, from highest-ranked down, the system picks the most informative sentence that contains it. Mid-length sentences that mention other key concepts score higher. Sentences that open with an anaphor such as "These reactions…" are penalized, because they need the previous sentence for context. Question types are produced in round-robin order:

| Type | Method | Bloom level |
|---|---|---|
| Fill-in-the-blank | cloze deletion of the concept | remember |
| MCQ | cloze stem + 3 distractors + the answer, shuffled | understand |
| True/False | the original sentence (true), or a false version made by swapping in a distractor concept, or by negation if no distractor fits | understand |
| Short answer | definition patterns ("X is defined as / refers to / means / is a … Y", "Y is called X") produce "What is meant by X?" | remember |

**Distractors** are wrong options that should look plausible:

1. Other concepts from the same document whose context sentences have high TF-IDF cosine similarity to the answer's context. These are hard negatives from the same topic.
2. WordNet co-hyponyms (sister terms under the same hypernym) of single-word answers, excluding rare lemmas.

Options must match the answer's "shape": a similar number of words, the same capitalization, the same singular/plural form (so the sentence stays grammatical when one is swapped in), and digits only if the answer has digits. No option may share words with the answer or appear in the stem.

**Flashcards:** definition sentences become term → definition cards. Remaining top concepts become "Explain: concept" → context-sentence cards.

**Difficulty** comes from the concept's rank (central concepts are easier) and the sentence length.

**Variety.** Concepts are visited in a weighted random order (Efraimidis–Spirakis sampling with weight = score², so important concepts still come first more often), sentence choice gets a small random jitter, and the first question type and distractor picks are randomised. The app records which source sentences each PDF has already been quizzed on and penalises them, so repeated Generate presses move through the chapter. A concept whose only sentences were already used is skipped in a first pass and reused only if the quiz would otherwise be short. On the 3-page sample chapter (54 sentences, 10 questions per quiz) the first three quizzes share no sentences; the fourth reuses one and the fifth seven, as the chapter runs out of fresh material. A fixed seed reproduces a quiz exactly (used for the evaluation).

### 2.5 Question generation, LLM engine (`llm.py`)

- The retriever selects the passages that cover the most important concepts.
- Each passage is sent to an open-weight LLM on **Ollama Cloud** (default `gpt-oss:120b`; any Ollama model can be chosen) through the Ollama API (`https://ollama.com`, Bearer API key, free tier). A local Ollama install also works. The system prompt tells the model to use only facts in the passage and to copy the supporting sentence into `source_quote`. Temperature is 0.3.
- The output is constrained with a JSON schema (Ollama's `format` parameter), and the prompt includes a worked JSON example. Open models still deviate, so the reply is normalised before validation:
  - code fences and surrounding prose are stripped, and the question list is found wherever it is nested;
  - field-name variants are accepted ("type", "choices", "correct_answer", "statement", "evidence", …), and type names are mapped ("Multiple Choice" → mcq, "True/False" → true_false);
  - letter answers ("B", "B) ATP") and options given as a dict or with "A)" prefixes are resolved to the option text; boolean answers become True/False;
  - an item without a usable quote is grounded on the passage sentence that contains its answer, searching the whole document if needed, so its page number is correct.
- MCQs must still have 3–6 distinct options containing the answer, and true/false answers must be True or False.
- Generation is varied: passages are sampled with probability weighted by concept importance, temperature is 0.7, and the prompt lists recent questions not to repeat.
- If the API is unreachable, the reply cannot be parsed, or no valid question comes back, the app falls back to the classic engine and shows a warning with the start of the reply. If only some LLM items pass the quality filter, the quiz is topped up with classic questions.
- Items whose quote cannot be found in the PDF text are discarded as hallucinations. The test suite checks this with a fake model response that includes an invented fact.

### 2.6 Quality filter (`quality.py`)

1. **Rule checks:** MCQs must have 4 distinct options including the answer; the answer must not leak into the stem; stems must be a sensible length.
2. **Grounding:** LLM items must quote text that exists in the source (at least 80% token overlap).
3. **Round-trip answerability:** a lightweight stand-in for extractive QA.
   - The question is used as a query to the TF-IDF sentence retriever.
   - The item passes if the correct answer is supported by one of the top-3 retrieved sentences.
   - For true/false items, the original sentence must be retrievable from the statement.
   - A failure lowers the item's quality score by 0.2. Together with another problem, that drops it below the 0.5 threshold and the item is rejected.
4. **De-duplication:** items are dropped if their content-word Jaccard similarity to an earlier item is above 0.6, or if they share its source sentence. Template words are ignored, so "What is meant by X?" questions are not mistaken for duplicates of each other.

### 2.7 Application (`app.py`, `srs.py`, `export.py`)

- **Streamlit UI:** upload a PDF or use the sample, choose a page range, the number of questions and cards, the question types and the engine.
- **Quiz tab:** answers are scored. MCQ and true/false must match exactly. Typed answers are checked after lemma normalization; short answers also accept paraphrases with token-F1 ≥ 0.6. Each answer shows an explanation and the source page.
- **Flashcards tab:** SM-2 scheduling. Again/Hard/Good/Easy map to quality 1/3/4/5, which updates the ease factor, interval and due day.
- **Key-concepts tab:** the ranked concept table and a chart.
- **Export:** PDF worksheet with answer key (ReportLab), Markdown, CSV, JSON, and an Anki `.apkg` deck (genanki).

### 2.8 Tools and technologies

Python 3.11, PyMuPDF/pypdf, NLTK (Punkt, averaged perceptron tagger, RegexpParser, WordNet, stop words), scikit-learn (TF-IDF, cosine similarity), NetworkX (PageRank), Ollama Python client with Ollama Cloud models (optional), Streamlit, ReportLab, genanki, pytest.

## 3. Testing and performance evaluation

### 3.1 Test suite

`python -m pytest` runs 26 tests. They cover:

- Cleaning, heading and hyphenation handling, and sentence filtering.
- Definition patterns and word-boundary-safe cloze replacement.
- Keyphrase ranking.
- Requested question counts and types; MCQ well-formedness; grounding; no duplicate source sentences.
- The type filter; LLM fallback without a key; the Ollama path with a mocked client: cloud host and API-key header, dropping a hallucinated item and a malformed MCQ, parsing a fenced reply, a local host with a custom model, falling back on an unusable reply, and the real `ollama` HTTP client against a local stand-in server (request path, Bearer header, model and schema).
- A messy, realistic model reply (prose around fenced JSON, nested keys, "Multiple Choice" / "True/False" type names, letter answers, options as a dict or with "A)" prefixes, boolean answers, missing quotes) is fully parsed and grounded; and repeated generations use different sentences while a fixed seed still reproduces a quiz.
- Answer checking, the SM-2 schedule, the metric implementations, all export formats, and the CLI end to end.

The Streamlit app was also driven in a headless Chromium browser: generate → answer → submit → flashcard review → export download → evaluation tab, with no exceptions. The screenshots in `docs/screenshots/` come from that run.

**Robustness check:** a second PDF on a different subject (the water cycle), with running headers, page numbers and narrow wrapped columns, was processed correctly. Headers and footers were removed and lines rejoined, and it produced 10 sensible questions and 8 definition flashcards.

### 3.2 Evaluation setup

- **Data:** `data/sample_textbook.pdf`, an original 3-page chapter on photosynthesis (54 usable sentences).
- **References:** 26 teacher-style questions with answers (`data/reference_questions.json`).
- **Systems compared:** 15 questions each; every system is run with 5 random seeds (13–17) and the metrics are averaged. `docs/evaluation_results.json` also lists the min–max range for each metric.
  - **Naive cloze baseline** (`baseline.py`): a random sentence with a random noun blanked, and random nouns from the document as distractors.
  - **Classic NLP without the quality filter.**
  - **Classic NLP, full pipeline.**
  - **LLM + RAG (Ollama):** added automatically when `OLLAMA_API_KEY` or `OLLAMA_HOST` is set.

**Metrics**

| Metric | Meaning |
|---|---|
| ROUGE-L F1 / BLEU-4 | similarity of each generated question (plus answer) to its best-matching reference question |
| Reference concept coverage | share of reference answers tested by at least one generated question or flashcard |
| Answer is key concept | share of cloze/MCQ answers that are among the top-30 concepts. Does the quiz test what matters? |
| Distractor is key concept | share of distractors that are top-30 concepts of the same chapter. Are wrong options on-topic and plausible? |
| Grounded / answerable / MCQ validity | reference-free checks from §2.6 |
| Diversity | 1 − mean pairwise Jaccard similarity between questions |

### 3.3 Results

| System | ROUGE-L | BLEU-4 | Concept coverage | Answer is key concept | Distractor is key concept | Grounded | Answerable | MCQ valid | Diversity |
|---|---|---|---|---|---|---|---|---|---|
| Naive cloze baseline | 0.457 | **0.173** | 46.9% | 14.7% | 7.5% | 100% | 100% | 100% | 0.827 |
| Classic NLP, no filter | 0.533 | 0.113 | 79.2% | 83.9% | 71.7% | 100% | 100% | 100% | 0.887 |
| **Classic NLP, full** | **0.533** | 0.113 | **79.2%** | **83.9%** | **71.7%** | 100% | 100% | 100% | **0.887** |

Ranges over the 5 runs (full pipeline vs baseline): concept coverage 77–81% vs 35–58%, answer is key concept 71–100% vs 0–27%, distractor is key concept 56–83% vs 4–13%, BLEU-4 0.08–0.15 vs 0.10–0.24.

Question mix (full pipeline, first run): 3 MCQ, 4 true/false, 4 fill-in-the-blank and 4 short answer. Difficulty: 3 easy, 5 medium, 7 hard. Bloom levels: 8 remember, 7 understand.

### 3.4 Analysis

- **Concept selection works.** 84% of answers are top-30 key concepts, against 15% for the random baseline, and the quiz covers 79% of the concepts a teacher chose, against 47%. The figure is below 100% by design: concepts are sampled in a weighted random order so that repeated quizzes differ, which sometimes picks a lower-ranked concept.
- **Distractors are on-topic.** 72% of distractors are themselves key concepts of the chapter, against 8% for random nouns. That makes MCQs answerable only by someone who knows the material, not by spotting the odd one out. The rest come from WordNet sister terms and less central document concepts.
- **ROUGE-L and BLEU are low in absolute terms for every system.** The references are written as wh-questions ("Which enzyme catalyses carbon fixation?"), while the classic engine writes cloze statements. ROUGE-L favours the full pipeline (0.53 vs 0.46), but BLEU-4 favours the baseline (0.17 vs 0.11). BLEU-4 needs exact 4-word overlaps, and the baseline keeps whole sentences with only one word blanked, whereas our true/false and "What is meant by…" templates add or change words. BLEU-4 also varies widely between runs (0.10–0.24 for the baseline), so it says little about question quality here; the concept-based metrics are more informative.
- **Ablation: quality filter.** The classic generator enforces most constraints while it builds each item (one question per source sentence, shape-matched distractors, answers that are always in the source). As a result the filter rejected **none** of the 47 candidates the classic engine can produce from this chapter. On the water-cycle PDF it also rejected none. The "no filter" and "full" rows are therefore identical. The filter is a safety net for the LLM engine, where hallucinated, malformed and duplicate items do happen: the mocked-LLM test shows an invented "photosynthesis in mitochondria" item being removed. Measuring its rejection rate on real LLM output needs an Ollama API key: run `python -m quizgen evaluate` with `OLLAMA_API_KEY` set and add the row to the table.
- **Speed:** about 0.1–0.2 s per chapter once warm, about 3–5 s for the first call while NLTK loads, on a CPU. No GPU is needed.

### 3.5 Human evaluation protocol

The app and the CLI export `human_eval_sheet.csv`. Each group member independently rates each item from 1 to 5 for relevance, fluency, answerability, distractor quality and suitability of difficulty. `evaluate.summarize_human_ratings()` averages the scores. Fill in the table below before the presentation.

| Criterion | Member 1 | Member 2 | Member 3 | Mean |
|---|---|---|---|---|
| Relevance | | | | |
| Fluency | | | | |
| Answerability | | | | |
| Distractor quality | | | | |
| Difficulty appropriate | | | | |

## 4. Innovation and use of NLP/LLMs

- A **hybrid design**. The classic NLP engine is transparent and runs offline; the optional LLM engine produces more varied, higher-Bloom questions. Both go through the **same grounding and quality filter**, so every item can be traced to a sentence and page in the textbook.
- **Hallucination control for the LLM:** retrieval-scoped context, a mandatory verbatim source quote, schema-constrained JSON, and automatic rejection of anything unsupported.
- **Context-aware distractors** combine document vector-space similarity with WordNet taxonomy and grammatical-agreement checks.
- **A learning loop, not only generation:** SM-2 spaced repetition and Anki export turn the output into a study tool.
- **Built-in evaluation:** a baseline, reference metrics, reference-free metrics and a human-rating workflow.

## 5. Limitations

- Scanned PDFs need OCR first (the app detects this and says so). Equations, tables and figures are ignored.
- The classic engine produces recall- and comprehension-level questions (Bloom: remember and understand). Application and analysis questions need the LLM engine.
- The POS tagger makes mistakes on unusual sentences. Heuristics catch the common ones (verb read as a noun), but some odd concepts still get through ("deforestation increases").
- A false true/false statement made by swapping a concept can occasionally be ungrammatical ("…membranes of the photosynthesis").
- English only. The reference set is small (one chapter, 26 questions), so the results show a trend and are not statistically conclusive.

## 6. Future work

- Neural question generation (e.g. a T5 model fine-tuned on SQuAD) as a third engine, plus a neural QA model (e.g. a DeBERTa model fine-tuned on SQuAD 2.0) for the answerability check.
- Sentence-embedding retrieval and distractor similarity in place of TF-IDF.
- OCR integration (Tesseract / ocrmypdf), and parsing tables and figure captions.
- Multilingual support, e.g. Hindi and Marathi textbooks.
- Adaptive quizzes: choose the next question from the learner's mistakes and SM-2 state.
- A larger human evaluation across several subjects, with inter-rater agreement (Cohen's κ).

## 7. How to run

See [`README.md`](../README.md). In short: `pip install -r requirements.txt`, then download the NLTK data, then `streamlit run app.py`.
