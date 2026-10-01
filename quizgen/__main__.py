"""Command-line interface.

    python -m quizgen generate data/sample_textbook.pdf -n 10 -f 10 --out outputs
    python -m quizgen evaluate data/sample_textbook.pdf --refs data/reference_questions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import export
from .baseline import naive_questions
from .evaluate import concept_metrics, human_eval_sheet, intrinsic_metrics, reference_metrics
from .llm import default_model, llm_available
from .models import QUESTION_TYPES
from .pipeline import generate


def _page_range(text: str | None):
    if not text:
        return None
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def cmd_generate(args) -> int:
    result = generate(args.pdf, n_questions=args.n, n_flashcards=args.flashcards,
                      types=args.types.split(","), engine=args.engine,
                      page_range=_page_range(args.pages), seed=args.seed, model=args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    title = Path(args.pdf).stem.replace("_", " ").title()
    (out / "quiz.json").write_text(export.to_json(result.questions, result.flashcards), encoding="utf-8")
    (out / "quiz.md").write_text(export.to_markdown(result.questions, result.flashcards, title), encoding="utf-8")
    (out / "questions.csv").write_text(export.questions_csv(result.questions), encoding="utf-8")
    (out / "flashcards.csv").write_text(export.flashcards_csv(result.flashcards), encoding="utf-8")
    (out / "quiz.pdf").write_bytes(export.to_pdf(result.questions, result.flashcards, title))
    (out / "human_eval_sheet.csv").write_text(human_eval_sheet(result.questions), encoding="utf-8")
    try:
        (out / "flashcards.apkg").write_bytes(export.to_anki(result.flashcards, title))
    except ImportError:
        pass
    for w in result.warnings:
        print("warning:", w, file=sys.stderr)
    print(f"engine={result.engine} stats={result.stats}")
    print(f"wrote quiz.json, quiz.md, quiz.pdf, questions.csv, flashcards.csv, flashcards.apkg to {out}/")
    return 0


def cmd_evaluate(args) -> int:
    refs = json.loads(Path(args.refs).read_text())["questions"] if args.refs else []

    def row(name, questions, cards, r):
        out = {"system": name, "n_questions": len(questions)}
        out.update(reference_metrics(questions, cards, refs))
        out.update(concept_metrics(questions, r.keyphrases))
        m = intrinsic_metrics(questions, [s.text for s in r.sentences])
        out.update({k: v for k, v in m.items() if k != "n_questions"})
        return out

    def average(runs: list) -> dict:
        """Mean of each numeric metric over runs with different seeds (+ the spread as min/max)."""
        out = {"system": runs[0]["system"], "runs": len(runs)}
        for k, v in runs[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
                out[k] = round(sum(vals) / len(vals), 4)
                if len(runs) > 1 and k not in ("n_questions",):
                    out[k + "_range"] = [round(min(vals), 4), round(max(vals), 4)]
            elif k != "system":
                out[k] = v  # distributions: from the first run
        return out

    systems = {"naive cloze baseline": [], "classic NLP, no quality filter": [], "classic NLP (full pipeline)": []}
    llm_runs, model = [], args.model or default_model()
    for seed in range(args.seed, args.seed + args.runs):
        full = generate(args.pdf, n_questions=args.n, n_flashcards=args.n, engine="classic", seed=seed)
        base = naive_questions(full.sentences, args.n, seed=seed)
        systems["naive cloze baseline"].append(row("naive cloze baseline", base, [], full))
        nofilter = generate(args.pdf, n_questions=args.n, n_flashcards=args.n, engine="classic", seed=seed,
                            use_filter=False)
        systems["classic NLP, no quality filter"].append(
            row("classic NLP, no quality filter", nofilter.questions, nofilter.flashcards, nofilter))
        systems["classic NLP (full pipeline)"].append(
            row("classic NLP (full pipeline)", full.questions, full.flashcards, full))
        if llm_available():
            llm = generate(args.pdf, n_questions=args.n, n_flashcards=args.n, engine="llm", seed=seed, model=model)
            for w in llm.warnings:
                print("warning:", w, file=sys.stderr)
            if llm.engine == "llm":
                llm_runs.append(row(f"LLM + RAG (Ollama {model})", llm.questions, llm.flashcards, llm))
    rows = [average(r) for r in systems.values()]
    if llm_runs:
        rows.append(average(llm_runs))
    elif not llm_available():
        print("note: no OLLAMA_API_KEY / OLLAMA_HOST set, so the LLM engine was not evaluated", file=sys.stderr)
    text = json.dumps(rows, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quizgen", description="Quiz & flashcard generator for PDF textbooks")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a quiz and flashcards from a PDF")
    g.add_argument("pdf")
    g.add_argument("-n", type=int, default=10, help="number of questions")
    g.add_argument("-f", "--flashcards", type=int, default=10, help="number of flashcards")
    g.add_argument("--types", default=",".join(QUESTION_TYPES), help="comma-separated question types")
    g.add_argument("--engine", choices=["auto", "classic", "llm"], default="auto")
    g.add_argument("--pages", help="page range, e.g. 3-7")
    g.add_argument("--model", help="Ollama model for --engine llm (default: gpt-oss:120b on Ollama Cloud)")
    g.add_argument("--seed", type=int, default=None, help="fix the random choices to reproduce a quiz")
    g.add_argument("--out", default="outputs")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="run the evaluation / ablation study")
    e.add_argument("pdf")
    e.add_argument("--refs", help="reference questions JSON")
    e.add_argument("-n", type=int, default=15)
    e.add_argument("--model", help="Ollama model to evaluate when an Ollama key / host is set")
    e.add_argument("--seed", type=int, default=13, help="first seed; runs use seed, seed+1, ...")
    e.add_argument("--runs", type=int, default=5, help="number of seeds to average over")
    e.add_argument("--out", help="write results JSON here")
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
