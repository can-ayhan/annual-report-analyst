"""Run the question set against a model and a prompt variant, and score what comes back.

Three things are measured per question:

* correct   the expected figures are in the answer (or, for a question the library
            cannot answer, the system said so);
* cited     the answer cites a page on which the figure is stated;
* cost and time, from the tokens and the duration of every model call.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from analyst.agents.state import Answer
from analyst.agents.verifier import normalise, squash_numbers
from analyst.config import ROOT
from analyst.graph import ask
from analyst.llm import ChatModel, LLMError, call_as_dict
from analyst.search import ReportSearch

QUESTIONS_FILE = ROOT / "evals" / "questions.yml"
MODELS_FILE = ROOT / "evals" / "models.yml"
RESULTS_DIR = ROOT / "evals" / "results"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expect: list[list[str]] = field(default_factory=list)
    pages: dict[str, list[int]] = field(default_factory=dict)
    no_answer: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    options: dict = field(default_factory=dict)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok) / 1_000_000


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Question(**item) for item in data["questions"]]


def load_models(path: Path = MODELS_FILE) -> dict[str, ModelSpec]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {name: ModelSpec(name=name, **(spec or {})) for name, spec in (data.get("models") or {}).items()}


def comparable(text: str) -> str:
    return squash_numbers(normalise(text))


def score(question: Question, answer: Answer) -> dict:
    """Was the answer right, and does it point at a page that says so?"""
    if question.no_answer:
        return {"correct": not answer.answered, "cited": None}
    text = comparable(" ".join(claim.text for claim in answer.claims))
    found = all(any(comparable(way) in text for way in group) for group in question.expect)
    cited_pages = {(c.report_id, c.page_no) for claim in answer.claims for c in claim.citations}
    cited = all(
        any((report_id, page) in cited_pages for page in pages) for report_id, pages in question.pages.items()
    )
    return {"correct": answer.answered and found, "cited": answer.answered and cited}


def run_question(question: Question, llm: ChatModel, search: ReportSearch, variant: str,
                 spec: ModelSpec) -> dict:
    state = ask(question.question, llm, search, variant)
    answer: Answer = state["answer"]
    calls = state.get("calls", [])
    prompt_tokens = sum(call.prompt_tokens for call in calls)
    completion_tokens = sum(call.completion_tokens for call in calls)
    return {
        "id": question.id,
        "question": question.question,
        **score(question, answer),
        "answered": answer.answered,
        "answer": answer.text,
        "claims_accepted": len(answer.claims),
        "claims_rejected": [{"text": claim.text, "reason": claim.reason} for claim in answer.rejected],
        "revisions": state.get("revisions", 0),
        "searches": len(state.get("searches_run", [])),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_seconds": round(sum(call.seconds for call in calls), 2),
        "cost_usd": round(spec.cost(prompt_tokens, completion_tokens), 6),
        "calls": [call_as_dict(call) for call in calls],
    }


def summarise(model: str, variant: str, rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["cited"] is not None]
    count = len(rows) or 1
    return {
        "model": model,
        "variant": variant,
        "questions": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": round(sum(row["correct"] for row in rows) / count, 3),
        "cited": sum(bool(row["cited"]) for row in answerable),
        "answerable": len(answerable),
        "claims_rejected": sum(len(row["claims_rejected"]) for row in rows),
        "revisions": sum(row["revisions"] for row in rows),
        "tokens_per_question": round(
            sum(row["prompt_tokens"] + row["completion_tokens"] for row in rows) / count),
        "median_seconds": round(statistics.median(row["model_seconds"] for row in rows), 2) if rows else 0.0,
        "cost_per_100_questions_usd": round(100 * sum(row["cost_usd"] for row in rows) / count, 3),
    }


def result_path(model: str, variant: str, directory: Path = RESULTS_DIR) -> Path:
    return directory / f"{model.replace('/', '_')}__{variant}.json"


def evaluate(llm: ChatModel, search: ReportSearch, variant: str, spec: ModelSpec,
             questions: list[Question], on_row=None) -> dict:
    rows = []
    stopped = ""
    for question in questions:
        try:
            row = run_question(question, llm, search, variant, spec)
        except LLMError as exc:
            # usually the daily allowance of the free tier: keep what is done, the cache resumes it
            stopped = str(exc)
            break
        rows.append(row)
        if on_row:
            on_row(row)
    result = {"summary": summarise(llm.model, variant, rows), "questions": rows}
    if stopped:
        result["stopped"] = stopped
    return result


def save(result: dict, directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = result_path(result["summary"]["model"], result["summary"]["variant"], directory)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def comparison_table(directory: Path = RESULTS_DIR) -> str:
    """All saved runs as one Markdown table."""
    summaries = [json.loads(path.read_text(encoding="utf-8"))["summary"]
                 for path in sorted(directory.glob("*.json"))]
    lines = [
        "| Model | Prompt | Correct | Right page cited | Claims rejected | Tokens / question "
        "| Median model time | Cost per 100 questions |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda s: (s["variant"] != "baseline", s["model"], s["variant"])):
        lines.append(
            f"| {s['model']} | {s['variant']} | {s['correct']}/{s['questions']} "
            f"| {s['cited']}/{s['answerable']} "
            f"| {s['claims_rejected']} | {s['tokens_per_question']:,} | {s['median_seconds']:.1f} s "
            f"| ${s['cost_per_100_questions_usd']:.3f} |"
        )
    return "\n".join(lines)
