import json

import pytest

from analyst import evaluation
from analyst.agents.state import Answer, AnswerClaim, Checked, Citation
from analyst.evaluation import ModelSpec, Question, load_models, load_questions, score, summarise
from analyst.reports import load_reports


def answer(*claims: tuple[str, str, int], rejected: int = 0) -> Answer:
    return Answer(
        question="?",
        claims=[AnswerClaim(text=text, citations=[Citation(report_id=report, company="X", fiscal_year=2025, page_no=page)])
                for text, report, page in claims],
        rejected=[Checked(text="r", sources=[], supported=False, reason="no")] * rejected,
        answered=bool(claims),
    )


STORES = Question(id="stores", question="How many stores?", expect=[["3,033"]], pages={"dino-2025": [8, 10]})
TWO_FIGURES = Question(id="fcf", question="FCF in both years?", expect=[["84.1"], ["934.5"]], pages={"inpost-2025": [38]})
TWO_REPORTS = Question(id="cross", question="Who has more?", expect=[["55,856"], ["13,419"]],
                       pages={"dino-2025": [82], "inpost-2025": [190, 287]})
MISSING = Question(id="missing", question="Allegro GMV?", no_answer=True)


def test_right_figure_on_the_right_page():
    assert score(STORES, answer(("Dino had 3,033 stores.", "dino-2025", 10))) == {"correct": True, "cited": True}


def test_separators_do_not_matter():
    assert score(STORES, answer(("Dino had 3 033 stores.", "dino-2025", 8)))["correct"]
    assert score(STORES, answer(("Dino had 3033 stores.", "dino-2025", 8)))["correct"]


def test_right_figure_from_the_wrong_page_is_correct_but_not_cited():
    assert score(STORES, answer(("Dino had 3,033 stores.", "dino-2025", 55))) == {"correct": True, "cited": False}


def test_wrong_figure():
    assert score(STORES, answer(("Dino had 2,688 stores.", "dino-2025", 10))) == {"correct": False, "cited": True}


def test_every_expected_figure_is_needed():
    assert not score(TWO_FIGURES, answer(("FCF was PLN 84.1 m in 2025.", "inpost-2025", 38)))["correct"]
    both = answer(("FCF was PLN 84.1 m in 2025.", "inpost-2025", 38), ("It was PLN 934.5 m in 2024.", "inpost-2025", 38))
    assert score(TWO_FIGURES, both) == {"correct": True, "cited": True}


def test_every_report_needs_a_cited_page():
    one_sided = answer(("Dino had 55,856 employees and InPost 13,419.", "dino-2025", 82))
    assert score(TWO_REPORTS, one_sided) == {"correct": True, "cited": False}
    both = answer(("Dino had 55,856 employees.", "dino-2025", 82), ("InPost had 13,419.", "inpost-2025", 287))
    assert score(TWO_REPORTS, both) == {"correct": True, "cited": True}


def test_no_answer_is_right_only_when_nothing_is_claimed():
    assert score(MISSING, answer()) == {"correct": True, "cited": None}
    assert score(MISSING, answer(("Allegro's GMV was PLN 70 billion.", "dino-2025", 3))) == {"correct": False, "cited": None}


def test_an_empty_answer_to_an_answerable_question_is_wrong():
    assert score(STORES, answer(rejected=2)) == {"correct": False, "cited": False}


def test_cost_uses_list_prices_per_million_tokens():
    spec = ModelSpec("m", input_per_mtok=0.15, output_per_mtok=0.60)
    assert spec.cost(1_000_000, 0) == pytest.approx(0.15)
    assert spec.cost(4000, 500) == pytest.approx(0.0006 + 0.0003)


def row(correct, cited, tokens=5000, seconds=2.0, rejected=0, revisions=0, cost=0.001):
    return {"correct": correct, "cited": cited, "prompt_tokens": tokens - 500, "completion_tokens": 500,
            "model_seconds": seconds, "claims_rejected": [{}] * rejected, "revisions": revisions, "cost_usd": cost}


def test_summary_counts_and_averages():
    rows = [row(True, True, seconds=1.0), row(True, False, seconds=3.0, rejected=1, revisions=1),
            row(False, False, tokens=7000, seconds=9.0), row(True, None, tokens=3000, seconds=2.0)]
    summary = summarise("m", "baseline", rows)
    assert (summary["correct"], summary["questions"], summary["accuracy"]) == (3, 4, 0.75)
    assert (summary["cited"], summary["answerable"]) == (1, 3)           # the unanswerable one has no page to cite
    assert summary["tokens_per_question"] == 5000
    assert summary["median_seconds"] == 2.5
    assert (summary["claims_rejected"], summary["revisions"]) == (1, 1)
    assert summary["cost_per_100_questions_usd"] == pytest.approx(0.1)


def test_saved_runs_become_one_table(tmp_path):
    for model, variant, correct in (("big", "baseline", 17), ("small", "baseline", 15), ("small", "minimal", 12)):
        rows = [row(True, True)] * correct + [row(False, False)] * (18 - correct)
        evaluation.save({"summary": summarise(model, variant, rows), "questions": rows}, tmp_path)
    assert json.loads((tmp_path / "small__minimal.json").read_text(encoding="utf-8"))["summary"]["correct"] == 12
    table = evaluation.comparison_table(tmp_path).splitlines()
    assert table[0].startswith("| Model | Prompt | Correct")
    assert [line.split("|")[1:4] for line in table[2:]] == [
        [" big ", " baseline ", " 17/18 "], [" small ", " baseline ", " 15/18 "], [" small ", " minimal ", " 12/18 "]]


# -- the question set itself ---------------------------------------------------------------------------

def test_questions_are_well_formed():
    questions = load_questions()
    reports = {report.report_id: report.pages for report in load_reports()}
    assert len({q.id for q in questions}) == len(questions) >= 15
    for q in questions:
        if q.no_answer:
            assert not q.expect and not q.pages
            continue
        assert q.expect and all(group for group in q.expect), q.id
        assert q.pages and set(q.pages) <= set(reports), q.id
        for report_id, pages in q.pages.items():
            assert pages and all(1 <= page <= reports[report_id] for page in pages), q.id


def test_the_set_has_questions_the_library_cannot_answer_and_ones_that_need_two_reports():
    questions = load_questions()
    assert sum(q.no_answer for q in questions) >= 2
    assert any(len(q.pages) > 1 for q in questions)


def test_compared_models_have_prices():
    models = load_models()
    assert len(models) == 3
    for spec in models.values():
        assert 0 < spec.input_per_mtok < spec.output_per_mtok
