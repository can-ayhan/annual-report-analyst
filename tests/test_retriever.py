from analyst.agents.retriever import interleave, resolve_company, with_question
from analyst.agents.state import Search
from analyst.search import Hit

CATALOGUE = [{"company": "CD PROJEKT"}, {"company": "CD PROJEKT"}, {"company": "InPost"}, {"company": "Dino Polska"}]


def hit(passage_id: int) -> Hit:
    return Hit(passage_id=passage_id, report_id="r", company="C", fiscal_year=2025, page_no=passage_id,
               text=f"passage {passage_id}", score=1.0)


def test_company_names_resolve_to_the_library_spelling():
    assert resolve_company("InPost Group", CATALOGUE) == "InPost"
    assert resolve_company("inpost", CATALOGUE) == "InPost"
    assert resolve_company("CD Projekt S.A.", CATALOGUE) == "CD PROJEKT"
    assert resolve_company("Dino", CATALOGUE) == "Dino Polska"
    assert resolve_company(None, CATALOGUE) is None


def test_a_company_the_library_does_not_have_is_left_alone():
    # so that the search finds nothing, instead of answering about another company
    assert resolve_company("Allegro", CATALOGUE) == "Allegro"


def test_the_question_is_added_once_per_report_the_plan_is_aimed_at():
    plan = [Search(query="employees", company="Dino Polska", fiscal_year=2025),
            Search(query="headcount", company="Dino Polska", fiscal_year=2025),
            Search(query="employees", company="InPost", fiscal_year=2025)]
    searches = with_question("Who has more employees?", plan)
    assert searches[:3] == plan
    assert [(s.query, s.company) for s in searches[3:]] == [
        ("Who has more employees?", "Dino Polska"), ("Who has more employees?", "InPost")]


def test_results_are_taken_in_turns_without_repeats():
    first, second = [hit(1), hit(2), hit(3)], [hit(2), hit(9), hit(8)]
    assert [h.passage_id for h in interleave([first, second], limit=4)] == [1, 2, 9, 3]
    assert interleave([[], []], limit=4) == []
