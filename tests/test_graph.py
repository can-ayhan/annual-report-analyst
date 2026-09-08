"""The whole graph on the sample report, with the model's replies scripted."""

from analyst.agents.state import (
    Claim,
    Draft,
    Plan,
    Review,
    Search,
    Verdict,
    Verification,
)
from analyst.graph import ask
from fakes import ScriptedChat

PLAN = Plan(searches=[Search(query="number of bakeries at the end of 2025", company="Baltic Mills", fiscal_year=2025)])
ENOUGH = Review(enough=True)


def label_of(state: dict, page: int) -> str:
    return next(p.label for p in state["passages"] if p.page_no == page)


def test_a_question_goes_through_all_four_agents(search):
    # the analyst and the verifier need the passage labels, which are only known after retrieval,
    # so their replies are built from a first run of planner and retriever
    probe = ask("How many bakeries?", ScriptedChat(planner=[PLAN], retriever=[ENOUGH],
                                                    analyst=[Draft(cannot_answer=True)]), search)
    p = label_of(probe, 1)
    llm = ScriptedChat(
        planner=[PLAN], retriever=[ENOUGH],
        analyst=[Draft(claims=[Claim(text="Baltic Mills operated 412 bakeries at 31 December 2025.", sources=[p])])],
        verifier=[Verification(verdicts=[Verdict(claim=1, supported=True, quote="the Group operated 412 bakeries")])],
    )
    state = ask("How many bakeries did Baltic Mills operate at the end of 2025?", llm, search)

    answer = state["answer"]
    assert answer.answered
    assert answer.text == "Baltic Mills operated 412 bakeries at 31 December 2025. [Baltic Mills 2025, p. 1]"
    assert [(c.report_id, c.page_no) for c in answer.claims[0].citations] == [("baltic-2025", 1)]
    assert [agent for agent, _ in llm.asked] == ["planner", "retriever", "analyst", "verifier"]
    assert state["usage"]["calls"] == 4 and state["usage"]["prompt_tokens"] == 400
    assert state.get("revisions", 0) == 0


def test_the_planner_sees_the_catalogue_and_the_analyst_sees_full_passages(search):
    llm = ScriptedChat(planner=[PLAN], retriever=[ENOUGH], analyst=[Draft(cannot_answer=True)])
    ask("How many bakeries?", llm, search)
    prompts = dict(llm.asked)
    assert "- Baltic Mills, fiscal year 2025: Baltic Mills annual report 2025" in prompts["planner"]
    assert "Baltic Mills 2025, p. 1:" in prompts["retriever"]             # previews carry the citation
    assert "compared with 390 a year earlier" in prompts["analyst"]


def test_a_rejected_claim_gets_one_revision(search):
    probe = ask("x", ScriptedChat(planner=[PLAN], retriever=[ENOUGH], analyst=[Draft(cannot_answer=True)]), search)
    p = label_of(probe, 1)
    llm = ScriptedChat(
        planner=[PLAN], retriever=[ENOUGH],
        analyst=[
            Draft(claims=[Claim(text="Baltic Mills operated 421 bakeries.", sources=[p])]),      # wrong figure
            Draft(claims=[Claim(text="Baltic Mills operated 412 bakeries.", sources=[p])]),
        ],
        verifier=[Verification(verdicts=[Verdict(claim=1, supported=True, quote="the Group operated 412 bakeries")])],
    )
    state = ask("How many bakeries?", llm, search)
    assert state["revisions"] == 1
    assert state["answer"].text.startswith("Baltic Mills operated 412 bakeries.")
    assert llm.calls_from("analyst") == 2 and llm.calls_from("verifier") == 1     # the first draft never needed the model
    revision_prompt = [user for agent, user in llm.asked if agent == "analyst"][1]
    assert "REJECTED: the figure 421 is not in" in revision_prompt


def test_claims_still_rejected_after_the_revision_are_left_out(search):
    probe = ask("x", ScriptedChat(planner=[PLAN], retriever=[ENOUGH], analyst=[Draft(cannot_answer=True)]), search)
    p = label_of(probe, 1)
    wrong = Draft(claims=[Claim(text="Baltic Mills operated 421 bakeries.", sources=[p])])
    llm = ScriptedChat(planner=[PLAN], retriever=[ENOUGH], analyst=[wrong, wrong])
    state = ask("How many bakeries?", llm, search)
    answer = state["answer"]
    assert not answer.answered
    assert answer.text == "The indexed reports do not contain an answer to this."
    assert [claim.reason for claim in answer.rejected] == [f"the figure 421 is not in {p} as written"]
    assert llm.calls_from("analyst") == 2                # one revision, not a loop


def test_nothing_found_means_no_analyst_and_no_verifier(search):
    plan = Plan(searches=[Search(query="gross merchandise value", company="Allegro", fiscal_year=2025)])
    llm = ScriptedChat(planner=[plan])
    state = ask("What was Allegro's GMV in 2025?", llm, search)
    assert state["passages"] == []
    assert not state["answer"].answered
    assert [agent for agent, _ in llm.asked] == ["planner"]


def test_the_retriever_can_ask_for_more(search):
    first = Plan(searches=[Search(query="number of bakeries", company="Baltic Mills", fiscal_year=2025)])
    more = Review(enough=False, follow_up=[Search(query="dividend per share", company="Baltic Mills")])
    llm = ScriptedChat(planner=[first], retriever=[more], analyst=[Draft(cannot_answer=True)])
    state = ask("How many bakeries, and what dividend?", llm, search)
    assert [s.query for s in state["searches_run"]] == ["number of bakeries", "dividend per share"]
    assert {p.page_no for p in state["passages"]} >= {1, 3}
