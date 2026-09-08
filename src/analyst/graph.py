"""The agents as a LangGraph graph.

    planner -> retriever -> analyst -> verifier -> finish
                               ^           |
                               +-- revise -+      (once, if a claim was rejected)

Only claims the verifier accepted reach the answer. If none is left, the answer
is that the reports do not say.
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from analyst.agents.analyst import make_analyst
from analyst.agents.planner import make_planner
from analyst.agents.retriever import make_retriever
from analyst.agents.state import Answer, AnswerClaim, Citation, State
from analyst.agents.verifier import make_verifier
from analyst.llm import ChatModel, totals
from analyst.search import ReportSearch

MAX_REVISIONS = 1


def after_verification(state: State) -> str:
    rejected = any(not claim.supported for claim in state["checked"])
    return "revise" if rejected and state.get("revisions", 0) < MAX_REVISIONS else "finish"


def finish(state: State) -> dict:
    by_label = {passage.label: passage for passage in state.get("passages") or []}
    claims = [
        AnswerClaim(text=claim.text, citations=[
            Citation(report_id=p.report_id, company=p.company, fiscal_year=p.fiscal_year, page_no=p.page_no)
            for p in (by_label[label] for label in claim.sources)
        ])
        for claim in state.get("checked", []) if claim.supported
    ]
    rejected = [claim for claim in state.get("checked", []) if not claim.supported]
    answer = Answer(question=state["question"], claims=claims, rejected=rejected, answered=bool(claims))
    return {"answer": answer}


def build_graph(llm: ChatModel, search: ReportSearch, variant: str = "baseline"):
    graph = StateGraph(State)
    graph.add_node("planner", make_planner(llm))
    graph.add_node("retriever", make_retriever(llm, search))
    graph.add_node("analyst", make_analyst(llm, variant))
    graph.add_node("verifier", make_verifier(llm))
    graph.add_node("finish", finish)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "verifier")
    graph.add_conditional_edges("verifier", after_verification, {"revise": "analyst", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()


def ask(question: str, llm: ChatModel, search: ReportSearch, variant: str = "baseline") -> dict:
    """Run one question through the graph. Returns the final state plus timing and token totals."""
    started = time.perf_counter()
    state = build_graph(llm, search, variant).invoke({"question": question, "catalogue": search.reports()})
    state["seconds"] = round(time.perf_counter() - started, 2)
    state["usage"] = totals(state.get("calls", []))
    return state
