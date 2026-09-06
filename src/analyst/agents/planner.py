"""Planner: turns a question into the searches that will answer it."""

from __future__ import annotations

from analyst.agents import prompts
from analyst.agents.state import Plan, State
from analyst.llm import ChatModel


def catalogue_listing(catalogue: list[dict]) -> str:
    return "\n".join(f"- {r['company']}, fiscal year {r['fiscal_year']}: {r['title']}" for r in catalogue)


def make_planner(llm: ChatModel):
    def planner(state: State) -> dict:
        user = (f"Reports in the library:\n{catalogue_listing(state['catalogue'])}\n\n"
                f"Question: {state['question']}")
        plan, call = llm.structured("planner", prompts.PLANNER, user, Plan)
        return {"plan": plan, "calls": [call]}

    return planner
