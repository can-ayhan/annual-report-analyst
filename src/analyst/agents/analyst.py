"""Analyst: writes the answer as claims, each tied to the passages that state it."""

from __future__ import annotations

from analyst.agents import prompts
from analyst.agents.state import Checked, Draft, Passage, State
from analyst.llm import ChatModel


def passages_block(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.label}] {p.citation}\n{p.text}" for p in passages)


def feedback(checked: list[Checked]) -> str:
    lines = []
    for number, claim in enumerate(checked, start=1):
        verdict = "accepted" if claim.supported else f"REJECTED: {claim.reason}"
        lines.append(f"{number}. {claim.text} {claim.sources} -> {verdict}")
    return "\n".join(lines)


def make_analyst(llm: ChatModel, variant: str = "baseline"):
    system = prompts.ANALYST[variant]

    def analyst(state: State) -> dict:
        passages = state.get("passages") or []
        if not passages:
            return {"draft": Draft(cannot_answer=True)}
        user = f"Question: {state['question']}\n\nPassages:\n{passages_block(passages)}"
        revising = bool(state.get("checked"))
        if revising:
            user += f"\n{prompts.REVISION}\nYour previous claims:\n{feedback(state['checked'])}"
        draft, call = llm.structured("analyst", system, user, Draft)
        update = {"draft": draft, "calls": [call]}
        if revising:
            update["revisions"] = state.get("revisions", 0) + 1
        return update

    return analyst
