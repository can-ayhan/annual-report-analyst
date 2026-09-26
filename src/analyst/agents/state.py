"""What the agents pass to each other."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from analyst.llm import Call

# -- planner ------------------------------------------------------------------------------------------


class Search(BaseModel):
    query: str = Field(description="Words likely to appear on the page: line items, units, names")
    company: str | None = Field(default=None, description="Company exactly as listed, or null for all")
    fiscal_year: int | None = Field(default=None, description="Fiscal year as listed, or null for all")


class Plan(BaseModel):
    searches: list[Search] = Field(min_length=1, max_length=4)


# -- retriever ----------------------------------------------------------------------------------------


class Passage(BaseModel):
    label: str                  # P1, P2, ... the name the analyst and the verifier use
    passage_id: int
    report_id: str
    company: str
    fiscal_year: int
    page_no: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.company} {self.fiscal_year}, p. {self.page_no}"


class Review(BaseModel):
    enough: bool = Field(description="True if the passages cover everything the question asks for")
    follow_up: list[Search] = Field(default_factory=list, max_length=2,
                                    description="Further searches for what is missing; empty if enough")


# -- analyst ------------------------------------------------------------------------------------------


class Claim(BaseModel):
    text: str = Field(description="One sentence that can be checked on its own")
    sources: list[str] = Field(description="Labels of the passages that state it, e.g. ['P2']")


class Draft(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    cannot_answer: bool = Field(default=False, description="True if the passages do not contain the answer")


class QuotedClaim(BaseModel):
    quote: str = Field(description="The sentence or table row copied from the passage, word for word")
    text: str = Field(description="The claim that the quote supports, one sentence")
    sources: list[str] = Field(description="Label of the passage the quote is from, e.g. ['P2']")


class QuotedDraft(BaseModel):
    claims: list[QuotedClaim] = Field(default_factory=list)
    cannot_answer: bool = Field(default=False, description="True if the passages do not contain the answer")

    def as_draft(self) -> Draft:
        return Draft(claims=[Claim(text=claim.text, sources=claim.sources) for claim in self.claims],
                     cannot_answer=self.cannot_answer)


# -- verifier -----------------------------------------------------------------------------------------


class Verdict(BaseModel):
    claim: int = Field(description="Number of the claim, starting at 1")
    supported: bool
    quote: str = Field(default="", description="Exact words from a cited passage that back the claim")
    reason: str = Field(default="", description="If not supported: what is missing or different")


class Verification(BaseModel):
    verdicts: list[Verdict]


class Checked(BaseModel):
    """A claim after verification."""
    text: str
    sources: list[str]
    supported: bool
    reason: str = ""            # why it was rejected
    quote: str = ""             # the words that back it


# -- result -------------------------------------------------------------------------------------------


class Citation(BaseModel):
    report_id: str
    company: str
    fiscal_year: int
    page_no: int


class AnswerClaim(BaseModel):
    text: str
    citations: list[Citation]


class Answer(BaseModel):
    question: str
    claims: list[AnswerClaim]
    rejected: list[Checked]
    answered: bool

    @property
    def text(self) -> str:
        if not self.answered:
            return "The indexed reports do not contain an answer to this."
        lines = []
        for claim in self.claims:
            pages = "; ".join(dict.fromkeys(
                f"{c.company} {c.fiscal_year}, p. {c.page_no}" for c in claim.citations))
            lines.append(f"{claim.text} [{pages}]")
        return "\n".join(lines)


class State(TypedDict, total=False):
    question: str
    catalogue: list[dict]
    plan: Plan
    passages: list[Passage]
    draft: Draft
    checked: list[Checked]
    revisions: int
    answer: Answer
    calls: Annotated[list[Call], operator.add]
    searches_run: Annotated[list[Search], operator.add]
