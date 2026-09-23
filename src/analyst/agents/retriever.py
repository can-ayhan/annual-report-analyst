"""Retriever: runs the planned searches, judges what came back and searches again if something is missing.

It sees only the start of each passage. The full text goes to the analyst.

Two things are done without asking a model, because both went wrong in real runs:
a company name is matched to the name in the library ("InPost Group" is "InPost"),
and the question itself is always one of the searches, since a planner's short
query can miss the page that the user's own words find.
"""

from __future__ import annotations

from analyst.agents import prompts
from analyst.agents.planner import catalogue_listing
from analyst.agents.state import Passage, Review, Search, State
from analyst.llm import ChatModel
from analyst.search import Hit, ReportSearch

PER_SEARCH = 4          # hits taken from each search
MAX_PASSAGES = 6        # passages handed to the analyst
MAX_PREVIEWS = 10       # passages the retriever looks at when judging the results
PREVIEW_CHARS = 240


def describe(search: Search) -> str:
    scope = ", ".join(str(part) for part in (search.company, search.fiscal_year) if part) or "all reports"
    return f'"{search.query}" in {scope}'


def resolve_company(name: str | None, catalogue: list[dict]) -> str | None:
    """The company as the library spells it. A name the library does not have is kept,
    so that the search comes back empty instead of answering about someone else."""
    if not name:
        return None
    wanted = name.strip().lower()
    companies = list(dict.fromkeys(report["company"] for report in catalogue))
    for company in companies:
        if company.lower() == wanted:
            return company
    for company in companies:
        if company.lower() in wanted or wanted in company.lower():
            return company
    return name


def with_question(question: str, searches: list[Search]) -> list[Search]:
    """The planned searches plus the question itself, once for every report they are aimed at."""
    scopes = dict.fromkeys((search.company, search.fiscal_year) for search in searches)
    return searches + [Search(query=question, company=company, fiscal_year=year) for company, year in scopes]


def interleave(results: list[list[Hit]], limit: int) -> list[Hit]:
    """Best hit of every search first, then the second best of each, and so on, without repeats."""
    chosen: dict[int, Hit] = {}
    for rank in range(max((len(hits) for hits in results), default=0)):
        for hits in results:
            if rank < len(hits) and len(chosen) < limit:
                chosen.setdefault(hits[rank].passage_id, hits[rank])
    return list(chosen.values())


def make_retriever(llm: ChatModel, search: ReportSearch):
    def retriever(state: State) -> dict:
        catalogue = state["catalogue"]

        def scoped(searches: list[Search]) -> list[Search]:
            return [s.model_copy(update={"company": resolve_company(s.company, catalogue)}) for s in searches]

        def run(searches: list[Search]) -> list[list[Hit]]:
            return [search.search(s.query, company=s.company, fiscal_year=s.fiscal_year, limit=PER_SEARCH)
                    for s in searches]

        searches = with_question(state["question"], scoped(list(state["plan"].searches)))
        results = run(searches)
        calls = []
        if any(results):
            previews = "\n".join(
                f"- {hit.citation}: {' '.join(hit.text.split())[:PREVIEW_CHARS]}"
                for hit in interleave(results, limit=MAX_PREVIEWS)
            )
            user = (f"Reports in the library:\n{catalogue_listing(catalogue)}\n\n"
                    f"Question: {state['question']}\n\nSearches run:\n"
                    + "\n".join(f"- {describe(s)}" for s in searches)
                    + f"\n\nPassages found:\n{previews}")
            review, call = llm.structured("retriever", prompts.RETRIEVER, user, Review)
            calls.append(call)
            if not review.enough and review.follow_up:
                more = scoped(review.follow_up[:2])
                searches += more
                results += run(more)
        passages = [
            Passage(label=f"P{number}", passage_id=hit.passage_id, report_id=hit.report_id,
                    company=hit.company, fiscal_year=hit.fiscal_year, page_no=hit.page_no, text=hit.text)
            for number, hit in enumerate(interleave(results, limit=MAX_PASSAGES), start=1)
        ]
        return {"passages": passages, "calls": calls, "searches_run": searches}

    return retriever
