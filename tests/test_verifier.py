import pytest

from analyst.agents.state import Claim, Draft, Passage, Verdict, Verification
from analyst.agents.verifier import (
    figures,
    make_verifier,
    problem_without_model,
    quote_found,
    squash_numbers,
)
from fakes import ScriptedChat


def passage(label: str, text: str, page: int = 1) -> Passage:
    return Passage(label=label, passage_id=page, report_id="baltic-2025", company="Baltic Mills",
                   fiscal_year=2025, page_no=page, text=text)


TABLE = passage("P1", "Employees by gender\n2024 2025\nMale 6,525 7,345\nFemale 43,362 48,511\nTotal headcount 49,887 55,856")
PROSE = passage("P2", "As at 31 December 2025 the Group operated 412 bakeries, compared with 390 a year earlier.")
THOUSANDS = passage("P3", "(in PLN thousands)\nSales revenue\n866 989\n798 372\nNet profit\n520 869\n442 991")
BY_LABEL = {p.label: p for p in (TABLE, PROSE, THOUSANDS)}


# -- figures ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "expected"), [
    ("3,033 stores", "3033 stores"),
    ("866 989 thousand", "866989 thousand"),
    ("1 234 567 and 2,5", "1234567 and 25"),
    ("page 12, line 3", "page 12, line 3"),         # a comma followed by a space is punctuation
])
def test_thousands_separators_are_removed(text, expected):
    assert squash_numbers(text) == expected


def test_years_are_not_treated_as_figures():
    assert figures("In 2025 the Group had 55,856 employees, up from 49,887 in 2024.") == ["55856", "49887"]
    assert figures("The margin was 60.1% in 2025.") == ["60.1"]


# -- checks without a model -------------------------------------------------------------------------

def test_a_claim_must_cite_something():
    assert problem_without_model(Claim(text="The Group has 412 bakeries.", sources=[]), BY_LABEL) == "no passage cited"


def test_a_claim_may_only_cite_what_was_retrieved():
    reason = problem_without_model(Claim(text="The Group has 412 bakeries.", sources=["P9"]), BY_LABEL)
    assert reason == "cites P9, which was not retrieved"


def test_a_figure_must_be_in_the_cited_passage():
    assert problem_without_model(Claim(text="The Group operated 412 bakeries in 2025.", sources=["P2"]), BY_LABEL) == ""
    reason = problem_without_model(Claim(text="The Group operated 421 bakeries in 2025.", sources=["P2"]), BY_LABEL)
    assert reason == "the figure 421 is not in P2 as written"


def test_a_figure_from_another_passage_does_not_count():
    reason = problem_without_model(Claim(text="The Group had 55,856 employees.", sources=["P2"]), BY_LABEL)
    assert "55856" in reason


def test_rounding_or_converting_a_figure_is_caught():
    # the table says 866 989 (thousands); "867 million" is not what the passage writes
    reason = problem_without_model(Claim(text="Sales revenue was PLN 867 million.", sources=["P3"]), BY_LABEL)
    assert reason == "the figure 867 is not in P3 as written"
    assert problem_without_model(Claim(text="Sales revenue was 866,989 thousand PLN.", sources=["P3"]), BY_LABEL) == ""


# -- the quote the verifier has to show --------------------------------------------------------------

@pytest.mark.parametrize("quote", [
    "the Group operated 412 bakeries",
    "The Group operated 412 bakeries, compared with 390",      # case and punctuation do not matter
    "the  Group operated\n412 bakeries",                         # nor does whitespace
])
def test_words_from_the_passage_are_accepted(quote):
    assert quote_found(quote, [PROSE])


def test_a_table_cell_can_be_quoted_with_its_row_label():
    assert quote_found("Total headcount 55,856", [TABLE])            # skips the 2024 column
    assert quote_found("Total headcount\n49,887 55,856", [TABLE])
    assert quote_found("Sales revenue 866 989", [THOUSANDS])


def test_an_ellipsis_separates_parts_that_are_each_looked_for():
    assert quote_found("Total headcount ... 2025 ... 55,856", [TABLE])        # row label, column, value
    assert quote_found("the Group operated 412 bakeries … a year earlier", [PROSE])
    assert not quote_found("Total headcount ... 2026 ... 55,856", [TABLE])    # one part is not in the passage
    assert quote_found("Total headcount 55,856 ... the Group operated 412 bakeries", [TABLE, PROSE])


@pytest.mark.parametrize("quote", [
    "the Group operated 500 bakeries",              # a word that is not there
    "bakeries operated Group the",                  # there, in another order
    "Male 55,856",                                  # there, in order, but far apart: different rows
    "412",                                          # too short to prove anything
    "",
])
def test_anything_else_is_not(quote):
    assert not quote_found(quote, [PROSE, TABLE])


def test_the_quote_must_come_from_a_cited_passage():
    assert not quote_found("the Group operated 412 bakeries", [TABLE])


# -- the agent ----------------------------------------------------------------------------------------

def verify(draft: Draft, llm: ScriptedChat) -> list:
    state = {"question": "How many bakeries?", "draft": draft, "passages": [TABLE, PROSE, THOUSANDS]}
    return make_verifier(llm)(state)["checked"]


def test_a_backed_claim_is_accepted_with_its_quote():
    llm = ScriptedChat(verifier=[Verification(verdicts=[
        Verdict(claim=1, supported=True, quote="the Group operated 412 bakeries")])])
    [checked] = verify(Draft(claims=[Claim(text="Baltic Mills operated 412 bakeries at the end of 2025.", sources=["P2"])]), llm)
    assert checked.supported and checked.quote == "the Group operated 412 bakeries"


def test_the_model_can_reject_a_claim():
    llm = ScriptedChat(verifier=[Verification(verdicts=[
        Verdict(claim=1, supported=False, reason="the passage gives the count for the Group, not for Poland")])])
    [checked] = verify(Draft(claims=[Claim(text="Poland has 412 bakeries.", sources=["P2"])]), llm)
    assert not checked.supported and "not for Poland" in checked.reason


def test_an_acceptance_without_real_words_is_overruled():
    llm = ScriptedChat(verifier=[Verification(verdicts=[
        Verdict(claim=1, supported=True, quote="Baltic Mills is the largest bakery group")])])
    [checked] = verify(Draft(claims=[Claim(text="Baltic Mills operated 412 bakeries.", sources=["P2"])]), llm)
    assert not checked.supported and "not in the passage" in checked.reason


def test_a_bare_figure_is_not_a_quote():
    llm = ScriptedChat(verifier=[Verification(verdicts=[Verdict(claim=1, supported=True, quote="412")])])
    [checked] = verify(Draft(claims=[Claim(text="Baltic Mills operated 412 bakeries.", sources=["P2"])]), llm)
    assert not checked.supported and "too short to show what the figure refers to" in checked.reason


def test_a_claim_without_a_verdict_is_rejected():
    llm = ScriptedChat(verifier=[Verification(verdicts=[
        Verdict(claim=1, supported=True, quote="the Group operated 412 bakeries")])])
    first, second = verify(Draft(claims=[
        Claim(text="Baltic Mills operated 412 bakeries.", sources=["P2"]),
        Claim(text="A year earlier it operated 390.", sources=["P2"]),
    ]), llm)
    assert first.supported and not second.supported
    assert second.reason == "the verifier gave no verdict"


def test_claims_that_fail_the_plain_checks_never_reach_the_model():
    llm = ScriptedChat()                                # any call would fail the test
    checked = verify(Draft(claims=[
        Claim(text="Baltic Mills operated 999 bakeries.", sources=["P2"]),
        Claim(text="It is growing.", sources=[]),
    ]), llm)
    assert [c.supported for c in checked] == [False, False]
    assert llm.asked == []


def test_the_model_sees_only_the_claims_still_standing_and_their_passages():
    llm = ScriptedChat(verifier=[Verification(verdicts=[
        Verdict(claim=1, supported=True, quote="Total headcount 49,887 55,856")])])
    checked = verify(Draft(claims=[
        Claim(text="Baltic Mills operated 999 bakeries.", sources=["P2"]),            # dropped before the model
        Claim(text="The Group had 55,856 employees in 2025.", sources=["P1"]),
    ]), llm)
    assert [c.supported for c in checked] == [False, True]
    [(_, user)] = llm.asked
    assert "1. The Group had 55,856 employees" in user and "999" not in user
    assert "[P1]" in user and "[P2]" not in user
