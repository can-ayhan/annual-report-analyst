"""Verifier: a claim reaches the answer only if a retrieved passage backs it.

Three checks, the first two without a model:

1. the claim cites at least one passage, and only passages that were retrieved;
2. every figure in the claim occurs in the passages it cites;
3. a model reads the claim against those passages and, to accept it, must quote the
   words that back it. A quote that is not in the passage does not count.

A claim that fails any of them is rejected, with the reason.
"""

from __future__ import annotations

import re

from analyst.agents import prompts
from analyst.agents.state import Checked, Claim, Passage, State, Verification
from analyst.llm import ChatModel

MIN_QUOTE = 10          # characters; shorter "quotes" prove nothing

_SEPARATOR_IN_NUMBER = re.compile(r"(?<=\d)[ ,\u00a0\u202f](?=\d)")
_COMMA_IN_NUMBER = re.compile(r"(?<=\d),(?=\d)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_YEAR = re.compile(r"(?:19|20)\d\d")


def squash_numbers(text: str) -> str:
    """Remove thousands separators so that 3,033 and 3 033 and 3033 read the same."""
    previous = None
    while previous != text:                    # "1 234 567" needs more than one pass
        previous, text = text, _SEPARATOR_IN_NUMBER.sub("", text)
    return text


def figures(text: str) -> list[str]:
    """The numbers in a claim that have to be found in its sources. Years are left to the model."""
    return [n for n in _NUMBER.findall(squash_numbers(text)) if not _YEAR.fullmatch(n)]


def normalise(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").replace("“", '"').replace("”", '"').split())


def words(text: str) -> list[str]:
    """Lower-case words and numbers. 3,033 becomes one word; 866 989 stays two, as it is in the tables."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", _COMMA_IN_NUMBER.sub("", normalise(text)))


def quote_parts(quote: str) -> list[list[str]]:
    """The quote as runs of words; an ellipsis in the quote separates the runs."""
    parts = [words(part) for part in re.split(r"\.{3}|…", quote)]
    return [part for part in parts if part]


def long_enough(quote: str) -> bool:
    wanted = [word for part in quote_parts(quote) for word in part]
    return len(wanted) >= 2 and sum(len(word) for word in wanted) >= MIN_QUOTE


def quote_found(quote: str, sources: list[Passage]) -> bool:
    """True if the words of the quote occur in the sources, in the same order and close together.

    Close together rather than adjacent because of tables: a row reads
    "Total headcount 49,887 55,856" and an honest quote for the second column is
    "Total headcount 55,856". Where the quote marks an omission with an ellipsis,
    each part is looked for on its own. Words that are not there, or that are there
    in another order, do not pass.
    """
    if not long_enough(quote):
        return False
    have = [words(source.text) for source in sources]
    return all(any(_run_found(part, text) for text in have) for part in quote_parts(quote))


def _run_found(wanted: list[str], have: list[str]) -> bool:
    allowed_span = 2 * len(wanted) + 4
    for start, word in enumerate(have):
        if word != wanted[0]:
            continue
        position = start
        for needed in wanted[1:]:
            try:
                position = have.index(needed, position + 1, start + allowed_span)
            except ValueError:
                break
        else:
            return True
    return False


def problem_without_model(claim: Claim, by_label: dict[str, Passage]) -> str:
    """Checks 1 and 2. Returns the reason for rejecting the claim, or an empty string."""
    if not claim.sources:
        return "no passage cited"
    unknown = [label for label in claim.sources if label not in by_label]
    if unknown:
        return f"cites {', '.join(unknown)}, which was not retrieved"
    cited = squash_numbers(" ".join(by_label[label].text for label in claim.sources))
    missing = [number for number in figures(claim.text) if number not in cited]
    if missing:
        return f"the figure {', '.join(missing)} is not in {', '.join(claim.sources)} as written"
    return ""


def make_verifier(llm: ChatModel):
    def verifier(state: State) -> dict:
        draft = state["draft"]
        by_label = {passage.label: passage for passage in state.get("passages") or []}
        checked: list[Checked] = []
        to_judge: list[int] = []
        for position, claim in enumerate(draft.claims):
            reason = problem_without_model(claim, by_label)
            checked.append(Checked(text=claim.text, sources=claim.sources, supported=False, reason=reason))
            if not reason:
                to_judge.append(position)

        calls = []
        if to_judge:
            labels = dict.fromkeys(label for position in to_judge for label in draft.claims[position].sources)
            claims = "\n".join(
                f"{number}. {draft.claims[position].text} (cites {', '.join(draft.claims[position].sources)})"
                for number, position in enumerate(to_judge, start=1)
            )
            sources = "\n\n".join(
                f"[{label}] {by_label[label].citation}\n{by_label[label].text}" for label in labels)
            user = f"Question: {state['question']}\n\nClaims:\n{claims}\n\nPassages:\n{sources}"
            verification, call = llm.structured("verifier", prompts.VERIFIER, user, Verification)
            calls.append(call)
            verdicts = {verdict.claim: verdict for verdict in verification.verdicts}
            for number, position in enumerate(to_judge, start=1):
                claim = draft.claims[position]
                verdict = verdicts.get(number)
                if verdict is None:
                    checked[position].reason = "the verifier gave no verdict"
                elif not verdict.supported:
                    checked[position].reason = verdict.reason or "not stated in the cited passages"
                elif not long_enough(verdict.quote):
                    checked[position].reason = (
                        f"accepted on the strength of {verdict.quote[:80]!r} alone, "
                        "which is too short to show what the figure refers to")
                elif not quote_found(verdict.quote, [by_label[label] for label in claim.sources]):
                    checked[position].reason = (
                        f"accepted on the strength of {verdict.quote[:80]!r}, which is not in the passage")
                else:
                    checked[position].supported = True
                    checked[position].quote = verdict.quote
        return {"checked": checked, "calls": calls}

    return verifier
