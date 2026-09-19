import json
from types import SimpleNamespace

import groq
import httpx
import pytest

from analyst import llm as llm_module
from analyst.agents.state import Plan, Verification
from analyst.llm import GroqChat, LLMError, tool_schema, totals

GOOD = json.dumps({"searches": [{"query": "sales revenue", "company": "CD PROJEKT", "fiscal_year": 2025}]})


def reply(arguments: str, prompt_tokens: int = 200, completion_tokens: int = 30):
    call = SimpleNamespace(function=SimpleNamespace(name="submit", arguments=arguments))
    message = SimpleNamespace(tool_calls=[call], content=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def rate_limited(retry_after: str):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": retry_after}, request=request)
    return groq.RateLimitError("rate limit reached", response=response, body=None)


class FakeClient:
    """Plays back replies; an exception in the list is raised instead of returned."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    return slept


# -- the schema that is sent ---------------------------------------------------------------------------

def test_schema_has_no_references_titles_or_defaults():
    text = json.dumps(tool_schema(Verification))
    assert "$ref" not in text and "$defs" not in text and '"title"' not in text and '"default"' not in text


def test_optional_fields_are_a_type_list_and_everything_is_required():
    search = tool_schema(Plan)["properties"]["searches"]["items"]
    assert search["properties"]["company"]["type"] == ["string", "null"]
    assert search["properties"]["fiscal_year"]["type"] == ["integer", "null"]
    assert search["required"] == ["query", "company", "fiscal_year"]
    assert "anyOf" not in json.dumps(search)


# -- one call ------------------------------------------------------------------------------------------

def test_a_valid_reply_comes_back_as_the_model_with_its_cost():
    client = FakeClient(reply(GOOD))
    plan, call = GroqChat("some-model", client=client).structured("planner", "system", "user", Plan)
    assert plan.searches[0].company == "CD PROJEKT"
    assert (call.agent, call.model, call.prompt_tokens, call.completion_tokens, call.cached) == (
        "planner", "some-model", 200, 30, False)
    request = client.requests[0]
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit"}}
    assert request["temperature"] == 0 and request["model"] == "some-model"


def test_model_options_are_passed_through():
    client = FakeClient(reply(GOOD))
    GroqChat("some-model", client=client, options={"reasoning_effort": "low"}).structured("planner", "s", "u", Plan)
    assert client.requests[0]["reasoning_effort"] == "low"


def test_a_reply_that_does_not_fit_is_sent_back_once():
    client = FakeClient(reply('{"searches": []}'), reply(GOOD))           # the plan needs at least one search
    plan, call = GroqChat("some-model", client=client).structured("planner", "system", "user", Plan)
    assert len(plan.searches) == 1
    assert call.prompt_tokens == 400                                       # both requests are paid for
    assert "was rejected" in client.requests[1]["messages"][-1]["content"]


def test_two_bad_replies_are_an_error():
    client = FakeClient(reply('{"searches": []}'), reply("not json"))
    with pytest.raises(LLMError, match="did not return a valid Plan"):
        GroqChat("some-model", client=client).structured("planner", "system", "user", Plan)


# -- rate limits ---------------------------------------------------------------------------------------

def test_a_short_rate_limit_is_waited_out(no_sleep):
    client = FakeClient(rate_limited("7"), reply(GOOD))
    plan, call = GroqChat("some-model", client=client).structured("planner", "system", "user", Plan)
    assert plan.searches and no_sleep == [7.5]
    assert call.seconds < 1                              # the wait is not counted as model time


def test_the_daily_allowance_is_reported_not_waited_for(no_sleep):
    client = FakeClient(rate_limited("5400"))
    with pytest.raises(LLMError, match="retry after 5400 s"):
        GroqChat("some-model", client=client).structured("planner", "system", "user", Plan)
    assert no_sleep == []


# -- cache ---------------------------------------------------------------------------------------------

def test_an_identical_request_is_replayed_from_the_cache(tmp_path):
    client = FakeClient(reply(GOOD))
    chat = GroqChat("some-model", client=client, cache_dir=tmp_path)
    first, paid = chat.structured("planner", "system", "user", Plan)
    again, replayed = chat.structured("planner", "system", "user", Plan)
    assert again == first and len(client.requests) == 1
    assert replayed.cached and not paid.cached
    assert (replayed.prompt_tokens, replayed.seconds) == (paid.prompt_tokens, paid.seconds)


def test_another_model_or_prompt_is_not_a_cache_hit(tmp_path):
    client = FakeClient(reply(GOOD), reply(GOOD), reply(GOOD))
    GroqChat("model-a", client=client, cache_dir=tmp_path).structured("planner", "system", "user", Plan)
    GroqChat("model-b", client=client, cache_dir=tmp_path).structured("planner", "system", "user", Plan)
    GroqChat("model-a", client=client, cache_dir=tmp_path).structured("planner", "system", "other", Plan)
    assert len(client.requests) == 3


def test_totals_add_up_by_agent():
    client = FakeClient(reply(GOOD, 100, 10), reply(GOOD, 300, 40))
    chat = GroqChat("some-model", client=client)
    calls = [chat.structured("planner", "s", "u", Plan)[1], chat.structured("analyst", "s", "u2", Plan)[1]]
    summary = totals(calls)
    assert (summary["calls"], summary["prompt_tokens"], summary["completion_tokens"]) == (2, 400, 50)
    assert summary["by_agent"] == {"planner": 110, "analyst": 340}
