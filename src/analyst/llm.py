"""Chat calls that come back as validated objects.

Every agent asks for a Pydantic model. The request forces one tool call whose
arguments are that model's JSON schema, the reply is validated, and a reply that
does not validate is sent back once with the error. Each call is recorded with
its tokens and its duration, which is what the evaluation adds up.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_OUTPUT_TOKENS = 2000
LONGEST_WAIT = 90           # seconds; a rate limit that asks for more is the daily allowance


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Call:
    """One request: who asked, what it cost, how long it took."""
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    cached: bool = False


class ChatModel(Protocol):
    model: str

    def structured(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Call]: ...


def tool_schema(schema: type[BaseModel]) -> dict:
    """The JSON schema of a model with $ref inlined and titles dropped: smaller, and self-contained."""
    raw = schema.model_json_schema()
    definitions = raw.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {key: resolve(value) for key, value in node.items() if key != "title"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(raw)


class GroqChat:
    def __init__(self, model: str, client=None, cache_dir: Path | None = None, options: dict | None = None,
                 max_attempts: int = 6):
        if client is None:
            import groq

            if not os.environ.get("GROQ_API_KEY"):
                raise SystemExit("GROQ_API_KEY is not set. A free key from console.groq.com is enough.")
            client = groq.Groq(max_retries=0)       # waiting for rate limits is done here, and timed apart
        self.model = model
        self.client = client
        self.cache_dir = cache_dir          # replays identical requests for free; used by the evaluation
        self.options = options or {}        # extra request fields for this model, e.g. reasoning_effort
        self.max_attempts = max_attempts

    # -- public ---------------------------------------------------------------------------------

    def structured(self, agent: str, system: str, user: str, schema: type[T]) -> tuple[T, Call]:
        parameters = tool_schema(schema)
        key = self._key(system, user, schema.__name__, parameters)
        if (cached := self._read_cache(key)) is not None:
            call = Call(agent, self.model, cached["prompt_tokens"], cached["completion_tokens"],
                        cached["seconds"], cached=True)
            return schema.model_validate_json(cached["arguments"]), call

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt_tokens = completion_tokens = 0
        seconds = 0.0               # time the model took; waiting for a rate limit does not count
        problem = ""
        for _ in range(2):          # the request, then one repair
            arguments, usage, took = self._request(messages, parameters)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens
            seconds += took
            try:
                result = schema.model_validate_json(arguments)
            except ValidationError as exc:
                problem = str(exc)
                log.warning("%s: reply did not fit %s, asking again", agent, schema.__name__)
                messages = messages[:2] + [
                    {"role": "user", "content": f"Your last answer was rejected:\n{problem[:800]}\n"
                                                "Submit it again in the required form."},
                ]
                continue
            seconds = round(seconds, 3)
            self._write_cache(key, {"arguments": arguments, "prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens, "seconds": seconds})
            return result, Call(agent, self.model, prompt_tokens, completion_tokens, seconds)
        raise LLMError(f"{agent}: {self.model} did not return a valid {schema.__name__}: {problem[:300]}")

    # -- the request, with patience for rate limits ----------------------------------------------

    def _request(self, messages: list[dict], parameters: dict):
        """One reply as (JSON text, usage, seconds the model took)."""
        import groq

        tool = {"type": "function", "function": {
            "name": "submit", "description": "Hand in the result.", "parameters": parameters}}
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=[tool],
                    tool_choice={"type": "function", "function": {"name": "submit"}},
                    temperature=0, max_completion_tokens=MAX_OUTPUT_TOKENS, **self.options,
                )
            except groq.RateLimitError as exc:
                wait = float(exc.response.headers.get("retry-after", 10))
                if wait > LONGEST_WAIT or attempt == self.max_attempts:
                    raise LLMError(f"{self.model}: rate limit, retry after {wait:.0f} s: {exc}") from exc
                log.info("rate limit on %s, waiting %.0f s", self.model, wait)
                time.sleep(wait + 0.5)
            except (groq.BadRequestError, groq.APIConnectionError, groq.InternalServerError) as exc:
                # a malformed tool call is reported as a 400; it usually goes away on a second try
                if attempt == self.max_attempts or (isinstance(exc, groq.BadRequestError)
                                                    and "tool" not in str(exc).lower()):
                    raise LLMError(f"{self.model}: {exc}") from exc
                time.sleep(2)
            else:
                calls = response.choices[0].message.tool_calls
                if calls:
                    return calls[0].function.arguments, response.usage, time.perf_counter() - started
                messages = messages + [{"role": "user", "content": "Call the submit function."}]
        raise LLMError(f"{self.model}: no tool call after {self.max_attempts} attempts")

    # -- cache -------------------------------------------------------------------------------------

    def _key(self, *parts) -> str:
        payload = json.dumps([self.model, self.options, *parts], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _write_cache(self, key: str, entry: dict) -> None:
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{key}.json"
            path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")


def totals(calls: list[Call]) -> dict:
    return {
        "calls": len(calls),
        "prompt_tokens": sum(call.prompt_tokens for call in calls),
        "completion_tokens": sum(call.completion_tokens for call in calls),
        "llm_seconds": round(sum(call.seconds for call in calls), 2),
        "by_agent": {
            agent: sum(call.prompt_tokens + call.completion_tokens for call in calls if call.agent == agent)
            for agent in dict.fromkeys(call.agent for call in calls)
        },
    }


def call_as_dict(call: Call) -> dict:
    return asdict(call)
