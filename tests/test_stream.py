"""Streaming output: tokens arrive as they are generated while the guards stay in force.

The model, search and yfinance are faked. Tokens flow through a real LangGraph (messages mode).
"""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.agents.deps import Deps
from app.agents.graph import build_graph
from app.agents.nodes import RouteDecision
from app.agents.runner import AgentResult, stream_agent
from app.api import routes
from app.api.errors import ApiError
from app.guardrails.stream_guard import StreamGuard
from app.llm.client import Usage
from app.main import api_error_handler
from app.rag.retrieval import Retrieved

# ---------- StreamGuard ----------


def feed_all(guard: StreamGuard, tokens: list[str]) -> list[str]:
    return [guard.feed(t) for t in tokens]


def test_clean_text_is_streamed_without_interruption():
    guard = StreamGuard()
    actions = feed_all(guard, ["Coca-Cola ", "has a ", "wide moat", ". ", "Its P/E ", "is 25.88", ". "])
    assert set(actions) == {"emit"} and not guard.halted


def test_advice_is_retracted_at_the_end_of_the_sentence_and_the_rest_is_dropped():
    guard = StreamGuard()
    actions = feed_all(guard, ["You ", "should ", "buy ", "Apple ", "shares ", "now", ". ", "It is cheap", ". "])
    assert actions[:6] == ["emit"] * 6  # a violation is not visible until the sentence ends
    assert actions[6] == "retract" and guard.halted and guard.violations
    assert actions[7:] == ["drop", "drop"]


def test_a_forecast_in_the_stream_is_retracted():
    guard = StreamGuard()
    actions = feed_all(guard, ["The market value ", "could reach $2 trillion ", "by the end of 2031", ". "])
    assert actions[-1] == "retract"


def test_final_answer_replaces_the_streamed_text_only_when_it_differs():
    guard = StreamGuard()
    feed_all(guard, ["Hello ", "world", ". "])
    assert not guard.needs_replace("Hello world.")  # surrounding whitespace is irrelevant
    assert guard.needs_replace("Hello there.")
    assert StreamGuard().needs_replace("Anything")  # no tokens at all: a replace is needed


def test_halted_stream_always_needs_a_replace_even_if_the_final_text_matches():
    guard = StreamGuard()
    feed_all(guard, ["You should buy it now", ". "])
    assert guard.halted and guard.needs_replace("You should buy it now.")


# ---------- Tokens through a real LangGraph ----------


def fake_deps(answer: str) -> Deps:
    async def ask_json(model, system, user, schema, **kwargs):
        return RouteDecision(needs_letters=True, letters_searches=[{"query": "coke", "years": []}]), Usage(1, 1)

    async def retrieve(pool, query, k=5, mode="hybrid", strategy=None, years=None, **kwargs):
        return [Retrieved("1988-042", 1988, "Coke is a wonderful business.", 0.7, 0.7)]

    async def generate(model, messages, temperature=None):
        # A real streaming model call inside the node: LangGraph captures tokens via callbacks.
        llm = GenericFakeChatModel(messages=iter([AIMessage(content=answer)]))
        response = await llm.ainvoke(messages)
        return str(response.content), Usage(5, 5)

    return Deps(pool=None, ask_json=ask_json, generate=generate, retrieve=retrieve)


async def collect(graph, question="How does Buffett see Coke?"):
    events = [event async for event in stream_agent(graph, question, "thread-1", None)]
    return events


async def test_tokens_arrive_one_by_one_from_the_synthesizer_before_the_result():
    answer = "Coke is a wonderful business [1988-042]."
    events = await collect(build_graph(fake_deps(answer)))
    kinds = [kind for kind, _ in events]
    tokens = [payload for kind, payload in events if kind == "token"]

    assert len(tokens) > 3  # streamed word by word, not in one piece
    assert "".join(tokens) == answer
    assert kinds[-1] == "result" and kinds.index("token") < kinds.index("result")
    assert events[-1][1].answer == answer and events[-1][1].grounded


async def test_router_output_is_never_streamed_to_the_user():
    events = await collect(build_graph(fake_deps("Coke is great [1988-042].")))
    streamed = "".join(payload for kind, payload in events if kind == "token")
    assert "letters_searches" not in streamed and "needs_letters" not in streamed


async def test_status_events_still_arrive_between_the_tokens_and_the_result():
    events = await collect(build_graph(fake_deps("Coke is great [1988-042].")))
    steps = [payload for kind, payload in events if kind == "status"]
    assert steps[0] == "input_guard" and "router" in steps and "synthesize" in steps


# ---------- Endpoint /v1/chat/stream ----------


def result_for(answer: str) -> AgentResult:
    return AgentResult(
        answer=answer, grounded=True, sources=[], tools_used=["letters"], route={}, usage=Usage(10, 5), guardrails={}
    )


@pytest.fixture
def stream_app(monkeypatch):
    saved = []

    async def save_exchange(pool, conversation_id, user_text, assistant_text, model, p, c, user_id=None):
        saved.append(assistant_text)

    monkeypatch.setattr(routes.repository, "save_exchange", save_exchange)
    scenario = {"events": [], "fail": None}

    async def fake_stream_agent(graph, question, thread_id, model, callbacks=None):
        if scenario["fail"]:
            raise scenario["fail"]
        for event in scenario["events"]:
            yield event

    monkeypatch.setattr(routes, "stream_agent", fake_stream_agent)
    app = FastAPI()
    app.include_router(routes.router)
    app.add_exception_handler(ApiError, api_error_handler)
    app.dependency_overrides[routes.get_pool] = lambda: "pool"
    app.dependency_overrides[routes.get_graph] = lambda: "graph"
    app.state.graph = SimpleNamespace()
    return SimpleNamespace(app=app, scenario=scenario, saved=saved)


async def post_stream(stream_app):
    transport = httpx.ASGITransport(app=stream_app.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/stream", json={"message": "hi"})
    return parse_sse(response.text)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line[5:].strip())))
            name = None
    return events


async def test_tokens_are_forwarded_live_and_the_final_answer_is_saved(stream_app):
    final = "Coke is great [1988-042]."
    stream_app.scenario["events"] = [
        ("status", "router"),
        ("token", "Coke is "),
        ("token", "great "),
        ("token", "[1988-042]."),
        ("result", result_for(final)),
    ]
    events = await post_stream(stream_app)
    names = [n for n, _ in events]
    assert names == [
        "meta",
        "status",
        "token",
        "token",
        "token",
        "done",
    ]  # no replace: what was shown matches the final text
    assert "".join(d["text"] for n, d in events if n == "token") == final
    assert events[-1][1]["answer"] == final and stream_app.saved == [final]


async def test_violation_in_the_stream_is_retracted_and_replaced_by_the_safe_answer(stream_app):
    safe = "This tool does not give recommendations. P/E is 25.88 [M:KO]."
    stream_app.scenario["events"] = [
        ("token", "You should buy "),
        ("token", "Coke now"),
        ("token", ". "),
        ("token", "It is cheap. "),
        ("result", result_for(safe)),
    ]
    events = await post_stream(stream_app)
    names = [n for n, _ in events]
    assert names == ["meta", "token", "token", "retract", "replace", "done"]  # no new tokens after retract
    assert events[names.index("replace")][1]["text"] == safe
    assert stream_app.saved == [safe]  # only the verified text is saved to history


async def test_when_no_tokens_were_streamed_the_answer_arrives_as_replace(stream_app):
    refusal = "The letters and data available do not contain enough information to answer this."
    stream_app.scenario["events"] = [("status", "input_guard"), ("result", result_for(refusal))]
    names = [n for n, _ in await post_stream(stream_app)]
    assert names == ["meta", "status", "replace", "done"]


async def test_faithfulness_edits_after_streaming_are_sent_as_replace(stream_app):
    stream_app.scenario["events"] = [
        ("token", "P/E is 25.88 [M:KO]. "),
        ("token", "Yield is 9.9% [M:KO]. "),
        ("result", result_for("P/E is 25.88 [M:KO].")),
    ]
    events = await post_stream(stream_app)
    assert [n for n, _ in events][-2:] == ["replace", "done"]
    assert events[-2][1]["text"] == "P/E is 25.88 [M:KO]."


async def test_error_during_the_stream_is_reported_as_an_error_event_and_nothing_is_saved(stream_app):
    stream_app.scenario["fail"] = RuntimeError("boom")
    events = await post_stream(stream_app)
    assert [n for n, _ in events] == ["meta", "error"]
    assert events[-1][1]["code"] == "upstream_error" and stream_app.saved == []
