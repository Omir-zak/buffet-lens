"""Frontend compatibility layer: stream format, API key, models, deletion and feedback.

The graph and database are faked, so these tests need neither network nor Postgres.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI

from app.agents.runner import AgentResult
from app.api import auth, compat
from app.api.errors import ApiError
from app.api.model_aliases import resolve_model
from app.api.routes import get_graph, get_pool
from app.llm.client import Usage
from app.main import api_error_handler


def make_result(**overrides) -> AgentResult:
    base = {
        "answer": "Coca-Cola's trailing P/E is 25.88 [M:KO].",
        "grounded": True,
        "sources": [{"type": "market", "id": "M:KO", "title": "Coca-Cola", "passage": "..."}],
        "tools_used": ["market_data"],
        "route": {},
        "usage": Usage(100, 20),
    }
    base.update(overrides)
    return AgentResult(**base)


class FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


class FakeRepository:
    """Records calls instead of hitting the DB."""

    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.feedback: list[tuple] = []
        self.existing: set = set()

    async def save_exchange(self, pool, conversation_id, user_text, assistant_text, model, p, c, user_id=None):
        self.saved.append({"id": conversation_id, "q": user_text, "a": assistant_text, "model": model, "user": user_id})

    async def save_feedback(self, pool, message_id, user_id, rating):
        self.feedback.append((message_id, user_id, rating))

    async def delete_conversation(self, pool, conversation_id):
        return conversation_id in self.existing


@pytest.fixture
def world(monkeypatch):
    repo = FakeRepository()
    for name in ("save_exchange", "save_feedback", "delete_conversation"):
        monkeypatch.setattr(compat.repository, name, getattr(repo, name))

    calls = {"stream": [], "result": make_result(), "fail": None}

    async def fake_stream_agent(graph, question, thread_id, model, callbacks=None):
        calls["stream"].append({"q": question, "thread": thread_id, "model": model})
        if calls["fail"]:
            raise calls["fail"]
        yield "status", "router"
        yield "status", "letters"
        yield "status", "input_guard"  # internal step: not shown to the frontend
        yield "status", "synthesize"
        yield "result", calls["result"]

    async def fake_run_agent(graph, question, thread_id, model, callbacks=None):
        calls["stream"].append({"q": question, "thread": thread_id, "model": model})
        if calls["fail"]:
            raise calls["fail"]
        return calls["result"]

    monkeypatch.setattr(compat, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(compat, "run_agent", fake_run_agent)
    return SimpleNamespace(repo=repo, calls=calls, checkpointer=FakeCheckpointer())


def build_client(world, api_key: str = "", monkeypatch=None) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(compat.router, dependencies=[Depends(auth.require_api_key)])
    app.add_exception_handler(ApiError, api_error_handler)
    app.dependency_overrides[get_pool] = lambda: "pool"
    app.dependency_overrides[get_graph] = lambda: "graph"
    app.state.graph = SimpleNamespace(checkpointer=world.checkpointer)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse_stream(text: str) -> list[dict | str]:
    """Parses the stream like the frontend: `data: ...` lines separated by `\n`."""
    events: list[dict | str] = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            body = line[6:]
            events.append(body if body == "[DONE]" else json.loads(body))
    return events


# ---------- Pure functions ----------


def msgs(*pairs):
    return [compat.ChatMessage(role=r, content=c) for r, c in pairs]


def test_last_user_message_is_used_and_earlier_history_is_ignored():
    messages = msgs(("user", "first"), ("assistant", "answer"), ("user", "  second question  "))
    assert compat.last_user_text(messages) == "second question"


def test_text_parts_format_is_supported():
    message = compat.ChatMessage(
        role="user", content=None, parts=[{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    )
    assert compat.last_user_text([message]) == "hello world"


@pytest.mark.parametrize("bad", [[("assistant", "hi")], [("user", "   ")], [("user", "x" * 4001)]])
def test_empty_or_too_long_questions_are_rejected(bad):
    with pytest.raises(ApiError) as exc:
        compat.last_user_text(msgs(*bad))
    assert exc.value.status_code == 422


def test_sources_are_appended_as_markdown():
    result = make_result(
        sources=[
            {"type": "letter", "id": "1988-042", "year": 1988, "passage": "Coke is a wonderful business. " * 20},
            {"type": "market", "id": "M:KO", "title": "Coca-Cola"},
            {"type": "web", "id": "W1", "title": "News [live]", "url": "https://example.com/a"},
        ]
    )
    text = compat.render_answer(result)
    assert text.startswith(result.answer) and "**Sources**" in text
    assert "Letter 1988 (`1988-042`)" in text and text.count("...") >= 1  # a long quote is truncated
    assert "[News (live)](https://example.com/a)" in text  # brackets in the title do not break the link


def test_answer_without_sources_is_returned_unchanged():
    result = make_result(sources=[], grounded=False, answer="No info.")
    assert compat.render_answer(result) == "No info."


def test_frontend_model_names_are_mapped_to_our_models_and_unknown_names_pass_through(monkeypatch):
    assert resolve_model("gpt-4o-mini") == "openai/gpt-oss-20b"
    assert resolve_model("claude-sonnet-4-6") == "openai/gpt-oss-120b"
    assert (
        resolve_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    )  # a real provider model id is passed through unchanged
    assert resolve_model(None)  # empty: falls back to the default model
    assert resolve_model("")


# ---------- Endpoints ----------


async def test_stream_matches_the_format_the_frontend_parses(world):
    cid = uuid4()
    async with build_client(world) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "P/E of Coke?"}],
                "model": "gpt-4o-mini",
                "user_id": "user_001",
                "conversation_id": str(cid),
                "stream": True,
            },
        )
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = parse_stream(response.text)

    assert events[0] == {"type": "metadata", "conversation_id": str(cid)}
    statuses = [e["message"] for e in events if isinstance(e, dict) and e.get("type") == "status"]
    assert statuses == ["question analysis...", "letters search...", "answer synthesis..."]  # input_guard is hidden

    pieces = [
        e["choices"][0]["delta"]["content"]
        for e in events
        if isinstance(e, dict) and e.get("choices") and e["choices"][0]["delta"]
    ]
    assert "".join(pieces) == compat.render_answer(world.calls["result"])
    assert events[-2]["choices"][0]["finish_reason"] == "stop" and events[-1] == "[DONE]"

    saved = world.repo.saved[0]
    assert saved["model"] == "openai/gpt-oss-20b" and saved["user"] == "user_001" and saved["id"] == cid
    assert world.calls["stream"][0]["thread"] == str(cid)


async def test_new_conversation_gets_an_id_in_the_metadata_event(world):
    async with build_client(world) as client:
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    metadata = parse_stream(response.text)[0]
    assert metadata["type"] == "metadata" and metadata["conversation_id"]


async def test_error_in_the_stream_becomes_visible_text_and_nothing_is_saved(world):
    world.calls["fail"] = RuntimeError("boom")
    async with build_client(world) as client:
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    events = parse_stream(response.text)
    text = "".join(
        e["choices"][0]["delta"].get("content", "") for e in events if isinstance(e, dict) and e.get("choices")
    )
    assert "⚠" in text and events[-1] == "[DONE]"
    assert world.repo.saved == []  # a failed answer is not saved to history


async def test_non_streaming_response_is_a_chat_completion_object(world):
    async with build_client(world) as client:
        response = await client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": False}
        )
    body = response.json()
    assert body["object"] == "chat.completion" and body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert body["grounded"] is True and body["tools_used"] == ["market_data"]


async def test_validation_error_for_missing_user_message(world):
    async with build_client(world) as client:
        response = await client.post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert response.status_code == 422 and response.json()["error"]["code"] == "empty_message"


async def test_feedback_is_stored_and_invalid_rating_is_rejected(world):
    async with build_client(world) as client:
        ok = await client.post("/v1/feedback", params={"message_id": "m-1", "rating": 1, "user_id": "user_001"})
        down = await client.post("/v1/feedback", params={"message_id": "m-2", "rating": -1})
        bad = await client.post("/v1/feedback", params={"message_id": "m-3", "rating": 5})
    assert ok.json() == {"success": True} and down.status_code == 200
    assert world.repo.feedback == [("m-1", "user_001", 1), ("m-2", "", -1)]
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_rating"


async def test_delete_removes_the_conversation_and_its_graph_state(world):
    cid = uuid4()
    world.repo.existing.add(cid)
    async with build_client(world) as client:
        response = await client.delete(f"/v1/conversations/{cid}")
    assert response.status_code == 204 and world.checkpointer.deleted == [str(cid)]


async def test_delete_of_unknown_conversation_is_404_and_touches_no_state(world):
    async with build_client(world) as client:
        response = await client.delete(f"/v1/conversations/{uuid4()}")
    assert response.status_code == 404 and world.checkpointer.deleted == []


# ---------- API key ----------


@pytest.fixture
def key_required(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(api_key="secret-key"))


async def test_requests_without_or_with_a_wrong_key_are_rejected_when_a_key_is_configured(world, key_required):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    async with build_client(world) as client:
        missing = await client.post("/v1/chat/completions", json=body)
        wrong = await client.post("/v1/chat/completions", json=body, headers={"x-api-key": "nope"})
        right = await client.post("/v1/chat/completions", json=body, headers={"x-api-key": "secret-key"})
    assert missing.status_code == 401 and wrong.status_code == 401
    assert missing.json()["error"]["code"] == "invalid_api_key"
    assert right.status_code == 200
    assert len(world.calls["stream"]) == 1  # without a key the graph never ran


async def test_all_compat_endpoints_are_protected(world, key_required):
    async with build_client(world) as client:
        responses = [
            await client.post("/v1/feedback", params={"message_id": "m", "rating": 1}),
            await client.delete(f"/v1/conversations/{uuid4()}"),
        ]
    assert [r.status_code for r in responses] == [401, 401]


async def test_api_is_open_when_no_key_is_configured(world, monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: SimpleNamespace(api_key=""))
    async with build_client(world) as client:
        response = await client.post("/v1/feedback", params={"message_id": "m", "rating": 1})
    assert response.status_code == 200
