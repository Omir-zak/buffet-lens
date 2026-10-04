"""OpenAI-compatible layer for the Next.js frontend, which was written against a different backend.

These endpoints run the same agent graph as /v1/chat.

  POST   /v1/chat/completions       OpenAI-format request; streamed (stream=true) or full response
  POST   /v1/feedback               answer rating (thumbs up/down), parameters in the query string
  DELETE /v1/conversations/{id}     delete a conversation

Stream format (lines like `data: {...}`):
  {"type": "metadata", "conversation_id": "..."}              conversation id (the frontend remembers it)
  {"type": "status", "message": "letters search..."}          "Running letters search..." badge
  {"choices": [{"delta": {"content": "..."}}]}                text pieces (OpenAI format)
  data: [DONE]                                                end
The frontend can't show in-stream errors, so an error is sent as ordinary answer text.
"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.agents.runner import AgentResult, run_agent, stream_agent
from app.api.errors import ApiError, translate_llm_error
from app.api.model_aliases import resolve_model
from app.api.routes import Graph, Pool, word_chunks
from app.db import repository
from app.observability import flush, make_callbacks, new_query_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["frontend-compat"])

MAX_MESSAGE_CHARS = 4000
SNIPPET_CHARS = 160

# The frontend renders "Running <label>..." / "Finished <label>", so labels are noun phrases.
STATUS_LABELS = {
    "router": "question analysis",
    "letters": "letters search",
    "market": "market data lookup",
    "web": "web search",
    "synthesize": "answer synthesis",
}


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")  # extra fields (id, createdAt, ...) are ignored

    role: str
    content: str | list[dict[str, Any]] | None = None
    parts: list[dict[str, Any]] | None = None  # newer Vercel AI SDK versions store text this way


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    user_id: str | None = Field(default=None, max_length=100)
    conversation_id: UUID | None = None
    stream: bool = True


def _text_of(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = message.content if isinstance(message.content, list) else (message.parts or [])
    return "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("type", "text") == "text")


def last_user_text(messages: list[ChatMessage]) -> str:
    """Text of the last user message. The frontend sends the whole history, but ours lives in
    the graph state keyed by conversation_id, so only the last question is needed."""
    for message in reversed(messages):
        if message.role == "user":
            text = _text_of(message).strip()
            if not text:
                break
            if len(text) > MAX_MESSAGE_CHARS:
                raise ApiError(422, "message_too_long", f"The message is longer than {MAX_MESSAGE_CHARS} characters.")
            return text
    raise ApiError(422, "empty_message", "There is no user message to answer.")


def _snippet(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rstrip() + "..."


def render_answer(result: AgentResult) -> str:
    """Answer as Markdown with a sources list. The frontend has no sources field but renders
    Markdown, so sources are appended to the text."""
    if not result.sources:
        return result.answer
    lines = []
    for source in result.sources:
        if source["type"] == "letter":
            shown = source.get("excerpt") or source.get("passage", "")
            lines.append(f"- Letter {source.get('year')} (`{source['id']}`): “{_snippet(shown)}”")
        elif source["type"] == "market":
            lines.append(f"- Market data: {source.get('title') or source['id']} (`{source['id']}`)")
        else:
            title = (source.get("title") or source["id"]).replace("[", "(").replace("]", ")")
            url = source.get("url")
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
    return f"{result.answer}\n\n---\n**Sources**\n" + "\n".join(lines)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk(chunk_id: str, created: int, model: str, content: str | None, finish: str | None) -> dict[str, Any]:
    delta = {} if content is None else {"content": content}
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


async def _completion_stream(
    graph: Any, pool: Any, req: CompletionRequest, question: str, conversation_id: UUID, model: str
) -> AsyncIterator[str]:
    chunk_id, created = f"chatcmpl-{uuid4().hex[:24]}", int(time.time())
    yield _sse({"type": "metadata", "conversation_id": str(conversation_id)})
    try:
        callbacks = make_callbacks(new_query_session(), question, str(conversation_id))
        result: AgentResult | None = None
        async for kind, payload in stream_agent(graph, question, str(conversation_id), model, callbacks):
            if kind == "status":
                if label := STATUS_LABELS.get(payload):
                    yield _sse({"type": "status", "message": f"{label}..."})
            elif kind == "result":
                result = payload
        flush(callbacks)
        if result is None:
            raise RuntimeError("The agent returned no result")
        text = render_answer(result)
        for piece in word_chunks(text):
            yield _sse(_chunk(chunk_id, created, model, piece, None))
        await repository.save_exchange(
            pool, conversation_id, question, text, model,
            result.usage.prompt_tokens, result.usage.completion_tokens, req.user_id,
        )  # fmt: skip
        yield _sse(_chunk(chunk_id, created, model, None, "stop"))
    except Exception as exc:
        logger.warning("Compat stream failed: %s", type(exc).__name__)
        err = translate_llm_error(exc)
        # The frontend doesn't display in-stream errors, so report it as answer text.
        yield _sse(_chunk(chunk_id, created, model, f"\n\n⚠️ {err.message}", None))
        yield _sse(_chunk(chunk_id, created, model, None, "error"))
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(req: CompletionRequest, pool: Pool, graph: Graph) -> Response:
    """OpenAI-compatible chat. stream=true: event stream; otherwise one chat.completion JSON."""
    question = last_user_text(req.messages)
    model = resolve_model(req.model)
    # An unknown conversation_id is not an error: if the previous answer was cancelled with Stop, the
    # conversation wasn't saved but the frontend already remembers its id. Continue under the same id.
    conversation_id = req.conversation_id or uuid4()

    if req.stream:
        headers = {"Cache-Control": "no-store", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        return StreamingResponse(
            _completion_stream(graph, pool, req, question, conversation_id, model),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        callbacks = make_callbacks(new_query_session(), question, str(conversation_id))
        try:
            result = await run_agent(graph, question, str(conversation_id), model, callbacks)
        finally:
            flush(callbacks)
    except Exception as exc:
        logger.warning("Compat completion failed: %s", type(exc).__name__)
        raise translate_llm_error(exc) from exc

    text = render_answer(result)
    await repository.save_exchange(
        pool, conversation_id, question, text, model,
        result.usage.prompt_tokens, result.usage.completion_tokens, req.user_id,
    )  # fmt: skip
    usage = result.usage
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": (usage.prompt_tokens or 0) + (usage.completion_tokens or 0),
            },
            # Extensions beyond the OpenAI format:
            "conversation_id": str(conversation_id),
            "grounded": result.grounded,
            "sources": result.sources,
            "tools_used": result.tools_used,
        }
    )


@router.post("/feedback")
async def submit_feedback(
    pool: Pool,
    message_id: Annotated[str, Query(min_length=1, max_length=100)],
    rating: Annotated[int, Query(description="1 (like) or -1 (dislike)")],
    user_id: Annotated[str | None, Query(max_length=100)] = None,
) -> dict[str, bool]:
    """Rate an answer. Parameters are in the query string, as the frontend sends them."""
    if rating not in (-1, 1):
        raise ApiError(422, "invalid_rating", "rating must be 1 or -1.")
    await repository.save_feedback(pool, message_id, user_id or "", rating)
    return {"success": True}


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: UUID, pool: Pool, request: Request) -> Response:
    """Delete a conversation: its messages and the saved graph state (conversation memory)."""
    if not await repository.delete_conversation(pool, conversation_id):
        raise ApiError(404, "conversation_not_found", "Conversation not found.")
    try:
        await request.app.state.graph.checkpointer.adelete_thread(str(conversation_id))
    except Exception as exc:  # already deleted from the DB; leftover graph state must not break the response
        logger.warning("Could not delete graph checkpoints: %s", type(exc).__name__)
    return Response(status_code=204)
