"""HTTP endpoints under /v1: chat, streaming chat (SSE), RAG, and conversation history."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.agents.nodes import normalize_text
from app.agents.runner import AgentResult, run_agent, stream_agent
from app.api.errors import ApiError, translate_llm_error
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    RagRequest,
    RagResponse,
    Source,
    SourceOut,
    TokenUsage,
)
from app.config import get_settings
from app.db import repository
from app.db.database import DictPool
from app.guardrails.stream_guard import StreamGuard
from app.llm import client as llm
from app.observability import flush, make_callbacks, new_query_session
from app.rag.answer import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


def get_pool(request: Request) -> DictPool:
    return request.app.state.pool


Pool = Annotated[DictPool, Depends(get_pool)]


def get_graph(request: Request) -> Any:
    return request.app.state.graph


Graph = Annotated[Any, Depends(get_graph)]


async def _conversation_id(pool: DictPool, req: ChatRequest) -> UUID:
    """New conversations get a fresh id; an existing one is verified (404 otherwise)."""
    if req.conversation_id is None:
        return uuid4()
    if not await repository.conversation_exists(pool, req.conversation_id):
        raise ApiError(404, "conversation_not_found", "Conversation not found.")
    return req.conversation_id


def _usage_out(usage: llm.Usage) -> TokenUsage:
    return TokenUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)


async def _run(graph: Any, req: ChatRequest, conversation_id: UUID, model: str) -> AgentResult:
    try:
        callbacks = make_callbacks(new_query_session(), req.message, str(conversation_id))
        try:
            return await run_agent(graph, req.message, str(conversation_id), model, callbacks)
        finally:
            flush(callbacks)
    except Exception as exc:  # any provider error becomes a clean API error
        logger.warning("Agent run failed: %s", type(exc).__name__)
        raise translate_llm_error(exc) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, pool: Pool, graph: Graph) -> ChatResponse:
    """Ask the research assistant: full answer with sources and a grounded flag."""
    conversation_id = await _conversation_id(pool, req)
    model = req.model or get_settings().default_chat_model
    result = await _run(graph, req, conversation_id, model)

    await repository.save_exchange(
        pool,
        conversation_id,
        req.message,
        result.answer,
        model,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
    )
    return ChatResponse(
        conversation_id=conversation_id,
        model=model,
        answer=result.answer,
        grounded=result.grounded,
        sources=[SourceOut(**s) for s in result.sources],
        tools_used=result.tools_used,
        usage=_usage_out(result.usage),
        guardrails=result.guardrails,
    )


def word_chunks(text: str, words_per_chunk: int = 4) -> list[str]:
    """Split a finished answer into small pieces for streaming (spaces are preserved)."""
    words = text.split(" ")
    return [
        " ".join(words[i : i + words_per_chunk]) + (" " if i + words_per_chunk < len(words) else "")
        for i in range(0, len(words), words_per_chunk)
    ]


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, pool: Pool, graph: Graph) -> EventSourceResponse:
    """Streaming variant over Server-Sent Events: tokens are sent as the model generates them.

    Events (event name and JSON data):
      meta     {conversation_id, model}      right after the stream opens
      status   {step}                        the graph finished a step (input_guard, router, letters, ...)
      token    {text}                        next piece of the answer
      retract  {violations}                  already-shown text violated the output guardrails: the client
                                             erases the answer, no more tokens follow
      replace  {text}                        the final answer differs from what was shown: the client swaps it
                                             in (after retract, after the grounding guard removed sentences,
                                             or when no tokens were streamed at all)
      done     {answer, grounded, sources, tools_used, usage, guardrails}   end; answer is the final text
      error    {code, message}               failure (the HTTP status is already 200 by then)

    Why: tokens must flow immediately while guardrails check the whole answer. The stream guard
    (stream_guard.py) checks shown text at sentence boundaries; the final, fully checked answer
    always arrives in `done` and is what gets saved to history.
    """
    conversation_id = await _conversation_id(
        pool, req
    )  # a 404 is returned as a normal response, before the stream opens
    model = req.model or get_settings().default_chat_model

    async def events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "meta", "data": json.dumps({"conversation_id": str(conversation_id), "model": model})}
        guard = StreamGuard()
        try:
            result: AgentResult | None = None
            callbacks = make_callbacks(new_query_session(), req.message, str(conversation_id))
            async for kind, payload in stream_agent(graph, req.message, str(conversation_id), model, callbacks):
                if kind == "status":
                    yield {"event": "status", "data": json.dumps({"step": payload})}
                elif kind == "token":
                    text = normalize_text(payload)
                    action = guard.feed(text)
                    if action == "emit":
                        yield {"event": "token", "data": json.dumps({"text": text})}
                    elif action == "retract":
                        yield {"event": "retract", "data": json.dumps({"violations": guard.violations})}
                else:
                    result = payload
            if result is None:
                raise RuntimeError("The agent returned no result")
            if guard.needs_replace(result.answer):
                yield {"event": "replace", "data": json.dumps({"text": result.answer})}
            await repository.save_exchange(
                pool,
                conversation_id,
                req.message,
                result.answer,
                model,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            )
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "answer": result.answer,
                        "grounded": result.grounded,
                        "sources": result.sources,
                        "tools_used": result.tools_used,
                        "usage": _usage_out(result.usage).model_dump(),
                        "guardrails": result.guardrails,
                    }
                ),
            }
        except Exception as exc:
            # The 200 status is already sent, so the error goes out as an event.
            logger.warning("Streaming failed: %s", type(exc).__name__)
            err = translate_llm_error(exc)
            yield {"event": "error", "data": json.dumps({"code": err.code, "message": err.message})}

    return EventSourceResponse(events())


@router.post("/rag/ask", response_model=RagResponse)
async def rag_ask(req: RagRequest, pool: Pool) -> RagResponse:
    """Ask about Buffett's letters: a cited answer, or grounded=false if the letters don't cover it."""
    try:
        result = await answer_question(pool, req.question, req.model, req.k, req.mode)
    except ApiError:
        raise
    except Exception as exc:
        logger.warning("RAG failed: %s", type(exc).__name__)
        raise translate_llm_error(exc) from exc
    return RagResponse(
        answer=result.answer,
        grounded=result.grounded,
        sources=[Source(year=s.year, passage=s.content, chunk_id=s.chunk_id) for s in result.sources],
        top_similarity=round(result.top_similarity, 3),
        reason=result.reason,
        usage=TokenUsage(prompt_tokens=result.usage.prompt_tokens, completion_tokens=result.usage.completion_tokens),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(pool: Pool) -> list[dict]:
    return await repository.list_conversations(pool)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: UUID, pool: Pool) -> dict:
    conversation = await repository.get_conversation(pool, conversation_id)
    if conversation is None:
        raise ApiError(404, "conversation_not_found", "Conversation not found.")
    return conversation
