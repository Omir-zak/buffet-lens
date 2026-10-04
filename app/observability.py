"""Langfuse tracing of every question via a LangGraph callback handler.

Each question is one trace (graph nodes appear as spans, LLM calls as generations). Without
Langfuse keys, tracing is silently disabled and the app works as usual.
"""

import logging
from typing import Any
from uuid import uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)


def install_langchain_compat() -> None:
    """Compatibility shim for the Langfuse v2 SDK with langchain 1.x.

    Langfuse v2 imports `langchain.callbacks.base`, `langchain.schema.agent` and
    `langchain.schema.document`, which no longer exist in langchain 1.x (they live in
    langchain-core). The old names are registered as aliases BEFORE the handler is imported.
    We stay on SDK v2 because the self-hosted Langfuse v3 server needs ClickHouse, Redis and S3,
    which is too heavy for the local stack. Test: tests/test_observability.py.
    """
    import sys
    import types

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.callbacks import base as callbacks_base
    from langchain_core.documents import Document

    def module(name: str, **attrs: Any) -> types.ModuleType:
        mod = sys.modules.get(name) or types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules.setdefault(name, mod)
        return sys.modules[name]

    base_module = module("langchain.callbacks.base", BaseCallbackHandler=callbacks_base.BaseCallbackHandler)
    module("langchain.callbacks", base=base_module)
    agent_module = module("langchain.schema.agent", AgentAction=AgentAction, AgentFinish=AgentFinish)
    document_module = module("langchain.schema.document", Document=Document)
    module("langchain.schema", agent=agent_module, document=document_module)


def tracing_enabled() -> bool:
    s = get_settings()
    placeholder = lambda v: not v or v.endswith("...")  # noqa: E731  (placeholder keys from .env.example don't count)
    return not (placeholder(s.langfuse_public_key) or placeholder(s.langfuse_secret_key))


def new_query_session() -> str:
    """New Langfuse session for a single question (the spec makes each user request its own session).
    The owning conversation is recorded in the trace metadata and tag."""
    return f"query-{uuid4().hex[:16]}"


def make_callbacks(session_id: str, query: str | None = None, conversation_id: str | None = None) -> list:
    """Callback handlers for one request (empty if tracing is disabled)."""
    if not tracing_enabled():
        return []
    try:
        install_langchain_compat()
        from langfuse.callback import CallbackHandler

        s = get_settings()
        handler = CallbackHandler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
            session_id=session_id,
            trace_name="buffett-lens-query",
            metadata={"question": (query or "")[:500], "conversation_id": conversation_id},
            tags=[f"conversation:{conversation_id}"] if conversation_id else None,
        )
        return [handler]
    except Exception as exc:  # tracing must never break the user's answer
        logger.warning("Langfuse tracing disabled for this request: %s", type(exc).__name__)
        return []


def flush(callbacks: list) -> None:
    """Langfuse sends data in background batches; short-lived processes (eval) must flush explicitly."""
    for handler in callbacks:
        try:
            handler.flush()
        except Exception:
            logger.warning("Langfuse flush failed")
