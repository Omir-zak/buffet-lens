"""Shared state passed between agent graph nodes.

The checkpointer persists it after each step, so a follow-up question in the same `thread_id`
already finds the full history in `messages`.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_turn(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Reducer for per-question bookkeeping (tokens, errors, per-node timings).

    Parallel nodes write to the same field, so dicts are merged. State persists across
    questions, hence each question starts with {"_reset": True} to clear earlier data.
    """
    new = dict(new or {})
    if new.pop("_reset", False):
        return new
    return {**(old or {}), **new}


class AgentState(TypedDict, total=False):
    # add_messages appends to the history instead of overwriting it.
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    model: str | None

    route: dict[str, Any]  # router decision: which sources are needed
    passages: list[dict[str, Any]]  # retrieved letter passages
    market: list[dict[str, Any]]  # yfinance results (one per ticker)
    web: dict[str, Any] | None  # web search result

    answer: str
    grounded: bool
    sources: list[dict[str, Any]]
    blocked: str | None  # reason the input guard blocked the question, if any
    advice_seeking: bool  # user asks for advice: answer without recommendations

    turn: Annotated[dict[str, Any], merge_turn]
