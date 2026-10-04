"""Runs the graph for a single question and collects the result."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage

from app.llm.client import Usage


@dataclass
class AgentResult:
    answer: str
    grounded: bool
    sources: list[dict[str, Any]]
    tools_used: list[str]
    route: dict[str, Any]
    usage: Usage
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    blocked: str | None = None
    guardrails: dict[str, Any] = field(default_factory=dict)
    tokens_by_node: dict[str, list[int]] = field(default_factory=dict)  # {node: [prompt, completion]}


def _initial_input(question: str, model: str | None) -> dict[str, Any]:
    """Input for a new turn. Per-turn fields are reset because state persists between questions;
    the history (messages) is intentionally appended to, not reset."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "model": model,
        "route": {},
        "passages": [],
        "market": [],
        "web": None,
        "answer": "",
        "grounded": False,
        "sources": [],
        "blocked": None,
        "advice_seeking": False,
        "turn": {"_reset": True},
    }


def result_from_state(state: dict[str, Any]) -> AgentResult:
    turn = state.get("turn", {})
    usage = Usage()
    for info in turn.values():
        if "tokens" in info:
            usage = usage + Usage(*info["tokens"])
    tools = []
    if state.get("passages"):
        tools.append("letters")
    if any(m.get("ok") for m in state.get("market", [])):
        tools.append("market_data")
    web = state.get("web")
    if web and web.get("ok") and web.get("results"):
        tools.append("web")
    return AgentResult(
        answer=state.get("answer", ""),
        grounded=bool(state.get("grounded")),
        sources=state.get("sources", []),
        tools_used=tools,
        route=state.get("route", {}),
        usage=usage,
        timings_ms={name: info["ms"] for name, info in turn.items() if "ms" in info},
        warnings=[info["error"] for info in turn.values() if info.get("error")],
        blocked=state.get("blocked"),
        tokens_by_node={name: info["tokens"] for name, info in turn.items() if "tokens" in info},
        guardrails={
            "input_rule": turn.get("input_guard", {}).get("rule"),
            "advice_seeking": bool(state.get("advice_seeking")),
            "faithfulness_removed": len(turn.get("faithfulness", {}).get("removed", [])),
            "output_action": turn.get("output_guard", {}).get("action"),
            "output_violations": turn.get("output_guard", {}).get("violations", []),
        },
    )


def _config(thread_id: str, callbacks: list | None) -> dict[str, Any]:
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return config


async def run_agent(
    graph: Any, question: str, thread_id: str, model: str | None = None, callbacks: list | None = None
) -> AgentResult:
    state = await graph.ainvoke(_initial_input(question, model), _config(thread_id, callbacks))
    return result_from_state(state)


async def stream_agent(
    graph: Any, question: str, thread_id: str, model: str | None = None, callbacks: list | None = None
) -> AsyncIterator[tuple[str, Any]]:
    """Yield events as the graph runs:
      ("status", node_name)    a node finished;
      ("token", text)          next chunk of the answer as the synthesizer generates it;
      ("result", AgentResult)  at the very end, after the guards (may differ from the token stream).

    Three LangGraph stream modes are consumed together: "updates" (node outputs), "messages"
    (LLM tokens) and "values" (full state after each step; the final result is taken from it,
    so a checkpointer is not required here).
    """
    config = _config(thread_id, callbacks)
    final_state: dict[str, Any] = {}
    async for mode, data in graph.astream(
        _initial_input(question, model), config, stream_mode=["updates", "messages", "values"]
    ):
        if mode == "values":
            final_state = data
        elif mode == "updates":
            for node_name in data:
                yield "status", node_name
        elif mode == "messages":
            chunk, metadata = data
            # Only the synthesizer's tokens are streamed: the router also calls the model, but its JSON is not user-facing.
            if metadata.get("langgraph_node") == "synthesize" and isinstance(chunk.content, str) and chunk.content:
                yield "token", chunk.content
    yield "result", result_from_state(final_state)
