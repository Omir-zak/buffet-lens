"""LangGraph assembly.

  START
    │
input_guard ──(attack)──► END
    │
  router ──(off-topic)──► out_of_scope ──► END
    │     ──(attack)────► blocked ───────► END
    │
    ├──► letters ─┐
    ├──► market  ─┼──► synthesize ──► faithfulness ──► output_guard ──► END
    └──► web     ─┘     (needed nodes run in parallel)
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.deps import Deps
from app.agents.nodes import AgentNodes
from app.agents.state import AgentState


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)

    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("router", nodes.router)
    graph.add_node("letters", nodes.letters)
    graph.add_node("market", nodes.market)
    graph.add_node("web", nodes.web)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("out_of_scope", nodes.out_of_scope)
    graph.add_node("blocked", nodes.blocked)
    graph.add_node("faithfulness", nodes.faithfulness)
    graph.add_node("output_guard", nodes.output_guard)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", nodes.after_input_guard, {"router": "router", "end": END})
    graph.add_conditional_edges(
        "router", nodes.dispatch, ["letters", "market", "web", "synthesize", "out_of_scope", "blocked"]
    )
    for source in ("letters", "market", "web"):
        graph.add_edge(source, "synthesize")
    graph.add_edge("synthesize", "faithfulness")
    graph.add_edge("faithfulness", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("blocked", END)

    # The checkpointer persists state after each step (conversation memory).
    return graph.compile(checkpointer=checkpointer)
