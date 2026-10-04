"""Agent graph nodes: router, letters, market, web, synthesize.

Design:
  * A single router decides which sources are needed and runs those nodes in parallel
    (letters, market and web are independent).
  * Specialist nodes only gather evidence; they do not write the answer.
  * The synthesizer turns the evidence into a Buffett-style answer with source citations.
  * Each node handles its own errors, so one failing source does not take down the graph.
A fixed graph is used instead of a tool-calling agent loop because the steps are known in
advance: it is cheaper (2 LLM calls instead of 3-6), faster and more predictable.
"""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from app.agents.deps import Deps
from app.agents.state import AgentState
from app.config import get_settings
from app.guardrails.faithfulness import check_faithfulness
from app.guardrails.input_guard import INJECTION_MESSAGE, check_input_rules
from app.guardrails.output_guard import apply_output_guard
from app.llm.client import Usage, load_prompt
from app.rag.retrieval import Retrieved
from app.tools.market import MarketData
from app.tools.web import WebSearchOutput

NO_INFO_MESSAGE = "The letters and data available do not contain enough information to answer this."
OUT_OF_SCOPE_MESSAGE = (
    "I can help with researching public companies through Warren Buffett's investing philosophy, using his "
    "shareholder letters, live market data and current news. That question is outside what I can help with, "
    "but feel free to ask about a company, a Buffett principle or a valuation metric."
)
REFUSAL_PREFIX = "the letters and data available do not contain enough information"
SYNTH_TEMPERATURE = 0.2  # low temperature: fact-based answers should be stable
MAX_TICKERS = 3
MAX_LETTER_SEARCHES = 3
PASSAGES_PER_SEARCH = 4
HISTORY_MESSAGES = 6
CITATION = re.compile(r"\[([^\[\]]+)\]")


class LetterSearch(BaseModel):
    """One search query against the letters; comparison questions get several (one per topic)."""

    query: str
    years: list[int] = Field(default_factory=list)  # letter years named by the user


class RouteDecision(BaseModel):
    """Router decision, validated against the shape the model must return."""

    in_scope: bool = True
    needs_letters: bool = False
    needs_market: bool = False
    needs_web: bool = False
    tickers: list[str] = Field(default_factory=list)
    letters_searches: list[LetterSearch] = Field(default_factory=list)
    web_query: str = ""
    injection_suspected: bool = False  # LLM layer of the input guard: looks like an injection attempt
    advice_seeking: bool = False  # asks for buy/sell advice
    reason: str = ""

    @model_validator(mode="after")
    def infer_flags(self) -> "RouteDecision":
        """A small model sometimes fills in searches or tickers but forgets the needs_* flag,
        so infer it: any search query or ticker implies the source is needed."""
        if self.letters_searches:
            self.needs_letters = True
        if self.tickers:
            self.needs_market = True
        if self.web_query:
            self.needs_web = True
        return self


@dataclass(frozen=True)
class Evidence:
    """One identified block of facts that the answer can cite."""

    id: str
    kind: str  # "letter" | "market" | "web"
    text: str
    meta: dict[str, Any]


# ---------- Pure functions (no network or DB) ----------


def build_evidence(
    passages: list[dict[str, Any]], market: list[dict[str, Any]], web: dict[str, Any] | None
) -> tuple[list[Evidence], list[str]]:
    """Build evidence blocks and a list of source problems. Failed calls are not evidence:
    they cannot be cited, the model only mentions them to the user."""
    evidence: list[Evidence] = []
    problems: list[str] = []

    for p in passages:
        evidence.append(
            Evidence(
                p["chunk_id"],
                "letter",
                f"(letter {p['year']}) {p['content']}",
                {"year": p["year"], "passage": p["content"]},
            )
        )
    for m in market:
        data = MarketData.model_validate(m)
        if data.ok:
            evidence.append(Evidence(f"M:{data.ticker}", "market", data.to_evidence_text(), {"title": data.name}))
        else:
            problems.append(f"Market data for {data.ticker} is unavailable: {data.error}")
    if web:
        output = WebSearchOutput.model_validate(web)
        if output.ok:
            for i, r in enumerate(output.results, start=1):
                evidence.append(
                    Evidence(
                        f"W{i}",
                        "web",
                        f"{r.title} ({r.url}): {r.snippet}",
                        {"title": r.title, "url": r.url, "passage": r.snippet},
                    )
                )
        else:
            problems.append(f"Web search failed: {output.error}")
    return evidence, problems


_TEXT_FIXES = str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]", " ": " ", " ": " ", "‑": "-"})


def normalize_text(text: str) -> str:
    """Normalize model output to plain characters.

    Reasoning models sometimes wrap citations in full-width brackets 【M:KO】 and use narrow
    spaces and non-breaking hyphens, which would break citation parsing.
    """
    return text.translate(_TEXT_FIXES)


def extract_citations(text: str, valid_ids: set[str]) -> list[str]:
    """Extract citations like [1988-042] or [1988-042, M:KO], keeping only known ids."""
    found: list[str] = []
    for match in CITATION.finditer(text):
        for part in match.group(1).split(","):
            cid = part.strip()
            if cid in valid_ids and cid not in found:
                found.append(cid)
    return found


def decide_answer(text: str, evidence: list[Evidence]) -> tuple[str, bool, list[Evidence]]:
    """Decide whether the answer is grounded. Returns (text, grounded, sources).

    Grounded means it cites at least one real evidence block and the model did not say
    "not enough information". An answer without source support is discarded.
    """
    stripped = normalize_text(text).strip()
    # Only an answer that STARTS with the refusal phrase counts as a refusal; the phrase in
    # the middle of a normal answer ("the letters don't mention Apple directly, but...") is fine.
    if not stripped or stripped.lower().startswith(REFUSAL_PREFIX):
        return NO_INFO_MESSAGE, False, []
    cited = extract_citations(stripped, {e.id for e in evidence})
    if not cited:
        return NO_INFO_MESSAGE, False, []
    return stripped, True, [e for e in evidence if e.id in cited]


def evidence_to_source(e: Evidence) -> dict[str, Any]:
    """API source format: type, id and what to show the user."""
    source: dict[str, Any] = {"type": e.kind, "id": e.id, "passage": e.meta.get("passage", e.text)}
    source.update({k: v for k, v in e.meta.items() if v is not None and k != "passage"})
    return source


_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STOP = frozenset(
    "that this with from have been were which their there would about they them than then also more most "
    "when what will into such only other over some these those your very much many each both".split()
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201c\[(])")


def best_excerpt(passage: str, answer: str, max_chars: int = 320) -> str:
    """Pick the excerpt of a letter passage that best matches the answer: the sentence sharing the
    most words with it, plus a neighbour if it fits. Showing the chunk start is useless when the
    supporting sentence is in the middle."""
    sentences = [s.strip() for s in _SENTENCE.split(re.sub(r"\s+", " ", passage).strip()) if s.strip()]
    if not sentences:
        return passage[:max_chars]
    wanted = {w for w in _WORD.findall(answer.lower()) if w not in _STOP}
    scores = [len(wanted & set(_WORD.findall(s.lower()))) for s in sentences]
    best = max(range(len(sentences)), key=lambda i: (scores[i], -i))
    chosen = [best]
    for neighbour in (best + 1, best - 1):  # add a neighbour while within the limit
        if (
            0 <= neighbour < len(sentences)
            and sum(len(sentences[i]) + 1 for i in chosen) + len(sentences[neighbour]) <= max_chars
        ):
            chosen.append(neighbour)
            break
    text = " ".join(sentences[i] for i in sorted(chosen))
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def format_history(messages: list[BaseMessage]) -> str:
    """Recent conversation turns (excluding the current question) for the router and synthesizer."""
    lines = []
    for m in messages[-HISTORY_MESSAGES:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {str(m.content)[:600]}")
    return "\n".join(lines) if lines else "(none)"


def _tokens(usage: Usage) -> list[int]:
    return [usage.prompt_tokens or 0, usage.completion_tokens or 0]


# ---------- Nodes ----------


class AgentNodes:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    async def _timed(self, name: str, work: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """Wrap a node: measure its time and record it in the turn bookkeeping."""
        started = time.perf_counter()
        update = await work()
        info = update.setdefault("turn", {}).setdefault(name, {})
        info["ms"] = round((time.perf_counter() - started) * 1000)
        return update

    # --- input guard ---
    async def input_guard(self, state: AgentState) -> dict[str, Any]:
        """Input guard layer 1 (rules). Layer 2 (LLM) is built into the router."""
        verdict = check_input_rules(state["question"])
        if verdict.blocked:
            return {
                "blocked": f"injection:{verdict.rule}",
                "answer": INJECTION_MESSAGE,
                "grounded": False,
                "sources": [],
                "messages": [AIMessage(content=INJECTION_MESSAGE)],
                "turn": {"input_guard": {"rule": verdict.rule}},
            }
        update: dict[str, Any] = {"advice_seeking": verdict.advice_seeking}
        if verdict.advice_seeking:
            update["turn"] = {"input_guard": {"advice_rule": verdict.rule}}
        return update

    def after_input_guard(self, state: AgentState) -> str:
        return "end" if state.get("blocked") else "router"

    async def blocked(self, state: AgentState) -> dict[str, Any]:
        """The router LLM suspected an injection that the rules missed."""
        return {
            "blocked": "injection:llm",
            "answer": INJECTION_MESSAGE,
            "grounded": False,
            "sources": [],
            "messages": [AIMessage(content=INJECTION_MESSAGE)],
        }

    # --- router ---
    async def router(self, state: AgentState) -> dict[str, Any]:
        return await self._timed("router", lambda: self._router(state))

    async def _router(self, state: AgentState) -> dict[str, Any]:
        question = state["question"]
        user = f"Conversation history:\n{format_history(state['messages'][:-1])}\n\nLatest question: {question}"
        # Routing is simple classification: a small model suffices (faster, separate token quota).
        router_model = get_settings().router_model or state.get("model")
        decision, usage = await self.deps.ask_json(router_model, load_prompt("router.md"), user, RouteDecision)
        info: dict[str, Any] = {"tokens": _tokens(usage)}
        if decision is None:
            # Degrade instead of failing: if the router returned garbage, search the letters.
            decision = RouteDecision(
                needs_letters=True, letters_searches=[LetterSearch(query=question)], reason="router fallback"
            )
            info["error"] = "router returned invalid JSON; fell back to letters search"
        advice = bool(state.get("advice_seeking")) or decision.advice_seeking
        return {"route": decision.model_dump(), "advice_seeking": advice, "turn": {"router": info}}

    def dispatch(self, state: AgentState) -> list[str]:
        """Choose the next nodes; several nodes in the list run in parallel."""
        route = RouteDecision.model_validate(state["route"])
        if route.injection_suspected:
            return ["blocked"]
        if not route.in_scope:
            return ["out_of_scope"]
        targets = []
        if route.needs_letters:
            targets.append("letters")
        if route.needs_market and route.tickers:
            targets.append("market")
        if route.needs_web:
            targets.append("web")
        return targets or ["synthesize"]  # no sources needed: the synthesizer will say there is no data

    # --- specialists ---
    async def letters(self, state: AgentState) -> dict[str, Any]:
        return await self._timed("letters", lambda: self._letters(state))

    async def _letters(self, state: AgentState) -> dict[str, Any]:
        route = RouteDecision.model_validate(state["route"])
        searches = route.letters_searches[:MAX_LETTER_SEARCHES] or [LetterSearch(query=state["question"])]

        async def one(search: LetterSearch) -> list[Retrieved]:
            years = [y for y in search.years if 1977 <= y <= 2024]
            return await self.deps.retrieve(
                self.deps.pool, search.query, k=PASSAGES_PER_SEARCH, mode="hybrid", years=years or None
            )

        # Multiple searches (e.g. "airlines" and "banks in 1990") run concurrently.
        results = await asyncio.gather(*(one(s) for s in searches), return_exceptions=True)
        passages: list[dict[str, Any]] = []
        seen: set[str] = set()
        errors = 0
        for result in results:
            if isinstance(result, BaseException):
                errors += 1
                continue
            for h in result:
                if h.chunk_id not in seen:  # the same chunk may be found by several queries
                    seen.add(h.chunk_id)
                    passages.append(
                        {
                            "chunk_id": h.chunk_id,
                            "year": h.year,
                            "content": h.content,
                            "similarity": round(h.similarity, 3),
                        }
                    )
        info: dict[str, Any] = {"found": len(passages), "searches": len(searches)}
        if errors:
            info["error"] = f"{errors} of {len(searches)} letter searches failed"
        return {"passages": passages, "turn": {"letters": info}}

    async def market(self, state: AgentState) -> dict[str, Any]:
        return await self._timed("market", lambda: self._market(state))

    async def _market(self, state: AgentState) -> dict[str, Any]:
        tickers = [t.strip() for t in state["route"].get("tickers", []) if t.strip()][:MAX_TICKERS]
        # Tickers are independent, so fetch them concurrently.
        results = await asyncio.gather(*(self.deps.get_metrics(t) for t in tickers), return_exceptions=True)
        market: list[dict[str, Any]] = []
        for ticker, result in zip(tickers, results, strict=True):
            if isinstance(result, BaseException):  # get_company_metrics should not raise; defensive
                result = MarketData(ok=False, ticker=ticker.upper(), error="Unexpected tool failure.")
            market.append(result.model_dump())
        failed = [m["ticker"] for m in market if not m["ok"]]
        return {"market": market, "turn": {"market": {"error": f"no data for {failed}"} if failed else {}}}

    async def web(self, state: AgentState) -> dict[str, Any]:
        return await self._timed("web", lambda: self._web(state))

    async def _web(self, state: AgentState) -> dict[str, Any]:
        query = state["route"].get("web_query") or state["question"]
        try:
            output = await self.deps.search_web(query)
        except Exception as exc:
            return {"web": None, "turn": {"web": {"error": f"web search failed: {type(exc).__name__}"}}}
        return {"web": output.model_dump(), "turn": {"web": {} if output.ok else {"error": output.error}}}

    # --- terminal nodes ---
    async def out_of_scope(self, state: AgentState) -> dict[str, Any]:
        return {
            "answer": OUT_OF_SCOPE_MESSAGE,
            "grounded": False,
            "sources": [],
            "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)],
        }

    async def synthesize(self, state: AgentState) -> dict[str, Any]:
        return await self._timed("synthesize", lambda: self._synthesize(state))

    async def _synthesize(self, state: AgentState) -> dict[str, Any]:
        evidence, problems = build_evidence(state.get("passages", []), state.get("market", []), state.get("web"))

        if not evidence:
            # No evidence: skip the LLM entirely (cheaper and rules out fabrication).
            answer = NO_INFO_MESSAGE if not problems else f"{NO_INFO_MESSAGE} ({'; '.join(problems)})"
            return {"answer": answer, "grounded": False, "sources": []}

        blocks = "\n\n".join(f"[{e.id}] {e.text}" for e in evidence)
        notes = (
            ("\n\nData problems (not citable, mention briefly if relevant):\n- " + "\n- ".join(problems))
            if problems
            else ""
        )
        advice_note = (
            "\n\nNote: the user is asking for investment advice or a forecast. Do NOT give any recommendation, "
            "price target or forecast. Say briefly that you cannot, then describe what Buffett's principles "
            "and the evidence show."
            if state.get("advice_seeking")
            else ""
        )
        user = (
            f"Conversation so far:\n{format_history(state['messages'][:-1])}\n\n"
            f"Evidence:\n\n{blocks}{notes}\n\nQuestion: {state['question']}{advice_note}"
        )
        text, usage = await self.deps.generate(
            state.get("model"),
            [SystemMessage(content=load_prompt("synthesizer.md")), HumanMessage(content=user)],
            temperature=SYNTH_TEMPERATURE,
        )
        answer, grounded, cited = decide_answer(text, evidence)
        info: dict[str, Any] = {"tokens": _tokens(usage)}
        if not grounded:
            info["raw_text"] = text[:300]  # what the model actually said (for debugging refusals)
        return {
            "answer": answer,
            "grounded": grounded,
            "sources": [
                {**evidence_to_source(e), "excerpt": best_excerpt(e.meta.get("passage", e.text), answer)} for e in cited
            ],
            "turn": {"synthesize": info},
        }

    # --- faithfulness guard ---
    async def faithfulness(self, state: AgentState) -> dict[str, Any]:
        """Remove sentences with unsupported numbers or citations from a grounded answer."""
        if not state.get("grounded"):
            return {}
        evidence, _ = build_evidence(state.get("passages", []), state.get("market", []), state.get("web"))
        result = check_faithfulness(state["answer"], [e.text for e in evidence], {e.id for e in evidence})
        if not result.removed:
            return {}
        info = {"removed": result.removed}
        if not result.grounded:
            # Nothing source-backed is left: better to refuse.
            return {"answer": NO_INFO_MESSAGE, "grounded": False, "sources": [], "turn": {"faithfulness": info}}
        kept_sources = [s for s in state.get("sources", []) if s["id"] in result.cited_ids]
        return {"answer": result.text, "sources": kept_sources, "turn": {"faithfulness": info}}

    # --- output guard ---
    async def output_guard(self, state: AgentState) -> dict[str, Any]:
        """Last step: no recommendation or price forecast reaches the user unchecked."""
        text, violations, action, usage = await apply_output_guard(
            state["answer"], self.deps.generate, state.get("model")
        )
        update: dict[str, Any] = {
            "answer": text,
            "messages": [AIMessage(content=text)],  # history gets only the final text
            "turn": {"output_guard": {"action": action, "violations": violations, "tokens": _tokens(usage)}},
        }
        if action == "replaced":
            update.update({"grounded": False, "sources": []})
        elif action == "rewritten":
            cited = extract_citations(text, {s["id"] for s in state.get("sources", [])})
            update["sources"] = [s for s in state.get("sources", []) if s["id"] in cited]
            update["grounded"] = bool(cited)
        return update
