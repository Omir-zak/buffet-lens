"""Routing accuracy: does the router pick the right sources for the questions in questions.jsonl.

Run: python -m eval.router_eval [model ...]   (defaults to ROUTER_MODEL from settings)
The expected route follows from the question type:
  letters      letters needed, no live data;
  live         market data needed (with a ticker), no letters;
  mixed        both letters and market data;
  unanswerable on-topic, no market data; the two clearly unrelated questions must get in_scope=False.
"""

import asyncio
import sys
import time
from functools import partial

from app.agents.nodes import RouteDecision
from app.config import get_settings
from app.llm.client import load_prompt
from app.llm.structured import ask_json
from eval.ratelimit import with_rate_limit_retry
from eval.retrieval_eval import load_questions

OFF_TOPIC_STARTS = ("Who is going to win", "What is the best recipe")


def route_is_correct(question: dict, decision: RouteDecision | None) -> bool:
    if decision is None:
        return False
    kind = question["type"]
    if kind == "letters":
        return decision.in_scope and decision.needs_letters and not decision.needs_market
    if kind == "live":
        return decision.in_scope and decision.needs_market and bool(decision.tickers) and not decision.needs_letters
    if kind == "mixed":
        return decision.in_scope and decision.needs_letters and decision.needs_market and bool(decision.tickers)
    if question["question"].startswith(OFF_TOPIC_STARTS):
        return not decision.in_scope
    return decision.in_scope and not decision.needs_market


async def evaluate_router(model: str) -> None:
    questions = load_questions()
    correct, tokens, wrong = 0, 0, []
    started = time.perf_counter()
    for q in questions:
        user = f"Conversation history:\n(none)\n\nLatest question: {q['question']}"
        decision, usage = await with_rate_limit_retry(
            partial(ask_json, model, load_prompt("router.md"), user, RouteDecision)
        )
        tokens += (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        if route_is_correct(q, decision):
            correct += 1
        else:
            wrong.append(
                (
                    q["id"],
                    q["question"][:60],
                    None
                    if decision is None
                    else (decision.in_scope, decision.needs_letters, decision.needs_market, decision.tickers),
                )
            )
    print(
        f"{model}: правильный маршрут {correct}/{len(questions)} | {time.perf_counter() - started:.0f} с | токенов {tokens}"
    )
    for item in wrong:
        print("   x", item)


async def main(models: list[str]) -> None:
    for model in models:
        await evaluate_router(model)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(sys.argv[1:] or [get_settings().router_model]))
