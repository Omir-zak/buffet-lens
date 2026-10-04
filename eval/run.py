"""Unified system evaluation:  python -m eval.run

Prints all the figures from section 4 of the spec and saves them to eval/results.json.
Parts can be run individually:  python -m eval.run --parts retrieval answers
Parts: retrieval (recall@5, MRR, chunking comparison), answers (answer quality on 40 questions),
       guardrails (guardrails on adversarial prompts), cost (latency, tokens and cost, two models).

Reading the results:
  * the Groq free tier limits tokens per minute and retries after 429 errors inflate latency.
    Latencies are reported as-is; on a paid tier they would be noticeably lower;
  * cost uses assumed list prices (eval/prices.json); real spend on the free tier is $0.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from app.agents.deps import Deps
from app.agents.graph import build_graph
from app.agents.runner import AgentResult, _initial_input, result_from_state
from app.config import get_settings
from app.db.database import DictPool, create_pool
from app.observability import flush, make_callbacks, tracing_enabled
from app.rag.retrieval import Mode
from eval import guardrails_eval
from eval.progress import Progress
from eval.ratelimit import QuotaExhausted, with_rate_limit_retry
from eval.retrieval_eval import evaluate_retrieval, load_questions

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "results.json"
PRICES = {
    k: v for k, v in json.loads((EVAL_DIR / "prices.json").read_text(encoding="utf-8")).items() if not k.startswith("_")
}


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """Markdown table (handy for pasting into the README)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def cost_usd(result: AgentResult, router_model: str, answer_model: str) -> float:
    """Cost of one request at list price: the router at the small model's price, the rest at the main model's."""
    total = 0.0
    for node, (prompt, completion) in result.tokens_by_node.items():
        price_in, price_out = PRICES.get(router_model if node == "router" else answer_model, (0.0, 0.0))
        total += (prompt * price_in + completion * price_out) / 1_000_000
    return total


async def run_question(
    graph: Any, question: str, model: str | None = None, session_id: str | None = None
) -> tuple[AgentResult, float]:
    """Run a question through the graph. With a session_id and Langfuse enabled it shows up there as a trace."""
    callbacks = make_callbacks(session_id, question) if session_id else []
    started = time.perf_counter()
    state = await with_rate_limit_retry(
        lambda: graph.ainvoke(_initial_input(question, model), {"callbacks": callbacks} if callbacks else None)
    )
    elapsed = time.perf_counter() - started
    flush(callbacks)
    return result_from_state(state), elapsed


def langfuse_trace(session_id: str) -> dict[str, Any] | None:
    """Langfuse trace for one request (each request is its own session). None if Langfuse is unavailable."""
    if not tracing_enabled():
        return None
    settings = get_settings()
    try:
        data = httpx.get(
            f"{settings.langfuse_host.rstrip('/')}/api/public/traces",
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=30,
            params={"sessionId": session_id, "limit": 5},
        ).json()
    except Exception:
        return None
    traces = data.get("data", [])
    return traces[0] if traces else None


def langfuse_cost_stats(session_prefix: str, question_ids: list[str]) -> dict[str, Any] | None:
    """Average and p95 latency and average cost from Langfuse's own data for a set of requests."""
    traces = [t for t in (langfuse_trace(f"{session_prefix}{qid}") for qid in question_ids) if t]
    if not traces:
        return None
    latencies = [t["latency"] for t in traces if t.get("latency") is not None]
    costs = [t.get("totalCost") or 0.0 for t in traces]
    return {
        "traces": len(traces),
        "avg_s": statistics.mean(latencies),
        "p95_s": percentile(latencies, 95),
        "avg_cost_usd": statistics.mean(costs),
    }


# ---------------------------------------------------------------- retrieval
async def part_retrieval(pool: DictPool) -> dict[str, Any]:
    print("\n## 1. Качество поиска (20 вопросов по письмам, k=5)\n")
    rows, data = [], {}
    for strategy in ("fixed_1000", "recursive_1000"):
        combos: list[tuple[Mode, bool | None]] = [
            ("vector", None),
            ("keyword", False),
            ("keyword", True),
            ("hybrid", False),
            ("hybrid", True),
        ]
        for mode, idf in combos:
            s = await evaluate_retrieval(pool, strategy, mode, idf_filter=True if idf is None else idf)
            idf_label = "-" if idf is None else ("вкл" if idf else "выкл")
            rows.append(
                [
                    strategy,
                    mode,
                    idf_label,
                    f"{s.recall_year:.2f}",
                    f"{s.mrr_year:.2f}",
                    f"{s.recall_chunk:.2f}",
                    f"{s.mrr_chunk:.2f}",
                ]
            )
            data[f"{strategy}/{mode}/idf_{idf_label}"] = vars(s)
    print(
        table(
            ["нарезка", "поиск", "учёт редкости слов", "recall@5 (год)", "MRR (год)", "recall@5 (чанк)", "MRR (чанк)"],
            rows,
        )
    )
    return data


# ------------------------------------------------------------------ answers
def judge_answer(q: dict[str, Any], r: AgentResult) -> dict[str, Any]:
    kind = q["type"]
    letter_years = {s["year"] for s in r.sources if s["type"] == "letter"}
    if kind == "letters":
        attributed = r.grounded and bool(letter_years & set(q["expected_years"]))
        return {"ok": attributed, "grounded": r.grounded}
    if kind == "live":
        return {
            "ok": r.grounded and "market_data" in r.tools_used and "letters" not in r.tools_used,
            "grounded": r.grounded,
        }
    if kind == "mixed":
        return {"ok": r.grounded and {"market_data", "letters"} <= set(r.tools_used), "grounded": r.grounded}
    return {"ok": not r.grounded, "grounded": r.grounded}  # unanswerable: the correct outcome is a refusal


async def part_answers(graph: Any, fresh: bool = False) -> dict[str, Any]:
    print("\n## 2. Качество ответов (40 вопросов, полный конвейер)\n")
    progress = Progress("answers", fresh)
    per_type: dict[str, list[dict[str, Any]]] = {}
    failures = []
    for q in load_questions():
        if q["id"] not in progress:
            result, _ = await run_question(graph, q["question"], session_id=f"eval-answers-{q['id']}")
            verdict = judge_answer(q, result)
            progress.put(
                q["id"],
                {
                    **verdict,
                    "type": q["type"],
                    "question": q["question"][:70],
                    "tools": result.tools_used,
                    "answer": result.answer[:100],
                },
            )
        saved = progress.get(q["id"])
        per_type.setdefault(q["type"], []).append(saved)
        if not saved["ok"]:
            failures.append((q["id"], saved["question"], saved["grounded"], saved["tools"], saved["answer"]))
        print(".", end="", flush=True)
    print()
    labels = {
        "letters": "по письмам: ответ обоснован и указан верный год письма",
        "live": "живые данные: использован yfinance, не письма",
        "mixed": "смешанные: использованы и письма, и yfinance",
        "unanswerable": "без ответа: корректный отказ (grounded=false)",
    }
    rows = [[labels[t], f"{sum(v['ok'] for v in vs)}/{len(vs)}"] for t, vs in per_type.items()]
    print(table(["тип вопроса", "верно"], rows))
    if failures:
        print("\nПровалы (разбираем, а не прячем):")
        for f in failures:
            print(f"  x [{f[0]}] {f[1]!r} grounded={f[2]} tools={f[3]}\n      -> {f[4]!r}")
    return {"per_type": {t: [sum(v["ok"] for v in vs), len(vs)] for t, vs in per_type.items()}, "failures": failures}


# --------------------------------------------------------------- guardrails
async def part_guardrails(graph: Any, fresh: bool = False) -> dict[str, Any]:
    print("\n## 3. Защитные барьеры\n")
    progress = Progress("guardrails", fresh)
    data = {}
    for label, file in (("dev", "guardrails.jsonl"), ("heldout", "guardrails_heldout.jsonl")):
        results = []
        for row in guardrails_eval.load_rows(file):
            key = f"{label}:{row['id']}"
            if key not in progress:
                progress.put(key, await guardrails_eval.evaluate_row(graph, row))
            results.append(progress.get(key))
            print(".", end="", flush=True)
        print()
        guardrails_eval.summarize(label, results)
        data[label] = [{k: v for k, v in r.items() if k in ("id", "category", "ok", "output_action")} for r in results]
    return data


# --------------------------------------------------------------------- cost
async def part_cost(pool: DictPool, sample_size: int = 30, fresh: bool = False) -> dict[str, Any]:
    print(f"\n## 4. Задержка, токены и стоимость ({sample_size} запросов, две модели ответа)\n")
    settings = get_settings()
    questions = [q for q in load_questions() if q["type"] != "unanswerable"][:sample_size]
    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    graph = build_graph(Deps(pool=pool))
    progress = Progress("cost", fresh)
    if "_run_id" not in progress:
        progress.put("_run_id", uuid.uuid4().hex[:8])  # same id on resume so Langfuse sessions stay stable
    run_id = progress.get("_run_id")
    rows, data = [], {}
    for model in models:
        latencies, tokens, costs = [], [], []
        session = f"eval-cost-{run_id}-{model.split('/')[-1]}"
        for q in questions:
            key = f"{model}:{q['id']}"
            if key not in progress:
                result, seconds = await run_question(graph, q["question"], model, session_id=f"{session}-{q['id']}")
                progress.put(
                    key,
                    {
                        "seconds": seconds,
                        "tokens": (result.usage.prompt_tokens or 0) + (result.usage.completion_tokens or 0),
                        "cost": cost_usd(result, settings.router_model or model, model),
                    },
                )
            saved = progress.get(key)
            latencies.append(saved["seconds"])
            tokens.append(saved["tokens"])
            costs.append(saved["cost"])
            print(".", end="", flush=True)
        print()
        await asyncio.sleep(8)  # Langfuse ingests data asynchronously
        lf = langfuse_cost_stats(f"{session}-", [q["id"] for q in questions])
        rows.append(
            [
                model,
                f"{statistics.mean(latencies):.1f}",
                f"{percentile(latencies, 95):.1f}",
                f"{statistics.mean(tokens):.0f}",
                f"{statistics.mean(costs) * 1000:.3f}",
                "н/д" if lf is None else f"{lf['avg_s']:.1f} / {lf['p95_s']:.1f}",
                "н/д" if lf is None else f"{lf['avg_cost_usd'] * 1000:.3f}",
            ]
        )
        data[model] = {
            "avg_s": statistics.mean(latencies),
            "p95_s": percentile(latencies, 95),
            "avg_tokens": statistics.mean(tokens),
            "avg_cost_usd": statistics.mean(costs),
            "n": len(questions),
            "langfuse": lf,
        }
    print(
        table(
            [
                "модель ответа",
                "задержка avg, с",
                "p95, с",
                "токенов/запрос",
                "стоимость x1000 $ (локально)",
                "Langfuse: avg / p95, с",
                "Langfuse: стоимость x1000 $",
            ],
            rows,
        )
    )
    return data


async def main(parts: list[str], fresh: bool = False) -> None:
    pool = create_pool(get_settings().database_url)
    await pool.open()
    graph = build_graph(Deps(pool=pool))
    results: dict[str, Any] = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {}
    runners: dict[str, Any] = {
        "retrieval": lambda: part_retrieval(pool),
        "answers": lambda: part_answers(graph, fresh),
        "guardrails": lambda: part_guardrails(graph, fresh),
        "cost": lambda: part_cost(pool, fresh=fresh),
    }
    failed: list[str] = []
    try:
        for name in (n for n in runners if n in parts):
            try:
                results[name] = await runners[name]()
            except QuotaExhausted as exc:
                # Limit exhausted for long: this part is unfinished, continue with the rest and rerun it later.
                failed.append(name)
                print(f"\n[часть {name} остановлена: {str(exc)[:160]}]")
            finally:
                # Save after EACH part so an hour-long run isn't lost to a late failure.
                RESULTS_PATH.write_bytes(json.dumps(results, ensure_ascii=False, indent=1).encode("utf-8"))
    finally:
        await pool.close()
    print(f"\nРезультаты сохранены в {RESULTS_PATH.relative_to(EVAL_DIR.parent)}")
    if failed:
        raise SystemExit(
            f"Не завершены части: {', '.join(failed)}. Запустите: python -m eval.run --parts {' '.join(failed)}"
        )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["retrieval", "answers", "guardrails", "cost"],
        choices=["retrieval", "answers", "guardrails", "cost"],
    )
    parser.add_argument("--fresh", action="store_true", help="не использовать сохранённый прогресс, начать с нуля")
    args = parser.parse_args()
    asyncio.run(main(args.parts, args.fresh))
