"""Answer a question from the letters: retrieve passages, decide if they suffice, generate, verify.

The "grounded" decision is made in three stages, cheapest first:
  1. Similarity threshold: if even the best passage is weakly related to the question, the letters
     most likely don't answer it, and the LLM is not called at all.
  2. LLM judgment: the model reports whether the passages can answer the question (answerable).
  3. Citation check: the answer counts as grounded only if it cites passages that were actually
     retrieved. A plausible answer that cites nothing is discarded.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.db.database import DictPool
from app.llm.client import Usage, build_chat_model, load_prompt
from app.rag.retrieval import Mode, Retrieved, retrieve

NOT_ENOUGH_INFO = "The letters do not contain enough information to answer this."


class LLMAnswer(BaseModel):
    """JSON shape the model must return."""

    answerable: bool
    answer: str
    used_ids: list[str] = []


@dataclass
class RagResult:
    answer: str
    grounded: bool
    sources: list[Retrieved] = field(default_factory=list)
    top_similarity: float = 0.0
    reason: str = ""  # why this decision was made (useful for debugging and eval)
    usage: Usage = field(default_factory=Usage)


def _format_context(passages: list[Retrieved]) -> str:
    return "\n\n".join(f"[{p.chunk_id}] (letter {p.year}) {p.content}" for p in passages)


def parse_llm_answer(raw: str) -> LLMAnswer | None:
    """Extract JSON from a model reply, which may be wrapped in a ```json block or prose."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return LLMAnswer.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


def decide_grounding(parsed: LLMAnswer, passages: list[Retrieved]) -> tuple[bool, list[Retrieved]]:
    """Pure function (no network or DB). Returns (grounded, sources).

    An answer is grounded if the model answered substantively AND cited at least one
    passage that was actually retrieved.
    """
    by_id = {p.chunk_id: p for p in passages}
    cited = [by_id[i] for i in dict.fromkeys(parsed.used_ids) if i in by_id]  # dedupe, keep order
    if not parsed.answerable or not cited:
        return False, []
    return True, cited


async def answer_question(
    pool: DictPool,
    question: str,
    model: str | None = None,
    k: int = 5,
    mode: Mode = "hybrid",
) -> RagResult:
    settings = get_settings()
    passages = await retrieve(pool, question, k=k, mode=mode)
    top = max((p.similarity for p in passages), default=0.0)

    # Stage 1: similarity threshold.
    if not passages or top < settings.rag_min_similarity:
        return RagResult(NOT_ENOUGH_INFO, False, [], top, reason=f"top similarity {top:.2f} below threshold")

    # Stage 2: generation. response_format=json_object asks the provider for valid JSON.
    messages = [
        SystemMessage(content=load_prompt("rag_answer.md")),
        HumanMessage(content=f"Passages:\n\n{_format_context(passages)}\n\nQuestion: {question}"),
    ]
    llm = build_chat_model(model).bind(response_format={"type": "json_object"})
    response = await llm.ainvoke(messages)
    meta: Mapping[str, Any] = response.usage_metadata or {}
    usage = Usage(meta.get("input_tokens"), meta.get("output_tokens"))

    parsed = parse_llm_answer(str(response.content))
    if parsed is None:
        return RagResult(NOT_ENOUGH_INFO, False, [], top, reason="model returned invalid JSON", usage=usage)

    # Stage 3: citation check.
    grounded, sources = decide_grounding(parsed, passages)
    if not grounded:
        reason = "model said not answerable" if not parsed.answerable else "answer cites no retrieved passage"
        return RagResult(NOT_ENOUGH_INFO, False, [], top, reason=reason, usage=usage)
    return RagResult(parsed.answer, True, sources, top, reason="ok", usage=usage)
