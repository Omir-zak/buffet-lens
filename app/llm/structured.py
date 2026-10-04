"""Model requests that must return JSON validated against a Pydantic schema.

Models sometimes wrap JSON in prose or a ``` block, or get fields wrong, so the reply is always
parsed and validated, with one retry that restates the expected format.
"""

import json
import re
from collections.abc import Mapping
from typing import Any, TypeVar

import openai
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.llm.client import Usage, build_chat_model

T = TypeVar("T", bound=BaseModel)


def parse_json_model(raw: str, schema: type[T]) -> T | None:
    """Extract the first JSON object from text and validate it; None on failure."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return schema.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError):
        return None


async def ask_json(
    model: str | None,
    system: str,
    user: str,
    schema: type[T],
    temperature: float = 0.0,
    retries: int = 1,
    max_tokens: int = 900,
) -> tuple[T | None, Usage]:
    """Ask the model; return (schema object or None, total token usage).

    Provider exceptions (rate limit, timeout) are not caught; the caller handles them.
    """
    # Keep max_tokens small: providers count it against the tokens-per-minute limit and the JSON is short.
    effort = get_settings().simple_step_reasoning_effort or None
    llm = build_chat_model(model, temperature, max_tokens, effort).bind(response_format={"type": "json_object"})
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    total = Usage()
    for attempt in range(retries + 1):
        try:
            response = await llm.ainvoke(messages)
        except openai.BadRequestError as exc:
            # The provider failed to get valid JSON (e.g. the model exhausted its reasoning budget).
            # That is a format failure, not a user error: treat it as an attempt with no result.
            if "json_validate_failed" not in str(exc):
                raise
            if attempt < retries:
                continue
            return None, total
        meta: Mapping[str, Any] = response.usage_metadata or {}
        total = total + Usage(meta.get("input_tokens"), meta.get("output_tokens"))
        parsed = parse_json_model(str(response.content), schema)
        if parsed is not None:
            return parsed, total
        if attempt < retries:
            messages = [
                *messages[:2],
                HumanMessage(
                    content="Your previous reply was not a valid JSON object in the required shape. "
                    "Reply again with ONLY the JSON object."
                ),
            ]
    return None, total
