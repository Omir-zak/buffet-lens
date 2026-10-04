"""LLM client for any OpenAI-compatible provider.

Nothing here is Groq-specific: switching to OpenRouter, Gemini or Ollama only requires
changing LLM_BASE_URL, LLM_API_KEY and the model name.
"""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file from the prompts directory."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __add__(self, other: "Usage") -> "Usage":
        """Sum token usage across several model calls (None counts as zero)."""
        return Usage(
            (self.prompt_tokens or 0) + (other.prompt_tokens or 0),
            (self.completion_tokens or 0) + (other.completion_tokens or 0),
        )


def build_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """Build a chat client for a given model. Cheap to create, so done per request."""
    settings = get_settings()
    extra: dict[str, Any] = {} if temperature is None else {"temperature": temperature}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(
        model=model or settings.default_chat_model,
        api_key=SecretStr(settings.llm_api_key),
        base_url=settings.llm_base_url,
        max_tokens=max_tokens or settings.max_output_tokens,  # type: ignore[call-arg]  # alias in ChatOpenAI
        timeout=settings.request_timeout_seconds,
        max_retries=4,  # retry transient failures and 429s with backoff
        stream_usage=True,  # ask the provider to report token usage when streaming too
        **extra,
    )


def build_messages(history: list[dict[str, str]], user_text: str) -> list[BaseMessage]:
    """Assemble the full conversation; the model is stateless, so history is resent every time."""
    messages: list[BaseMessage] = [SystemMessage(content=load_prompt("system.md"))]
    for item in history:
        cls = HumanMessage if item["role"] == "user" else AIMessage
        messages.append(cls(content=item["content"]))
    messages.append(HumanMessage(content=user_text))
    return messages


def _usage_of(message: AIMessage | AIMessageChunk) -> Usage:
    meta: Mapping[str, Any] = message.usage_metadata or {}
    return Usage(meta.get("input_tokens"), meta.get("output_tokens"))


async def complete(
    model: str | None, messages: list[BaseMessage], temperature: float | None = None
) -> tuple[str, Usage]:
    """Non-streaming request: wait for the full reply."""
    response = await build_chat_model(model, temperature).ainvoke(messages)
    return str(response.content), _usage_of(response)


async def stream(model: str | None, messages: list[BaseMessage], usage_out: list[Usage]) -> AsyncIterator[str]:
    """Stream text chunks as they are generated.

    Token usage arrives at the very end of the stream, so it is appended to `usage_out`.
    """
    total: AIMessageChunk | None = None
    async for chunk in build_chat_model(model).astream(messages):
        # Chunks are additive, which accumulates the final message together with usage.
        total = chunk if total is None else total + chunk
        if chunk.content:
            yield str(chunk.content)
    if total is not None:
        usage_out.append(_usage_of(total))
