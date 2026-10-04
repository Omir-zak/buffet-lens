"""Request/response models with validation (invalid requests get a 422 from FastAPI)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: UUID | None = Field(
        default=None, description="Id of an existing conversation. A new one is created if omitted."
    )
    model: str | None = Field(
        default=None, description="Provider model, e.g. openai/gpt-oss-120b. Defaults to the configured model."
    )
    message: str = Field(min_length=1, max_length=4000)


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class SourceOut(BaseModel):
    """A source backing the answer: a letter, market data or a web page."""

    type: str  # "letter" | "market" | "web"
    id: str  # letters: chunk_id (e.g. 1988-042), market: M:KO, web: W1
    passage: str | None = None  # the whole retrieved chunk
    excerpt: str | None = None  # the part of it most relevant to the answer
    chunk_id: str | None = None  # same as id, for letters (field name required by the spec)
    year: int | None = None  # letter year
    title: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    model: str
    answer: str
    grounded: bool
    sources: list[SourceOut]
    tools_used: list[str]
    usage: TokenUsage
    guardrails: dict | None = None  # which guardrails fired (input rules, answer rewrite, ...)


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    model: str | None = None
    k: int = Field(default=5, ge=1, le=10, description="How many letter passages to retrieve")
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"


class Source(BaseModel):
    year: int
    passage: str
    chunk_id: str


class RagResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]
    top_similarity: float
    reason: str
    usage: TokenUsage


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: UUID
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationDetail(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]
