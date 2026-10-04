"""Application settings loaded from environment variables and .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Chat: any OpenAI-compatible provider (Groq by default).
    llm_api_key: str
    llm_base_url: str = "https://api.groq.com/openai/v1"
    default_chat_model: str = "openai/gpt-oss-120b"

    # Embeddings use a separate provider because Groq has none.
    # Defaults to local Ollama (no key needed, but the client requires a non-empty string).
    embedding_api_key: str = "ollama"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"
    # Vector size is fixed by the model (nomic-embed-text = 768) and baked into the table schema.
    # If you switch to a model with another dimension, recreate the chunks and embedding_cache tables.
    embedding_dim: int = 768
    # nomic-embed-text was trained with prefixes: documents and queries are tagged differently,
    # and retrieval quality drops noticeably without them. Other models usually need none.
    embedding_doc_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "

    # Chunking strategy used for retrieval (see app/rag/chunking.py).
    active_chunk_strategy: str = "recursive_1000"

    # "Grounded" threshold: if the best chunk is less similar to the question than this, answer
    # that the letters don't cover it. Tuned on the eval set (see eval/), not by eye.
    rag_min_similarity: float = 0.5

    # Generation limits. Reasoning models (gpt-oss) spend part of the budget on reasoning,
    # so max_output_tokens needs headroom or the answer gets truncated.
    max_output_tokens: int = 2000
    request_timeout_seconds: float = 60.0
    # Model for simple steps (routing, eval judge). Smaller and faster than the main one, and Groq
    # rate limits are per model, so load is split across two buckets.
    # An empty string means "use the same model as for answers".
    router_model: str = "openai/gpt-oss-20b"
    # Reasoning effort for simple steps (low/medium/high). Otherwise a reasoning model burns hundreds
    # of tokens before a short JSON. Empty = don't pass the parameter to the provider.
    simple_step_reasoning_effort: str = "low"
    # Number of most recent conversation messages sent to the model.
    history_limit: int = 20

    # API key. If set, all /v1 endpoints require an x-api-key header with this value.
    # An empty key leaves the API open (handy for local development and tests).
    api_key: str = ""
    # Model names sent by the frontend (its dropdown lists OpenAI, Anthropic and Google models)
    # are mapped to our provider's models. Unknown names pass through as-is, so real provider
    # ids work too. Can be overridden with JSON in MODEL_ALIASES.
    model_aliases: dict[str, str] = {
        "gpt-4o-mini": "openai/gpt-oss-20b",
        "claude-haiku-4-5": "openai/gpt-oss-20b",
        "gemini-3.1-flash-lite-preview": "openai/gpt-oss-20b",
        "gemini-2.5-flash-lite": "openai/gpt-oss-20b",
        "gpt-4o": "openai/gpt-oss-120b",
        "claude-sonnet-4-6": "openai/gpt-oss-120b",
    }

    database_url: str = "postgresql://buffett:buffett@localhost:5433/buffett"

    # Web search: use Tavily if a real key is set, otherwise DuckDuckGo (no key).
    tavily_api_key: str = ""
    # External sources (yfinance, web) are flaky: cap wait time and cache responses.
    tool_timeout_seconds: float = 15.0
    tool_cache_seconds: float = 600.0

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Env var name = upper-cased field name (LLM_API_KEY).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # required fields come from the environment and .env
