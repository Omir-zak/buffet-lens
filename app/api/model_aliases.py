"""Map frontend model names to models of our provider (Groq).

The frontend lists OpenAI, Anthropic and Google models; instead of an "unknown model" error we
substitute a comparable one. The table lives in settings (config.py, MODEL_ALIASES).
"""

from app.config import get_settings


def resolve_model(requested: str | None) -> str:
    """Provider model name. Empty: the default model. Unknown names pass through unchanged."""
    settings = get_settings()
    if not requested:
        return settings.default_chat_model
    return settings.model_aliases.get(requested, requested)
