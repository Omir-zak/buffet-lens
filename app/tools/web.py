"""Web search tool: Tavily (if a key is set) or DuckDuckGo (no key).

Search results are untrusted internet data and may contain "instructions" aimed at the model.
They are passed on as facts to cite, never as commands (see the synthesizer prompt).
"""

import asyncio

import httpx
from ddgs import DDGS
from pydantic import BaseModel

from app.config import get_settings
from app.tools.cache import TTLCache

_cache = TTLCache(get_settings().tool_cache_seconds)


class WebResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchOutput(BaseModel):
    ok: bool
    query: str
    provider: str
    results: list[WebResult] = []
    error: str | None = None


def _has_real_tavily_key(key: str) -> bool:
    return bool(key) and not key.endswith("...")  # the .env.example placeholder doesn't count


async def _search_tavily(query: str, max_results: int) -> list[WebResult]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.tool_timeout_seconds) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.tavily_api_key, "query": query, "max_results": max_results},
        )
        response.raise_for_status()
    return [
        WebResult(title=r.get("title", ""), url=r.get("url", ""), snippet=(r.get("content") or "")[:500])
        for r in response.json().get("results", [])
    ]


def _search_ddgs(query: str, max_results: int) -> list[WebResult]:
    """Blocking library, so it must be called in a thread."""
    return [
        WebResult(title=r.get("title", ""), url=r.get("href", ""), snippet=(r.get("body") or "")[:500])
        for r in DDGS().text(query, max_results=max_results)
    ]


async def search_web(query: str, max_results: int = 5) -> WebSearchOutput:
    """Search the web for recent news and context.

    Args:
        query: English search query, e.g. "Coca-Cola Q2 2026 earnings".
        max_results: number of results to return (1-10).

    Returns:
        WebSearchOutput. On error or timeout, ok=False and error explains why.
        Never raises.
    """
    settings = get_settings()
    query = query.strip()
    max_results = max(1, min(max_results, 10))
    if not query:
        return WebSearchOutput(ok=False, query=query, provider="none", error="Empty search query.")

    provider = "tavily" if _has_real_tavily_key(settings.tavily_api_key) else "duckduckgo"
    cache_key = f"{provider}|{max_results}|{query.lower()}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        if provider == "tavily":
            results = await asyncio.wait_for(_search_tavily(query, max_results), settings.tool_timeout_seconds)
        else:
            results = await asyncio.wait_for(
                asyncio.to_thread(_search_ddgs, query, max_results), settings.tool_timeout_seconds
            )
    except TimeoutError:
        return WebSearchOutput(ok=False, query=query, provider=provider, error="Web search timed out.")
    except Exception as exc:
        return WebSearchOutput(
            ok=False, query=query, provider=provider, error=f"Web search is unavailable ({type(exc).__name__})."
        )

    output = WebSearchOutput(ok=True, query=query, provider=provider, results=results)
    if results:
        _cache.set(cache_key, output)
    return output
