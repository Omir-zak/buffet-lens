"""Graph dependencies: everything that touches the outside world (LLM, DB, yfinance, web).

Kept in one place so tests can inject fakes and avoid network access.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.llm.client import Usage
from app.llm.client import complete as llm_complete
from app.llm.structured import ask_json as real_ask_json
from app.rag.retrieval import Retrieved
from app.rag.retrieval import retrieve as real_retrieve
from app.tools.market import MarketData, get_company_metrics
from app.tools.web import WebSearchOutput
from app.tools.web import search_web as real_search_web


@dataclass
class Deps:
    pool: Any  # AsyncConnectionPool; may be None in tests when retrieve is replaced
    ask_json: Callable[..., Awaitable[tuple[Any, Usage]]] = real_ask_json
    generate: Callable[..., Awaitable[tuple[str, Usage]]] = llm_complete  # (model, messages, temperature=None)
    retrieve: Callable[..., Awaitable[list[Retrieved]]] = real_retrieve
    get_metrics: Callable[[str], Awaitable[MarketData]] = get_company_metrics
    search_web: Callable[..., Awaitable[WebSearchOutput]] = real_search_web
