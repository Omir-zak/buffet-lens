"""Market data tool (yfinance).

yfinance is an unofficial Yahoo Finance scraper that sometimes breaks, returns empty fields or
hits rate limits. Hence:
  * the call runs in a thread (the library is blocking and would stall the async server);
  * there is a timeout and a cache;
  * the function never raises: any problem becomes an ok=False result with a clear message,
    so the agent graph keeps going.
"""

import asyncio
import re
from datetime import UTC, datetime

import yfinance as yf
from pydantic import BaseModel

from app.config import get_settings
from app.tools.cache import TTLCache

TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,12}$")
_cache = TTLCache(get_settings().tool_cache_seconds)


class MarketData(BaseModel):
    """Company market metrics. Units matter, so they are noted per field."""

    ok: bool
    ticker: str
    name: str | None = None
    currency: str | None = None
    price: float | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None  # P/E on trailing 12 months
    forward_pe: float | None = None  # P/E on forecast earnings
    price_to_book: float | None = None
    dividend_yield_pct: float | None = None  # percent (2.5 means 2.5%)
    trailing_eps: float | None = None
    return_on_equity: float | None = None  # fraction (0.42 means 42%)
    operating_margin: float | None = None  # fraction
    debt_to_equity: float | None = None  # percent, as Yahoo returns it (115 means 115%)
    free_cash_flow: float | None = None
    as_of: str | None = None  # when the data was fetched (UTC)
    error: str | None = None

    def to_evidence_text(self) -> str:
        """Human-readable summary for the LLM, with numbers already in readable units."""
        if not self.ok:
            return f"Market data for {self.ticker} is unavailable: {self.error}"

        def fmt(label: str, value: float | None, spec: str, suffix: str = "") -> str | None:
            return None if value is None else f"{label}: {format(value, spec)}{suffix}"

        lines = [
            f"{self.name or self.ticker} ({self.ticker}), data as of {self.as_of}",
            fmt("Price", self.price, ",.2f", f" {self.currency or ''}".rstrip()),
            fmt("Market cap", self.market_cap, ",.0f"),
            fmt("Trailing P/E", self.trailing_pe, ".2f"),
            fmt("Forward P/E", self.forward_pe, ".2f"),
            fmt("Price/Book", self.price_to_book, ".2f"),
            fmt("Dividend yield", self.dividend_yield_pct, ".2f", "%"),
            fmt("Trailing EPS", self.trailing_eps, ".2f"),
            None if self.return_on_equity is None else f"Return on equity: {self.return_on_equity * 100:.1f}%",
            None if self.operating_margin is None else f"Operating margin: {self.operating_margin * 100:.1f}%",
            fmt("Debt/Equity", self.debt_to_equity, ".1f", "%"),
            fmt("Free cash flow", self.free_cash_flow, ",.0f"),
        ]
        return "\n".join(line for line in lines if line)


def _fetch(ticker: str) -> MarketData:
    """Blocking Yahoo request; runs in a thread."""
    info = yf.Ticker(ticker).info
    # For a nonexistent ticker Yahoo returns a nearly empty dict with no price.
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        return MarketData(ok=False, ticker=ticker, error=f"Unknown ticker or no market data for '{ticker}'.")
    return MarketData(
        ok=True,
        ticker=ticker,
        name=info.get("shortName") or info.get("longName"),
        currency=info.get("currency"),
        price=price,
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        price_to_book=info.get("priceToBook"),
        dividend_yield_pct=info.get("dividendYield"),
        trailing_eps=info.get("trailingEps"),
        return_on_equity=info.get("returnOnEquity"),
        operating_margin=info.get("operatingMargins"),
        debt_to_equity=info.get("debtToEquity"),
        free_cash_flow=info.get("freeCashflow"),
        as_of=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


async def get_company_metrics(ticker: str) -> MarketData:
    """Return current market metrics and valuation ratios for a public company.

    Args:
        ticker: exchange ticker, e.g. "KO" (Coca-Cola), "AAPL" (Apple), "BRK-B" (Berkshire).

    Returns:
        MarketData. If the ticker is unknown or the source is unavailable, ok=False and error
        explains why. Never raises.
    """
    ticker = ticker.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        return MarketData(ok=False, ticker=ticker[:12], error="Invalid ticker format.")

    cached = _cache.get(ticker)
    if cached is not None:
        return cached

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_fetch, ticker), timeout=get_settings().tool_timeout_seconds)
    except TimeoutError:
        return MarketData(ok=False, ticker=ticker, error="The market data service timed out.")
    except Exception as exc:  # yfinance raises many kinds of exceptions (network, parsing, rate limits)
        return MarketData(
            ok=False, ticker=ticker, error=f"Market data is temporarily unavailable ({type(exc).__name__})."
        )

    if result.ok:  # don't cache errors: the next request may succeed
        _cache.set(ticker, result)
    return result
