"""Tools must not raise: any problem becomes an ok=False result."""

import asyncio
import time

import pytest

from app.tools import market, web
from app.tools.cache import TTLCache
from app.tools.market import MarketData, get_company_metrics
from app.tools.web import WebResult, search_web


@pytest.fixture(autouse=True)
def clean_caches():
    market._cache.clear()
    web._cache.clear()


def good_data(ticker: str) -> MarketData:
    return MarketData(
        ok=True,
        ticker=ticker,
        name="Coca-Cola",
        currency="USD",
        price=85.65,
        trailing_pe=25.88,
        as_of="2026-10-03 11:00 UTC",
    )


async def test_valid_ticker_returns_data_and_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(market, "_fetch", lambda t: calls.append(t) or good_data(t))
    first = await get_company_metrics(" ko ")
    second = await get_company_metrics("KO")
    assert first.ok and first.ticker == "KO" and second.ok
    assert calls == ["KO"]  # the second call was served from cache


async def test_unknown_ticker_is_a_clean_failure_not_an_exception(monkeypatch):
    monkeypatch.setattr(market, "_fetch", lambda t: MarketData(ok=False, ticker=t, error="Unknown ticker"))
    result = await get_company_metrics("ZZZZ")
    assert not result.ok and "Unknown" in result.error


async def test_failures_are_not_cached(monkeypatch):
    attempts = []

    def flaky(t):
        attempts.append(t)
        return MarketData(ok=False, ticker=t, error="boom") if len(attempts) == 1 else good_data(t)

    monkeypatch.setattr(market, "_fetch", flaky)
    assert not (await get_company_metrics("KO")).ok
    assert (await get_company_metrics("KO")).ok  # the retry succeeded, so the error was not cached


@pytest.mark.parametrize("bad", ["", "DROP TABLE;", "A" * 40, "../etc/passwd"])
async def test_malformed_tickers_are_rejected_without_calling_the_network(monkeypatch, bad):
    monkeypatch.setattr(market, "_fetch", lambda t: pytest.fail("network must not be called"))
    result = await get_company_metrics(bad)
    assert not result.ok and result.error == "Invalid ticker format."


async def test_yfinance_exception_becomes_a_friendly_error(monkeypatch):
    def explode(t):
        raise ConnectionError("yahoo is down")

    monkeypatch.setattr(market, "_fetch", explode)
    result = await get_company_metrics("KO")
    assert not result.ok and "temporarily unavailable" in result.error and "ConnectionError" in result.error


async def test_slow_source_times_out(monkeypatch):
    monkeypatch.setattr(market, "_fetch", lambda t: time.sleep(0.5) or good_data(t))
    monkeypatch.setattr(market, "get_settings", lambda: type("S", (), {"tool_timeout_seconds": 0.05})())
    result = await get_company_metrics("KO")
    assert not result.ok and "timed out" in result.error


def test_failed_market_data_is_described_in_plain_words():
    text = MarketData(ok=False, ticker="ZZZ", error="Unknown ticker").to_evidence_text()
    assert "ZZZ" in text and "unavailable" in text


def test_units_in_evidence_text_are_human_readable():
    data = good_data("KO").model_copy(update={"return_on_equity": 0.421, "dividend_yield_pct": 2.48})
    text = data.to_evidence_text()
    assert "Return on equity: 42.1%" in text and "Dividend yield: 2.48%" in text


async def test_web_search_success_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(
        web, "_search_ddgs", lambda q, n: calls.append(q) or [WebResult(title="t", url="u", snippet="s")]
    )
    first = await search_web("Coca-Cola earnings")
    await search_web("coca-cola EARNINGS")  # case-insensitive
    assert first.ok and first.results[0].title == "t" and len(calls) == 1


async def test_web_search_failure_is_reported_not_raised(monkeypatch):
    def fail(q, n):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(web, "_search_ddgs", fail)
    result = await search_web("anything")
    assert not result.ok and "unavailable" in result.error


async def test_empty_web_query_is_rejected():
    assert not (await search_web("   ")).ok


def test_placeholder_tavily_key_is_not_treated_as_a_real_key():
    assert not web._has_real_tavily_key("tvly-...")
    assert not web._has_real_tavily_key("")
    assert web._has_real_tavily_key("tvly-abc123")


def test_ttl_cache_expires_entries():
    cache = TTLCache(0.05)
    cache.set("k", 1)
    assert cache.get("k") == 1
    time.sleep(0.08)
    assert cache.get("k") is None


async def test_asyncio_to_thread_is_used_so_the_event_loop_is_not_blocked(monkeypatch):
    """While a "slow" yfinance call runs in a thread, the event loop keeps serving other tasks."""
    monkeypatch.setattr(market, "_fetch", lambda t: time.sleep(0.2) or good_data(t))
    ticks = []

    async def ticker():
        for _ in range(5):
            ticks.append(1)
            await asyncio.sleep(0.02)

    await asyncio.gather(get_company_metrics("KO"), ticker())
    assert len(ticks) == 5
