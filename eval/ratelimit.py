"""Provider rate-limit resilience for the eval.

The free tier caps tokens per minute (TPM) and per day (TPD). The model client handles short waits
itself (retry with backoff); minute-long waits used to abort the whole run. Here the call is retried
after the wait time the provider names, so the eval can finish unattended.
If the wait exceeds `max_wait_seconds` (e.g. the daily limit is exhausted until tomorrow),
`QuotaExhausted` is raised: better to stop and run the remainder later.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai

from app.api.errors import RATE_LIMIT_PATTERN, parse_wait_seconds

T = TypeVar("T")

DEFAULT_MAX_WAIT_SECONDS = 1800.0  # never wait longer than 30 minutes
MAX_ATTEMPTS = 40
SAFETY_MARGIN_SECONDS = 2.0


class QuotaExhausted(RuntimeError):
    """Limit exhausted for a long time: not worth waiting within the run."""


def wait_seconds_from(exc: Exception) -> float | None:
    """Seconds the provider asks to wait (None if the message doesn't say)."""
    match = RATE_LIMIT_PATTERN.search(str(exc))
    return parse_wait_seconds(match["wait"]) if match else None


async def with_rate_limit_retry(
    call: Callable[[], Awaitable[T]],
    *,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `call()`; on a rate-limit error wait the provider-stated time and retry."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await call()
        except openai.RateLimitError as exc:
            wait = wait_seconds_from(exc)
            if wait is None:
                wait = min(30.0 * attempt, 300.0)  # provider gave no wait time: back off with growing pauses
            if wait > max_wait_seconds or attempt == MAX_ATTEMPTS:
                raise QuotaExhausted(f"Rate limit needs a wait of {wait:.0f} s (attempt {attempt}): {exc}") from exc
            print(f"\n[лимит провайдера, жду {wait + SAFETY_MARGIN_SECONDS:.0f} с, попытка {attempt}]", flush=True)
            await sleep(wait + SAFETY_MARGIN_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover
