import httpx
import openai
import pytest

from app.api.errors import parse_wait_seconds
from eval.ratelimit import QuotaExhausted, with_rate_limit_retry

TPM_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-20b` in organization `org_x` service tier `on_demand` "
    "on tokens per minute (TPM): Limit 8000, Used 7000, Requested 2000. Please try again in 6.5s."
)
TPD_MESSAGE = TPM_MESSAGE.replace("tokens per minute (TPM)", "tokens per day (TPD)").replace("6.5s", "3h20m10s")


def rate_limit(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return openai.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


def test_wait_time_is_parsed_into_seconds():
    assert parse_wait_seconds("945ms") == pytest.approx(0.945)
    assert parse_wait_seconds("10m43.248s") == pytest.approx(643.248)
    assert parse_wait_seconds("1h50m52.8s") == pytest.approx(6652.8)


class Flaky:
    """Raises a rate-limit error a given number of times, then returns a result."""

    def __init__(self, failures: int, message: str = TPM_MESSAGE) -> None:
        self.failures, self.message, self.calls = failures, message, 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise rate_limit(self.message)
        return "ok"


async def test_a_call_is_repeated_after_the_wait_the_provider_asks_for():
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    flaky = Flaky(failures=2)
    assert await with_rate_limit_retry(flaky, sleep=fake_sleep) == "ok"
    assert flaky.calls == 3 and slept == [pytest.approx(8.5), pytest.approx(8.5)]  # 6.5 s from the provider + margin


async def test_a_long_daily_wait_is_not_waited_out_inside_the_run():
    async def fake_sleep(seconds: float) -> None:
        raise AssertionError("must not sleep for hours")

    with pytest.raises(QuotaExhausted):
        await with_rate_limit_retry(Flaky(failures=1, message=TPD_MESSAGE), sleep=fake_sleep)


async def test_unrelated_errors_are_not_swallowed():
    async def broken() -> str:
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        await with_rate_limit_retry(broken)
