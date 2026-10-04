import httpx
import openai
import pytest

from app.api.errors import describe_rate_limit, format_wait, translate_llm_error

TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization "
    "`org_secret123` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 5792, Requested 2334. "
    "Please try again in 945ms. Need more tokens?'}}"
)
TPD = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_secret123` service tier `on_demand` "
    "on tokens per day (TPD): Limit 200000, Used 199661, Requested 1828. Please try again in 10m43.248s."
)


def rate_limit_error(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return openai.RateLimitError(message, response=httpx.Response(429, request=request), body=None)


def test_per_minute_limit_is_named_and_the_wait_is_short():
    text = describe_rate_limit(rate_limit_error(TPM))
    assert "openai/gpt-oss-20b" in text and "tokens per minute" in text and "8,000" in text
    assert "about 1 s" in text


def test_per_day_limit_says_it_is_a_daily_limit_and_gives_a_long_wait():
    text = describe_rate_limit(rate_limit_error(TPD))
    assert "tokens per day" in text and "200,000" in text and "about 11 min" in text


def test_the_organisation_id_never_reaches_the_client():
    assert "org_secret123" not in describe_rate_limit(rate_limit_error(TPM))


def test_unparseable_provider_message_falls_back_to_a_generic_sentence():
    assert describe_rate_limit(rate_limit_error("something unexpected")) == (
        "The model provider rate limit was hit. Try again shortly."
    )


def test_translate_uses_429_and_the_detailed_message():
    err = translate_llm_error(rate_limit_error(TPM))
    assert err.status_code == 429 and err.code == "rate_limited" and "tokens per minute" in err.message


@pytest.mark.parametrize(
    "raw,expected",
    [("945ms", "about 1 s"), ("7.5s", "about 8 s"), ("10m43.248s", "about 11 min"), ("1h50m52.8s", "about 1.8 h")],
)
def test_wait_times_are_made_human_readable(raw, expected):
    assert format_wait(raw) == expected
