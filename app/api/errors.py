"""Uniform API error format; translates provider (openai.*) exceptions into HTTP errors."""

import re

import openai


class ApiError(Exception):
    """An error deliberately returned to the client."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# Provider message format: "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_..`
# service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 5792, Requested 2334. Please try again in 945ms."
RATE_LIMIT_PATTERN = re.compile(
    r"for model [`']?(?P<model>[^`'\s]+)[`']?.*?on (?P<what>requests|tokens) per (?P<per>minute|day)"
    r".*?Limit (?P<limit>\d+).*?try again in (?P<wait>[0-9hms.]+)",
    re.DOTALL,
)
_WAIT_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_wait_seconds(text: str) -> float:
    """'945ms' -> 0.945, '10m43.248s' -> 643.248, '1h50m52.8s' -> 6652.8."""
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in _WAIT_PART.findall(text))


def format_wait(text: str) -> str:
    """'945ms' -> 'about 1 s', '10m43.248s' -> 'about 11 min', '1h50m52.8s' -> 'about 1.8 h'."""
    seconds = parse_wait_seconds(text)
    if seconds < 1.5:
        return "about 1 s"
    if seconds < 90:
        return f"about {round(seconds)} s"
    if seconds < 5400:
        return f"about {round(seconds / 60)} min"
    return f"about {seconds / 3600:.1f} h"


def describe_rate_limit(exc: Exception) -> str:
    """Human-readable message: which limit was hit and when to retry. Falls back to a generic
    phrase if the provider message can't be parsed. The organization id is never exposed."""
    match = RATE_LIMIT_PATTERN.search(str(exc))
    if not match:
        return "The model provider rate limit was hit. Try again shortly."
    limit = int(match["limit"])
    return (
        f"Rate limit reached for {match['model']}: {match['what']} per {match['per']} (limit {limit:,}). "
        f"Try again in {format_wait(match['wait'])}."
    )


def translate_llm_error(exc: Exception) -> ApiError:
    """Map a provider error to ApiError. Order matters: APITimeoutError subclasses APIConnectionError."""
    if isinstance(exc, openai.RateLimitError):
        return ApiError(429, "rate_limited", describe_rate_limit(exc))
    if isinstance(exc, openai.APITimeoutError):
        return ApiError(504, "upstream_timeout", "The model provider did not respond in time.")
    if isinstance(exc, openai.NotFoundError):
        return ApiError(400, "invalid_model", "The requested model does not exist at the provider.")
    if isinstance(exc, openai.BadRequestError):
        return ApiError(400, "bad_request", "The provider rejected the request (check the model name and input).")
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        # Our misconfiguration, not the client's: 502, no details exposed.
        return ApiError(502, "upstream_auth", "The server is misconfigured for the model provider.")
    if isinstance(exc, openai.APIConnectionError):
        return ApiError(502, "upstream_unreachable", "Could not reach the model provider.")
    return ApiError(502, "upstream_error", "The model provider returned an unexpected error.")
