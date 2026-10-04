"""API key check: if API_KEY is configured, /v1 requests must send a matching `x-api-key` header.

The comparison is constant-time (hmac.compare_digest) to avoid leaking the key through timing.
"""

import hmac

from fastapi import Security
from fastapi.security import APIKeyHeader

from app.api.errors import ApiError
from app.config import get_settings

# auto_error=False: we raise ApiError ourselves to keep the error format uniform.
# Registering the scheme also adds the Authorize button to Swagger (/docs).
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False, description="API key (if the server requires one)")


async def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = get_settings().api_key
    if not expected:
        return
    if not key or not hmac.compare_digest(key.encode("utf-8"), expected.encode("utf-8")):
        raise ApiError(401, "invalid_api_key", "Missing or invalid API key.")
