"""Shared test setup.

Tests must not hit the network: the LLM, yfinance and web search are replaced with fakes (see Deps).
The only exception is integration tests, which need the Postgres from docker-compose.
"""

import asyncio
import os
import sys

import pytest

# The config requires a key even in tests (no real request is made: everything is faked).
os.environ.setdefault("LLM_API_KEY", "test-key")
# Tests must not depend on a developer's local .env: environment variables win over the .env file.
# An empty API_KEY means "API is open"; the API-key tests set it themselves.
os.environ["API_KEY"] = ""
# Tracing is off in tests: Langfuse keys from .env must not send data from tests.
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# Async psycopg on Windows requires this event loop policy (see run.py).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
async def db_pool():
    """Pool to the real Postgres. Integration tests are skipped if the container is not running."""
    from app.config import get_settings
    from app.db.database import create_pool, init_schema

    pool = create_pool(get_settings().database_url)
    try:
        await pool.open(wait=True, timeout=5)
    except Exception:
        await pool.close()
        pytest.skip("Postgres is not available (run `docker compose up -d`)")
    await init_schema(pool)
    yield pool
    await pool.close()


@pytest.fixture
async def test_conversations(db_pool):
    """IDs of conversations created by a test. They are deleted afterwards: tests use the same database
    as the app and must not leave junk in it (it would show up in the connected frontend's chat history)."""
    created: list = []
    yield created
    if created:
        async with db_pool.connection() as conn:
            await conn.execute("DELETE FROM conversations WHERE id = ANY(%s)", (created,))
