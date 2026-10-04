"""Embeddings via any OpenAI-compatible /embeddings endpoint (Ollama here).

Embedding is slow and the letters never change, so document vectors are cached in the
embedding_cache table. The cache key includes the model, so switching models never reuses stale vectors.
"""

import asyncio
import hashlib
import json
from collections.abc import Callable
from typing import Literal

import httpx

from app.config import get_settings
from app.db.database import DictPool

Kind = Literal["document", "query"]
BATCH_SIZE = 32  # texts per HTTP request
SAVE_EVERY = 256  # texts between cache saves to the DB


def vector_literal(vector: list[float]) -> str:
    """Postgres vector literal: '[0.1, 0.2, ...]'."""
    return json.dumps(vector)


def _with_prefix(text: str, kind: Kind) -> str:
    settings = get_settings()
    prefix = settings.embedding_doc_prefix if kind == "document" else settings.embedding_query_prefix
    return prefix + text


async def _embed_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    last_error: Exception | None = None
    for attempt in range(3):  # retry with growing pauses: 1s, 2s
        try:
            response = await client.post(
                f"{settings.embedding_base_url}/embeddings",
                json={"model": settings.embedding_model, "input": texts},
                headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in data]
        except (httpx.HTTPError, KeyError) as exc:
            last_error = exc
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"Embedding service failed after 3 attempts: {last_error!r}")


async def embed_texts(texts: list[str], kind: Kind) -> list[list[float]]:
    """Embed texts in batches, without caching (enough for user queries)."""
    prefixed = [_with_prefix(t, kind) for t in texts]
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for start in range(0, len(prefixed), BATCH_SIZE):
            vectors.extend(await _embed_batch(client, prefixed[start : start + BATCH_SIZE]))
    return vectors


def _cache_key(text: str) -> str:
    settings = get_settings()
    return hashlib.sha256(f"{settings.embedding_model}\n{_with_prefix(text, 'document')}".encode()).hexdigest()


async def embed_documents_cached(
    pool: DictPool,
    texts: list[str],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[list[float]], int]:
    """Embed documents with a DB cache. Returns (vectors, number of cache hits)."""
    keys = [_cache_key(t) for t in texts]
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT hash, embedding::text AS embedding FROM embedding_cache WHERE hash = ANY(%s)", (keys,)
        )
        cached = {row["hash"]: json.loads(row["embedding"]) for row in await cur.fetchall()}

    missing = sorted({i for i, k in enumerate(keys) if k not in cached})
    model = get_settings().embedding_model
    # Compute and save in portions so a crash mid-run loses nothing and a rerun resumes where it stopped.
    for start in range(0, len(missing), SAVE_EVERY):
        part = missing[start : start + SAVE_EVERY]
        fresh = await embed_texts([texts[i] for i in part], "document")
        async with pool.connection() as conn:
            for i, vector in zip(part, fresh, strict=True):
                cached[keys[i]] = vector
                await conn.execute(
                    "INSERT INTO embedding_cache (hash, model, embedding) VALUES (%s, %s, %s::vector) "
                    "ON CONFLICT DO NOTHING",
                    (keys[i], model, vector_literal(vector)),
                )
        if on_progress:
            on_progress(min(start + SAVE_EVERY, len(missing)), len(missing))
    hits = len(texts) - len(missing)
    return [cached[k] for k in keys], hits
