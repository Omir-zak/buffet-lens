"""Index Buffett letters: download -> extract text -> chunk -> embed -> Postgres.

Run from the project root (Postgres and Ollama must be up):
    python -m scripts.ingest                          # default strategy (recursive_1000)
    python -m scripts.ingest fixed_1000 recursive_1000   # several strategies for comparison

Safe to re-run: downloaded files and computed embeddings are cached.
"""

import argparse
import asyncio
import statistics
import sys
import time

from app.config import get_settings
from app.db.database import DictPool, create_pool, init_schema
from app.rag.chunking import STRATEGIES, Chunk, chunk_letter
from app.rag.embeddings import embed_documents_cached, vector_literal
from app.rag.loader import download_letters, load_text

# Async psycopg on Windows needs the selector loop (see run.py).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def ingest_strategy(pool: DictPool, name: str, letters: dict[int, str]) -> None:
    strategy = STRATEGIES[name]
    print(f"\n=== Стратегия {name}: {strategy.kind}, размер {strategy.size}, overlap {strategy.overlap} ===")

    chunks: list[Chunk] = []
    dropped_total = 0
    for year, text in sorted(letters.items()):
        year_chunks, dropped = chunk_letter(year, text, strategy)
        chunks.extend(year_chunks)
        dropped_total += dropped
    sizes = [len(c.text) for c in chunks]
    print(
        f"чанков: {len(chunks)} (отброшено таблиц: {dropped_total}) | "
        f"длина: средняя {statistics.mean(sizes):.0f}, мин {min(sizes)}, макс {max(sizes)}"
    )

    started = time.perf_counter()

    def show_progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        eta = elapsed / done * (total - done)
        print(f"  эмбеддинги: {done}/{total}  прошло {elapsed:.0f} с, осталось ~{eta:.0f} с", flush=True)

    vectors, cache_hits = await embed_documents_cached(pool, [c.text for c in chunks], show_progress)
    print(f"эмбеддинги: {len(vectors)} шт. за {time.perf_counter() - started:.0f} с (из кэша: {cache_hits})")

    async with pool.connection() as conn:
        # Single transaction: the strategy is replaced entirely or left untouched.
        await conn.execute("DELETE FROM chunks WHERE strategy = %s", (name,))
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunks (strategy, id, year, position, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                [
                    (name, c.id, c.year, c.position, c.text, vector_literal(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
    print(f"сохранено в Postgres: {len(chunks)} чанков")


async def main(strategy_names: list[str]) -> None:
    print("Загрузка писем (уже скачанные пропускаются)...")
    paths = download_letters()
    letters = {year: load_text(path) for year, path in paths.items()}
    print(f"писем: {len(letters)}, символов: {sum(len(t) for t in letters.values()):,}")

    pool = create_pool(get_settings().database_url)
    await pool.open()
    try:
        await init_schema(pool)
        for name in strategy_names:
            await ingest_strategy(pool, name, letters)
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "strategies", nargs="*", default=[get_settings().active_chunk_strategy], choices=list(STRATEGIES)
    )
    asyncio.run(main(parser.parse_args().strategies))
