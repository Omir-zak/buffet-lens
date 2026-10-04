"""Postgres connection pools and schema migrations.

Migrations are versioned SQL files in `app/db/migrations/NNNN_name.sql`. At startup, those missing
from the `schema_migrations` table are applied in numeric order. The files are idempotent
(IF NOT EXISTS), so they also apply cleanly to a database created before versioning existed.
"""

from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

# Pool whose connections return rows as dicts; the alias lets mypy accept row["id"] access.
DictPool = AsyncConnectionPool[AsyncConnection[DictRow]]

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Arbitrary advisory-lock id: the API and the MCP server may start at the same time, and without
# the lock both would try to apply the same migration.
MIGRATION_LOCK_ID = 7_265_001


def list_migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[str, str]]:
    """All migrations in the folder as [(version, SQL)], ascending. The version is the file stem."""
    return [(path.stem, path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.sql"))]


def render(sql: str, embedding_dim: int) -> str:
    """Fill settings into a migration template (the vector size depends on the embedding model)."""
    return sql.replace("{{embedding_dim}}", str(embedding_dim))


def create_pool(database_url: str) -> DictPool:
    """Create the pool, still closed: it is opened in the app lifespan."""
    return AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"row_factory": dict_row},
    )


def create_checkpoint_pool(database_url: str) -> DictPool:
    """Separate pool for the LangGraph checkpointer, which needs autocommit and dict rows."""
    return AsyncConnectionPool[AsyncConnection[DictRow]](
        conninfo=database_url, kwargs={"autocommit": True, "row_factory": dict_row}, open=False
    )


async def init_schema(pool: DictPool) -> list[str]:
    """Apply missing migrations; returns the versions applied by this call."""
    dim = get_settings().embedding_dim
    applied_now: list[str] = []
    # One transaction: either all missing migrations apply or none.
    async with pool.connection() as conn:
        await conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cur = await conn.execute("SELECT version FROM schema_migrations")
        done = {row["version"] for row in await cur.fetchall()}
        for version, sql in list_migrations():
            if version in done:
                continue
            await conn.execute(render(sql, dim))
            await conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied_now.append(version)
    return applied_now
