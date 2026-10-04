import pytest

from app.db.database import init_schema, list_migrations, render


def test_migrations_are_ordered_unique_and_cover_the_whole_schema():
    migrations = list_migrations()
    versions = [version for version, _ in migrations]
    assert versions == sorted(versions) and len(set(versions)) == len(versions)
    assert [v.split("_")[0] for v in versions] == [f"{i:04d}" for i in range(1, len(versions) + 1)]  # no gaps
    sql = " ".join(text for _, text in migrations)
    for table in ("conversations", "messages", "chunks", "embedding_cache", "feedback"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


def test_every_migration_is_idempotent_so_it_can_be_applied_to_an_existing_database():
    for version, sql in list_migrations():
        statements = [s for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
        for statement in statements:
            body = "\n".join(line for line in statement.splitlines() if not line.strip().startswith("--")).strip()
            if body.upper().startswith(("CREATE TABLE", "CREATE INDEX", "CREATE EXTENSION")):
                assert "IF NOT EXISTS" in body.upper(), f"{version}: {body[:60]}"
            if body.upper().startswith("ALTER TABLE") and "ADD COLUMN" in body.upper():
                assert "IF NOT EXISTS" in body.upper(), f"{version}: {body[:60]}"


def test_embedding_dimension_is_substituted_into_the_rag_migration():
    sql = dict(list_migrations())["0002_rag"]
    assert "{{embedding_dim}}" in sql
    rendered = render(sql, 384)
    assert "{{" not in rendered and "vector(384)" in rendered


@pytest.mark.integration
async def test_applying_migrations_twice_is_harmless_and_records_every_version(db_pool):
    await init_schema(db_pool)
    again = await init_schema(db_pool)
    assert again == []  # everything is already applied
    async with db_pool.connection() as conn:
        rows = await (await conn.execute("SELECT version FROM schema_migrations ORDER BY version")).fetchall()
    assert [r["version"] for r in rows] == [v for v, _ in list_migrations()]
