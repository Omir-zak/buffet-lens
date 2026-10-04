"""Integration tests for conversation storage. Requires the docker-compose Postgres (skipped otherwise)."""

from uuid import uuid4

import pytest

from app.db import repository

pytestmark = pytest.mark.integration


async def test_exchange_is_saved_and_history_is_returned_in_order(db_pool, test_conversations):
    cid = uuid4()
    test_conversations.append(cid)
    await repository.save_exchange(db_pool, cid, "Q1", "A1", "model-x", 10, 5)
    await repository.save_exchange(db_pool, cid, "Q2", "A2", "model-x", 20, 8)

    history = await repository.get_history(db_pool, cid, limit=10)
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "Q1"),
        ("assistant", "A1"),
        ("user", "Q2"),
        ("assistant", "A2"),
    ]

    conversation = await repository.get_conversation(db_pool, cid)
    assert conversation["title"] == "Q1"  # the title comes from the first question
    assert conversation["messages"][1]["prompt_tokens"] == 10


async def test_history_limit_keeps_the_most_recent_messages_in_chronological_order(db_pool, test_conversations):
    cid = uuid4()
    test_conversations.append(cid)
    for i in range(5):
        await repository.save_exchange(db_pool, cid, f"Q{i}", f"A{i}", "m", None, None)
    history = await repository.get_history(db_pool, cid, limit=4)
    assert [m["content"] for m in history] == ["Q3", "A3", "Q4", "A4"]


async def test_unknown_conversation_is_reported_as_missing(db_pool, test_conversations):
    assert not await repository.conversation_exists(db_pool, uuid4())
    assert await repository.get_conversation(db_pool, uuid4()) is None


async def test_listing_shows_message_counts(db_pool, test_conversations):
    cid = uuid4()
    test_conversations.append(cid)
    await repository.save_exchange(db_pool, cid, "Hello", "Hi", "m", 1, 1)
    listed = {str(c["id"]): c for c in await repository.list_conversations(db_pool, limit=200)}
    assert listed[str(cid)]["message_count"] == 2


async def test_chunk_storage_supports_vector_and_keyword_search(db_pool, test_conversations):
    """Checks the RAG schema: the vector dimension and the full-text column work."""
    from app.config import get_settings
    from app.rag.embeddings import vector_literal
    from app.rag.retrieval import _keyword_search, _vector_search

    dim = get_settings().embedding_dim
    strategy = f"test_{uuid4().hex[:8]}"
    vec = [0.0] * dim
    vec[0] = 1.0
    try:
        async with db_pool.connection() as conn:
            await conn.execute(
                "INSERT INTO chunks (strategy, id, year, position, content, embedding) VALUES (%s, '1999-000', 1999, 0, %s, %s::vector)",
                (strategy, "Buffett warns about derivatives and leverage.", vector_literal(vec)),
            )
        by_vector = await _vector_search(db_pool, strategy, vector_literal(vec), 3, None)
        by_keyword = await _keyword_search(db_pool, strategy, "derivatives risk", vector_literal(vec), 3, None)
        filtered_out = await _vector_search(db_pool, strategy, vector_literal(vec), 3, [2005])
        assert by_vector[0].chunk_id == "1999-000" and by_vector[0].similarity == pytest.approx(1.0)
        assert [h.chunk_id for h in by_keyword] == ["1999-000"]
        assert filtered_out == []  # the year filter excluded the chunk
    finally:
        async with db_pool.connection() as conn:
            await conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))


async def test_messages_have_ids_and_user_id_is_stored_with_the_conversation(db_pool, test_conversations):
    cid = uuid4()
    test_conversations.append(cid)
    await repository.save_exchange(db_pool, cid, "Q", "A", "m", 1, 1, user_id="user_001")
    conversation = await repository.get_conversation(db_pool, cid)
    ids = [m["id"] for m in conversation["messages"]]
    assert len(ids) == 2 and ids == sorted(ids)  # messages have ids (the frontend uses them for feedback)
    async with db_pool.connection() as conn:
        row = await (await conn.execute("SELECT user_id FROM conversations WHERE id = %s", (cid,))).fetchone()
    assert row["user_id"] == "user_001"


async def test_deleting_a_conversation_removes_its_messages_and_reports_missing_ones(db_pool, test_conversations):
    cid = uuid4()
    test_conversations.append(cid)
    await repository.save_exchange(db_pool, cid, "Q", "A", "m", 1, 1)
    assert await repository.delete_conversation(db_pool, cid) is True
    assert await repository.get_conversation(db_pool, cid) is None
    async with db_pool.connection() as conn:
        count = await (
            await conn.execute("SELECT count(*) AS n FROM messages WHERE conversation_id = %s", (cid,))
        ).fetchone()
    assert count["n"] == 0  # messages were removed by cascade
    assert await repository.delete_conversation(db_pool, cid) is False  # deleting again reports "not found"


async def test_feedback_is_one_row_per_message_and_user_and_can_be_changed(db_pool, test_conversations):
    message_id = f"test-{uuid4()}"
    try:
        await repository.save_feedback(db_pool, message_id, "user_001", 1)
        await repository.save_feedback(db_pool, message_id, "user_001", -1)  # the user changed their mind
        await repository.save_feedback(db_pool, message_id, "", 1)  # a different author gets a separate record
        async with db_pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT user_id, rating FROM feedback WHERE message_id = %s ORDER BY user_id", (message_id,)
                )
            ).fetchall()
        assert [(r["user_id"], r["rating"]) for r in rows] == [("", 1), ("user_001", -1)]
    finally:
        async with db_pool.connection() as conn:
            await conn.execute("DELETE FROM feedback WHERE message_id = %s", (message_id,))
