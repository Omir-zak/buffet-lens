"""Conversation history storage in plain SQL (no ORM). Every function takes the pool first."""

from typing import Any
from uuid import UUID

from app.db.database import DictPool

Row = dict[str, Any]


async def conversation_exists(pool: DictPool, conversation_id: UUID) -> bool:
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM conversations WHERE id = %s", (conversation_id,))
        return await cur.fetchone() is not None


async def get_history(pool: DictPool, conversation_id: UUID, limit: int) -> list[Row]:
    """The last `limit` messages of a conversation in chronological order."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE conversation_id = %s ORDER BY id DESC LIMIT %s
            ) AS last_messages
            ORDER BY id ASC
            """,
            (conversation_id, limit),
        )
        return await cur.fetchall()


async def save_exchange(
    pool: DictPool,
    conversation_id: UUID,
    user_text: str,
    assistant_text: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    user_id: str | None = None,
) -> None:
    """Save a question + answer pair in one transaction, creating the conversation if needed.

    Called only after a successful model answer, so LLM failures leave no unanswered questions.
    user_id (optional) is recorded when the conversation is created.
    """
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO conversations (id, title, user_id) VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            (conversation_id, user_text[:80], user_id),
        )
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
            (conversation_id, user_text),
        )
        await conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, model, prompt_tokens, completion_tokens)
            VALUES (%s, 'assistant', %s, %s, %s, %s)
            """,
            (conversation_id, assistant_text, model, prompt_tokens, completion_tokens),
        )


async def list_conversations(pool: DictPool, limit: int = 50) -> list[Row]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c ORDER BY c.updated_at DESC LIMIT %s
            """,
            (limit,),
        )
        return await cur.fetchall()


async def get_conversation(pool: DictPool, conversation_id: UUID) -> Row | None:
    """A conversation with its messages, or None if it doesn't exist."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        conversation = await cur.fetchone()
        if conversation is None:
            return None
        cur = await conn.execute(
            """
            SELECT id, role, content, model, prompt_tokens, completion_tokens, created_at
            FROM messages WHERE conversation_id = %s ORDER BY id ASC
            """,
            (conversation_id,),
        )
        conversation["messages"] = await cur.fetchall()
        return conversation


async def delete_conversation(pool: DictPool, conversation_id: UUID) -> bool:
    """Delete a conversation and (by cascade) its messages. Returns False if it didn't exist."""
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        return cur.rowcount > 0


async def save_feedback(pool: DictPool, message_id: str, user_id: str, rating: int) -> None:
    """Save an answer rating (1 or -1). Rating the same message again as the same user updates the
    existing row, so changing one's mind keeps a single record."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO feedback (message_id, user_id, rating) VALUES (%s, %s, %s)
            ON CONFLICT (message_id, user_id) DO UPDATE SET rating = EXCLUDED.rating, created_at = now()
            """,
            (message_id, user_id, rating),
        )
