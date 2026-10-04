-- Frontend support: conversation author and answer ratings (thumbs up/down).
ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT;

-- message_id is TEXT with no foreign key: the frontend sends its own id of a just-received message,
-- which may not exist in our DB yet. user_id is an empty string when the author is unknown.
CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    message_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL DEFAULT '',
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_id, user_id)
);
