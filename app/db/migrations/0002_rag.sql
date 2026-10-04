-- RAG tables. {{embedding_dim}} is substituted by the migration loader from settings
-- (it depends on the embedding model).
--
-- chunks: one row per letter chunk; the strategy column lets several chunking variants coexist
--   so they can be compared fairly in eval.
-- tsv: Postgres full-text column, generated automatically; used for hybrid (keyword + semantic) search.
-- No vector (HNSW) index on purpose: with ~4000 vectors an exact scan takes milliseconds, while an
--   ANN index would add approximation for no speed gain. For millions of vectors use:
--   CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
-- embedding_cache: avoids re-embedding letters when ingest is re-run.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    strategy   TEXT NOT NULL,
    id         TEXT NOT NULL,
    year       INTEGER NOT NULL,
    position   INTEGER NOT NULL,
    content    TEXT NOT NULL,
    embedding  vector({{embedding_dim}}) NOT NULL,
    tsv        tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    PRIMARY KEY (strategy, id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks (strategy, year);

CREATE TABLE IF NOT EXISTS embedding_cache (
    hash       TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    embedding  vector({{embedding_dim}}) NOT NULL
);
