"""Retrieve relevant letter passages: vector, keyword, or hybrid.

Vector search finds semantic matches but can miss exact terms and names ("See's Candies", "float").
Full-text search is exact on words but blind to synonyms. Hybrid combines both with Reciprocal Rank Fusion.
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.config import get_settings
from app.db.database import DictPool
from app.rag.embeddings import embed_texts, vector_literal

Mode = Literal["vector", "keyword", "hybrid"]
RRF_K = 60  # standard RRF constant: smooths the contribution of list position
CANDIDATES = 20  # candidates taken from each search before fusion


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    year: int
    content: str
    similarity: float  # cosine similarity to the question (1 = identical, 0 = unrelated)
    score: float  # final ranking score (depends on mode)


async def _vector_search(
    pool: DictPool, strategy: str, qvec: str, limit: int, years: list[int] | None
) -> list[Retrieved]:
    # <=> is pgvector's cosine DISTANCE; similarity = 1 - distance.
    sql = """
        SELECT id, year, content, 1 - (embedding <=> %(q)s::vector) AS similarity
        FROM chunks WHERE strategy = %(s)s AND (%(y)s::int[] IS NULL OR year = ANY(%(y)s))
        ORDER BY embedding <=> %(q)s::vector LIMIT %(n)s
    """
    async with pool.connection() as conn:
        rows = await (await conn.execute(sql, {"q": qvec, "s": strategy, "n": limit, "y": years})).fetchall()
    return [Retrieved(r["id"], r["year"], r["content"], float(r["similarity"]), float(r["similarity"])) for r in rows]


# Words present in every letter (all written by Buffett for Berkshire shareholders). In a query they
# only pull results toward unrelated passages: "financial weapons of mass destruction Buffett" lost the
# right chunk, while without "Buffett" it ranks first. Found while checking a real user question.
CORPUS_WIDE_WORDS = frozenset(
    {"buffett", "warren", "berkshire", "hathaway", "letter", "letters", "shareholder", "shareholders"}
)
QUOTE_CHARS = re.compile(r"[\"\u201c\u201d\u2018\u2019]")


def clean_query(query: str) -> str:
    """Drop quotes and corpus-wide words; if nothing else remains, return the query unchanged."""
    without_quotes = QUOTE_CHARS.sub(" ", query)
    kept = [w for w in without_quotes.split() if w.lower().strip(".,;:!?()") not in CORPUS_WIDE_WORDS]
    return " ".join(kept) if kept else query.strip()


RARE_TERM_MAX_DF = 0.10  # a word in more than 10% of chunks is treated as uninformative

# Corpus term stats for rarity weighting: {strategy: (chunk count, {word: number of chunks containing it})}.
_term_stats_cache: dict[str, tuple[int, dict[str, int]]] = {}


async def _term_stats(pool: DictPool, strategy: str) -> tuple[int, dict[str, int]]:
    """Document frequency of each word; computed once and cached."""
    cached = _term_stats_cache.get(strategy)
    if cached is not None:
        return cached
    if not re.fullmatch(r"[a-z0-9_]+", strategy):  # the strategy name is interpolated into the ts_stat SQL string
        raise ValueError(f"Invalid strategy name: {strategy!r}")
    async with pool.connection() as conn:
        row = await (await conn.execute("SELECT count(*) AS n FROM chunks WHERE strategy = %s", (strategy,))).fetchone()
        total = row["n"] if row else 0
        rows = await (
            await conn.execute(
                f"SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks WHERE strategy = ''{strategy}''')"
            )
        ).fetchall()
    stats = (total, {r["word"]: r["ndoc"] for r in rows})
    _term_stats_cache[strategy] = stats
    return stats


async def _query_lexemes(pool: DictPool, query: str) -> list[str]:
    """Query words normalized the same way as in the index (stop words removed, stemmed)."""
    async with pool.connection() as conn:
        rows = await (await conn.execute("SELECT lexeme FROM unnest(to_tsvector('english', %s))", (query,))).fetchall()
    return [r["lexeme"] for r in rows]


async def _keyword_search(
    pool: DictPool, strategy: str, query: str, qvec: str, limit: int, years: list[int] | None, idf_filter: bool = True
) -> list[Retrieved]:
    # Query words are joined with OR (|), not AND: with AND the whole question almost never matches
    # a single chunk (not every chunk has "Buffett", "call", etc.) and search returns nothing.
    #
    # Also, ts_rank_cd ignores term rarity (IDF in BM25): a rare word ("covid", in one chunk of 4000)
    # weighs the same as a common one ("Buffett") and drowns. So if the query has rare words,
    # common ones (in more than 10% of chunks) are dropped.
    lexemes = await _query_lexemes(pool, query)
    if not lexemes:
        return []
    if idf_filter:
        total, doc_freq = await _term_stats(pool, strategy)
        # Words absent from the corpus can't match; excluding them from "rare" keeps them from
        # displacing words that actually find chunks.
        present = [w for w in lexemes if doc_freq.get(w, 0) > 0]
        rare = [w for w in present if total and doc_freq[w] / total <= RARE_TERM_MAX_DF]
        lexemes = rare or present or lexemes  # if all words are common, keep them all
    # 'simple': lexemes are already stemmed by the 'english' config; don't stem twice.
    tsquery = " | ".join("'" + w.replace("'", "''") + "'" for w in lexemes)
    sql = """
        SELECT id, year, content, ts_rank_cd(tsv, q) AS rank,
               1 - (embedding <=> %(v)s::vector) AS similarity
        FROM chunks, to_tsquery('simple', %(t)s) AS q
        WHERE strategy = %(s)s AND tsv @@ q AND (%(y)s::int[] IS NULL OR year = ANY(%(y)s))
        ORDER BY rank DESC LIMIT %(n)s
    """
    params = {"v": qvec, "t": tsquery, "s": strategy, "n": limit, "y": years}
    async with pool.connection() as conn:
        rows = await (await conn.execute(sql, params)).fetchall()
    return [Retrieved(r["id"], r["year"], r["content"], float(r["similarity"]), float(r["rank"])) for r in rows]


def reciprocal_rank_fusion(rankings: list[list[Retrieved]], k: int) -> list[Retrieved]:
    """Merge several ranked lists into one.

    Each document scores the sum of 1/(RRF_K + position) over the lists it appears in. Positions are
    summed rather than raw scores because vector and full-text scores are not comparable in scale.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Retrieved] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (RRF_K + position)
            by_id.setdefault(item.chunk_id, item)
    best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]
    return [Retrieved(by_id[i].chunk_id, by_id[i].year, by_id[i].content, by_id[i].similarity, scores[i]) for i in best]


async def retrieve(
    pool: DictPool,
    query: str,
    k: int = 5,
    mode: Mode = "hybrid",
    strategy: str | None = None,
    years: list[int] | None = None,
    idf_filter: bool = True,
    clean: bool = True,
) -> list[Retrieved]:
    """Return the k best-matching passages.

    years: metadata filter. When the user asks about a specific letter ("in 1990"), only chunks from
    those years are searched; this is more precise than hoping the year leaks through the query
    meaning, since embeddings encode numbers poorly.
    """
    years = years or None  # an empty list means no filter
    if clean:
        query = clean_query(query)
    strategy = strategy or get_settings().active_chunk_strategy
    qvec = vector_literal((await embed_texts([query], "query"))[0])

    if mode == "vector":
        return (await _vector_search(pool, strategy, qvec, k, years))[:k]
    if mode == "keyword":
        return (await _keyword_search(pool, strategy, query, qvec, k, years, idf_filter))[:k]

    vector_hits = await _vector_search(pool, strategy, qvec, CANDIDATES, years)
    keyword_hits = await _keyword_search(pool, strategy, query, qvec, CANDIDATES, years, idf_filter)
    return reciprocal_rank_fusion([vector_hits, keyword_hits], k)
