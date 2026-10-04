"""Retrieval evaluation: recall@k and MRR across chunking strategies and search modes.

Each "letters" question in eval/questions.jsonl has known correct letter years.
Two hit definitions:
  * by year (spec requirement): the top-k contains a chunk from a letter of the correct year;
  * by chunk (stricter): the top-k contains a chunk that actually holds the evidence phrase.
    Fairer for comparing chunking: the year can be "guessed" from a neighbouring paragraph.

recall@k = share of questions with a hit in the top-k.
MRR = mean of 1/rank of the first hit (0 if none).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.db.database import DictPool
from app.rag.retrieval import Mode, retrieve

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def load_questions(types: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if types is None or r["type"] in types]


@dataclass(frozen=True)
class RetrievalScore:
    recall_year: float
    mrr_year: float
    recall_chunk: float
    mrr_chunk: float
    n: int


def first_hit_rank(flags: list[bool]) -> int | None:
    """1-based rank of the first True, or None. Separate so it can be tested without a DB."""
    return next((i for i, f in enumerate(flags, start=1) if f), None)


def score_ranks(ranks: list[int | None], k: int) -> tuple[float, float]:
    """Compute (recall@k, MRR) from first-hit ranks."""
    n = len(ranks)
    recall = sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None and r <= k) / n
    return recall, mrr


async def evaluate_retrieval(
    pool: DictPool,
    strategy: str,
    mode: Mode,
    k: int = 5,
    questions: list[dict] | None = None,
    idf_filter: bool = True,
) -> RetrievalScore:
    questions = questions if questions is not None else load_questions({"letters"})
    year_ranks: list[int | None] = []
    chunk_ranks: list[int | None] = []
    for q in questions:
        hits = await retrieve(pool, q["question"], k=k, mode=mode, strategy=strategy, idf_filter=idf_filter)
        expected = set(q["expected_years"])
        evidence = re.compile(q["evidence_regex"], re.IGNORECASE)
        year_ranks.append(first_hit_rank([h.year in expected for h in hits]))
        chunk_ranks.append(first_hit_rank([bool(evidence.search(h.content)) for h in hits]))
    recall_y, mrr_y = score_ranks(year_ranks, k)
    recall_c, mrr_c = score_ranks(chunk_ranks, k)
    return RetrievalScore(recall_y, mrr_y, recall_c, mrr_c, len(questions))
