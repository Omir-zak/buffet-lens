"""Split letters into chunks for retrieval.

An embedding model maps a whole input to one vector, so a 100k-character letter would give a vague
average that matches nothing specific, while tiny pieces lose context. Chunk size has to be tuned on
data, so several strategies are defined here and compared by eval.
"""

import re
from dataclasses import dataclass
from typing import Literal

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter


@dataclass(frozen=True)
class Chunk:
    year: int
    position: int  # index of the chunk within the letter
    text: str

    @property
    def id(self) -> str:
        return f"{self.year}-{self.position:03d}"  # e.g. 1988-042


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    kind: Literal["fixed", "recursive"]
    size: int  # max characters per chunk
    overlap: int  # characters repeated between adjacent chunks


# fixed: cuts at an exact character count, ignoring meaning (may split a sentence).
# recursive: prefers paragraph, then line, then sentence, then word boundaries, i.e. the
#            largest boundary that still fits the size.
# overlap keeps a thought that straddles a boundary intact in at least one chunk.
STRATEGIES: dict[str, ChunkingStrategy] = {
    s.name: s
    for s in [
        ChunkingStrategy("fixed_1000", "fixed", 1000, 200),
        ChunkingStrategy("recursive_500", "recursive", 500, 100),
        ChunkingStrategy("recursive_1000", "recursive", 1000, 150),
        ChunkingStrategy("recursive_2000", "recursive", 2000, 300),
    ]
}

DOT_LEADERS = re.compile(r"(?:\s?\.){4,}")  # "......." and ". . . . ." from tables


TABLE_ROW_MAX_ALPHA = 0.45  # a table row has a letter share below this
TABLE_ROW_MIN_DIGITS = 3
TABLE_BLOCK_MIN_ROWS = 3  # a table is several such rows in a row; a single one may be part of a sentence


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 8:
        return False
    if not any(c.isalpha() for c in stripped):  # a rule of dashes or underscores between table rows
        return True
    letters = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    return letters / len(stripped) < TABLE_ROW_MAX_ALPHA and digits >= TABLE_ROW_MIN_DIGITS


def strip_table_blocks(text: str) -> str:
    """Remove blocks of three or more consecutive table-like lines, keeping the surrounding prose.

    Dropping whole chunks with under 55% letters also lost prose next to a table (a paragraph about
    houses in the 2010 letter), so only the table rows are cut now. The three-in-a-row rule avoids
    cutting a line that is merely a fragment of a normal sentence ("$19.46 to $2,974.52, or").
    """
    lines = text.split("\n")
    flags = [_looks_like_table_row(line) for line in lines]
    keep = [True] * len(lines)
    i = 0
    while i < len(lines):
        if flags[i]:
            j = i
            while j < len(lines) and flags[j]:
                j += 1
            if j - i >= TABLE_BLOCK_MIN_ROWS:
                for k in range(i, j):
                    keep[k] = False
            i = j
        else:
            i += 1
    return "\n".join(line for line, kept in zip(lines, keep, strict=True) if kept)


def clean_for_chunking(text: str) -> str:
    """Strip dot leaders and table blocks: they carry no meaning but take up chunk space."""
    return strip_table_blocks(DOT_LEADERS.sub(" ", text))


def alpha_ratio(text: str) -> float:
    """Share of letters in the text: low for numeric tables, high for prose."""
    return sum(c.isalpha() for c in text) / max(len(text), 1)


def is_prose(text: str, min_alpha_ratio: float = 0.55) -> bool:
    """Reject chunks made almost entirely of numbers and symbols (financial tables)."""
    return alpha_ratio(text) >= min_alpha_ratio


def _make_splitter(strategy: ChunkingStrategy) -> TextSplitter:
    if strategy.kind == "fixed":
        # separator="" means cut anywhere, strictly by length.
        return CharacterTextSplitter(
            separator="", chunk_size=strategy.size, chunk_overlap=strategy.overlap, keep_separator=False
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=strategy.size,
        chunk_overlap=strategy.overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_letter(year: int, text: str, strategy: ChunkingStrategy) -> tuple[list[Chunk], int]:
    """Chunk one letter. Returns (chunks, number of chunks dropped as tables)."""
    pieces = _make_splitter(strategy).split_text(clean_for_chunking(text))
    chunks: list[Chunk] = []
    dropped = 0
    for piece in pieces:
        # Replace in-paragraph line breaks (layout leftovers) with spaces so quotes read better.
        piece = re.sub(r"\s*\n\s*", " ", piece).strip()
        if not piece:
            continue
        if not is_prose(piece):
            dropped += 1
            continue
        # position counts kept chunks only, so ids stay consecutive: 1988-000, 1988-001, ...
        chunks.append(Chunk(year=year, position=len(chunks), text=piece))
    return chunks, dropped
