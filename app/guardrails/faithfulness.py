"""Faithfulness guard: every claim must be backed by the retrieved sources.

Checks that need no LLM (free and reproducible):
  * numbers: an amount, percentage or ratio in the answer must appear in the sources
    (with rounding tolerance: "$368.5 billion" matches 368,513,318,912);
  * quotations: quoted text must appear verbatim (ignoring case and punctuation) in the
    sources, which catches invented "Buffett quotes".
A sentence with an unsupported number or quotation is REMOVED. If no source citation is left
afterwards, the answer is considered ungrounded.

Known limitations: the check does not understand meaning. A sentence with correct numbers but a
wrong conclusion passes, and a number computed by the model itself (e.g. a difference of two
figures) is flagged as unsupported.
"""

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[([^\[\]]+)\]")
NUMBER = re.compile(r"(?<![\w.])(\$)?\s?(\d[\d,]*(?:\.\d+)?)\s?(%|trillion|billion|million|thousand)?", re.IGNORECASE)
QUOTE = re.compile(r"[“\"]([^”\"]{20,}?)[”\"]")
SCALES = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
TOLERANCE = 0.03  # 3% tolerance: rounding "45.3" to "45" is not an error


@dataclass
class FaithfulnessResult:
    text: str
    removed: list[str] = field(default_factory=list)  # removed sentences (for debugging and eval)
    cited_ids: list[str] = field(default_factory=list)  # sources still cited in the text

    @property
    def grounded(self) -> bool:
        return bool(self.text.strip()) and bool(self.cited_ids)


def _parse_numbers(text: str, only_material: bool) -> list[float]:
    """Extract numbers with their scale (billion, %). only_material drops minor ones:
    enumerations ("top 3"), years and single-digit numbers without units."""
    text = CITATION.sub(" ", text)  # ignore numbers inside [1988-042]
    values: list[float] = []
    for dollar, raw, suffix in NUMBER.findall(text):
        digits = raw.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        is_year = "." not in digits and len(digits) == 4 and 1800 <= value <= 2100 and not (dollar or suffix)
        material = bool(dollar or suffix) or "." in digits or (len(digits) >= 3 and not is_year)
        if only_material and not material:
            continue
        values.append(value * SCALES.get((suffix or "").lower(), 1.0))
    return values


def _supported(number: float, evidence_numbers: list[float]) -> bool:
    return any(abs(number - e) <= TOLERANCE * max(abs(e), 1e-9) for e in evidence_numbers)


def _norm(text: str) -> str:
    """For comparing quotations: lowercase, no punctuation or extra whitespace."""
    text = text.lower().replace("‑", "-").replace("’", "'").replace("‘", "'")
    # Periods are stripped too: American style puts them INSIDE the quotes ("...destruction."), while the
    # letter has a comma or nothing there, which made genuine quotes look invented.
    text = re.sub(r"[^a-z0-9$% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unsupported_reasons(sentence: str, evidence_numbers: list[float], norm_corpus: str) -> list[str]:
    reasons: list[str] = []
    for number in _parse_numbers(sentence, only_material=True):
        if not _supported(number, evidence_numbers):
            reasons.append(f"number {number:g} not in sources")
            break
    for quoted in QUOTE.findall(sentence):
        # A quotation may contain ellipses: check each piece separately.
        for part in re.split(r"…|\.\.\.|\[\s*\.\.\.\s*\]", quoted):
            words = _norm(part).split()
            if len(words) >= 4 and _norm(part) not in norm_corpus:
                reasons.append("quotation not found in sources")
                break
    return reasons


def check_faithfulness(answer: str, evidence_texts: list[str], valid_ids: set[str]) -> FaithfulnessResult:
    """Remove sentences with unsupported numbers or quotations from the answer."""
    corpus = " ".join(evidence_texts)
    evidence_numbers = _parse_numbers(corpus, only_material=False)
    norm_corpus = _norm(corpus)

    kept: list[str] = []
    removed: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        if not sentence.strip():
            continue
        reasons = _unsupported_reasons(sentence, evidence_numbers, norm_corpus)
        (removed if reasons else kept).append(
            sentence.strip() if not reasons else f"{sentence.strip()} [{'; '.join(reasons)}]"
        )

    text = " ".join(kept)
    cited: list[str] = []
    for match in CITATION.finditer(text):
        for part in match.group(1).split(","):
            cid = part.strip()
            if cid in valid_ids and cid not in cited:
                cited.append(cid)
    return FaithfulnessResult(text=text, removed=removed, cited_ids=cited)
