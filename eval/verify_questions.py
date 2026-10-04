"""Check the eval set labels: expected years must match the actual letter text.

Run: python -m eval.verify_questions            check
     python -m eval.verify_questions --fix      recompute expected_years from evidence_regex

For "letters" questions, expected_years are the years whose letters contain the evidence phrase.
For "unanswerable" ones, the probe word must not appear in any letter.
This way ground truth comes from the text, not from memory.
"""

import json
import re
import sys
from pathlib import Path

from app.rag.loader import TEXT_DIR

QUESTIONS = Path(__file__).parent / "questions.jsonl"
UNANSWERABLE_PROBES = {  # words that must not appear in the letters (otherwise the question is answerable)
    "unanswerable-01": r"bitcoin",
    "unanswerable-02": r"ChatGPT",
    "unanswerable-03": r"Tesla",
    "unanswerable-04": r"cryptocurrenc",
    "unanswerable-05": r"toothpaste",
    "unanswerable-06": r"\b2031\b",
}


def load_texts() -> dict[int, str]:
    return {int(p.stem): re.sub(r"\s+", " ", p.read_text(encoding="utf-8")) for p in TEXT_DIR.glob("*.txt")}


def years_matching(texts: dict[int, str], pattern: str) -> list[int]:
    return [year for year, text in sorted(texts.items()) if re.search(pattern, text, flags=re.IGNORECASE)]


def main(fix: bool) -> int:
    texts = load_texts()
    if not texts:
        print("Нет текстов писем в data/text. Сначала: python -m scripts.ingest")
        return 2
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = 0
    for row in rows:
        if row["type"] == "letters":
            actual = years_matching(texts, row["evidence_regex"])
            if actual != row["expected_years"]:
                problems += 1
                print(f"x {row['id']}: ожидалось {row['expected_years']}, по тексту {actual}")
                if fix:
                    row["expected_years"] = actual
        elif row["id"] in UNANSWERABLE_PROBES:
            found = years_matching(texts, UNANSWERABLE_PROBES[row["id"]])
            if found:
                problems += 1
                print(f"x {row['id']}: слово из вопроса встречается в письмах {found}, это не вопрос «без ответа»")
    if fix:
        QUESTIONS.write_bytes(("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8"))
    print("Разметка согласована с текстом писем." if not problems else f"Найдено расхождений: {problems}")
    return 0 if not problems or fix else 1


if __name__ == "__main__":
    sys.exit(main("--fix" in sys.argv))
