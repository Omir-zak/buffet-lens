"""Per-question eval progress persistence.

A full run costs hundreds of thousands of tokens and the free tier has a daily limit, so each
question's result is written to disk immediately and a re-run resumes where it stopped.

To avoid mixing results from different system versions, a fingerprint of the code that affects
answers (prompts, graph, guardrails, retrieval) is stored alongside. If any such file changes, the
progress is stale and starts over. `--fresh` discards progress unconditionally.
"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PROGRESS_DIR = Path(__file__).parent / ".progress"
FINGERPRINT_GLOBS = (
    "prompts/*.md",
    "agents/*.py",
    "guardrails/*.py",
    "rag/*.py",
    "llm/*.py",
    "tools/*.py",
    "config.py",
)


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]


def _semantic_bytes(path: Path) -> bytes:
    """File content that affects behaviour: Python files as AST without docstrings (comments never
    reach the AST), everything else (prompts) verbatim."""
    if path.suffix != ".py":
        return path.read_bytes()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    _strip_docstrings(tree)
    return ast.dump(tree).encode()


def code_fingerprint() -> str:
    """Hash of the code the system's answers depend on; editing comments or docstrings keeps it stable."""
    digest = hashlib.sha256()
    for pattern in FINGERPRINT_GLOBS:
        for path in sorted(APP_DIR.glob(pattern)):
            digest.update(path.name.encode())
            digest.update(_semantic_bytes(path))
    return digest.hexdigest()[:16]


class Progress:
    """Key -> result mapping persisted to disk after every write."""

    def __init__(self, part: str, fresh: bool = False) -> None:
        self.path = PROGRESS_DIR / f"{part}.json"
        self.fingerprint = code_fingerprint()
        self.items: dict[str, Any] = {}
        if not fresh and self.path.exists():
            stored = json.loads(self.path.read_text(encoding="utf-8"))
            if stored.get("fingerprint") == self.fingerprint:
                self.items = stored.get("items", {})
            else:
                print(f"[прогресс части {part} устарел: код изменился, начинаю заново]")

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def get(self, key: str) -> Any:
        return self.items.get(key)

    def put(self, key: str, value: Any) -> None:
        self.items[key] = value
        PROGRESS_DIR.mkdir(exist_ok=True)
        payload = {"fingerprint": self.fingerprint, "items": self.items}
        self.path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def __len__(self) -> int:
        return len(self.items)
