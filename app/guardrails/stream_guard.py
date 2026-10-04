"""Stream guard: checks the answer on the fly while tokens are still streaming to the client.

Streaming tokens as they are generated conflicts with the output guard, which must check the
whole answer (see output_guard.py): the text is already shown when a violation is found.
The approach has three parts:
  1. Tokens stream immediately (fast response).
  2. After each completed sentence, the text shown so far is checked with the output guard rules.
     On a violation the client gets a `retract` event ("erase what was shown") and no further
     tokens are sent.
  3. At the end the graph's final answer (after rewriting and the faithfulness check) is ALWAYS
     compared with what was shown; if it differs, a `replace` event carries the final text.
The user may briefly see the start of an answer that is later replaced, but the final answer
(the one stored and returned in `done`) has always passed all guards.
"""

import re
from dataclasses import dataclass, field

from app.guardrails.output_guard import check_output_rules

SENTENCE_END = re.compile(r"[.!?\n]")


@dataclass
class StreamGuard:
    """Accumulates the shown text and decides whether the next token may be shown."""

    shown: list[str] = field(default_factory=list)
    halted: bool = False
    violations: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.shown)

    def feed(self, token: str) -> str:
        """Accept the next token. Returns an action: "emit" (show it), "retract" (erase what was
        shown; do not show the token) or "drop" (stream already halted; silently discard)."""
        if self.halted:
            return "drop"
        self.shown.append(token)
        # Check only at sentence boundaries: cheaper and avoids false positives on partial text.
        if SENTENCE_END.search(token):
            check = check_output_rules(self.text)
            if not check.clean:
                self.halted = True
                self.violations = check.violations
                return "retract"
        return "emit"

    def needs_replace(self, final_answer: str) -> bool:
        """Does the final answer differ from what the client has already seen?"""
        return self.halted or self.text.strip() != final_answer.strip()
