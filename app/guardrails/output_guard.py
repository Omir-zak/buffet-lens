"""Output guard: nothing resembling a financial recommendation may reach the user.

If the answer contains a recommendation, price forecast or portfolio advice, the model REWRITES
it without them; if a violation remains after that, the answer is REPLACED with a safe template.

Layer 1 is rules (free, fast). Layer 2 is an LLM rewrite (tokens are spent only when a rule
fires). Known limitation: the rules miss unusually phrased recommendations ("looks like an
attractive entry point right now"); these false negatives are measured by the eval.
"""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.llm.client import Usage, load_prompt

SAFE_FALLBACK_MESSAGE = (
    "I can't provide buy, sell or hold recommendations, price targets or personalized portfolio advice. "
    "What I can do is explain what Buffett's letters say about the kind of business involved and show "
    "current market data, so you can form your own view. Try asking about a specific company or principle."
)
FUTURE_YEAR_FROM = 2027  # the current year is 2026: anything about 2027+ is a forecast

ADVICE_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "you_should_trade": r"\byou (should|could|might want to|may want to|ought to) (consider )?(buy|buying|sell|selling|hold|holding|short|shorting|invest|investing|add|accumulate|trim|avoid buying)\b",
        "we_recommend": r"\b(i|we) (would )?(recommend|suggest|advise|urge)\b",
        "rating": r"\b(strong |a )?(buy|sell|hold) (rating|recommendation|signal)\b|\b(overweight|underweight|outperform|underperform)\b rating",
        "price_target": r"\b(price target|target price|fair value (estimate )?of \$|12-month target)\b",
        # Only a direct "buy/sell THIS now" instruction (imperative at sentence start or with explicit timing).
        # An earlier version matched descriptions ("Mr. Market is willing to buy your shares or sell
        # them to you") and replaced a correct answer with a refusal: a false positive found in eval.
        "buy_now": r"(?:(?:^|[.!?]\s+)(?:buy|sell|short|accumulate)\s+(?:it|them|the stock|this stock|these shares|shares)\b)|(?:\b(?:buy|sell|short|accumulate)\s+(?:it|them|the stock|this stock|these shares|shares\s+)?(?:now|today|immediately|right away)\b)",
        "attractive_entry": r"\b(attractive|good|great|ideal) (entry|buying) (point|opportunity|price)\b|\bworth buying\b|\bshould be bought\b",
        "allocation": r"\ballocate \d+\s?%|\b\d+\s?%\s+(of your|in) (portfolio|stocks|bonds)\b",
    }.items()
}
# Value forecast: a future year (2027+) in the same sentence as an amount, price or market cap.
FUTURE_YEAR = re.compile(r"\b(20(2[7-9]|[3-9]\d))\b")
VALUE_WORDS = re.compile(
    r"\$|\btrillion\b|\bbillion\b|market cap|market value|share price|stock price|\bworth\b|net worth", re.IGNORECASE
)


@dataclass(frozen=True)
class OutputCheck:
    violations: list[str]

    @property
    def clean(self) -> bool:
        return not self.violations


def check_output_rules(text: str) -> OutputCheck:
    """Layer 1: rules. Returns the rules that fired (empty means no violations)."""
    found = [name for name, pattern in ADVICE_PATTERNS.items() if pattern.search(text)]
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if FUTURE_YEAR.search(sentence) and VALUE_WORDS.search(sentence):
            found.append("future_valuation_forecast")
            break
    return OutputCheck(found)


async def apply_output_guard(
    text: str, generate: Callable[..., Awaitable[tuple[str, Usage]]], model: str | None
) -> tuple[str, list[str], str, Usage]:
    """Apply the guard. Returns (text, violations, action, token usage).

    action: "pass" (clean), "rewritten" (rewritten by the model) or "replaced" (replaced by the template).
    """
    check = check_output_rules(text)
    if check.clean:
        return text, [], "pass", Usage()

    messages: list[BaseMessage] = [
        SystemMessage(content=load_prompt("output_rewrite.md")),
        HumanMessage(content=f"Violations detected: {', '.join(check.violations)}\n\nText to rewrite:\n{text}"),
    ]
    rewritten, usage = await generate(model, messages, temperature=0.0)
    if check_output_rules(rewritten).clean and rewritten.strip():
        return rewritten.strip(), check.violations, "rewritten", usage
    return SAFE_FALLBACK_MESSAGE, check.violations, "replaced", usage
