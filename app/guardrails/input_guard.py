"""Input guard: what happens to a question BEFORE it reaches the models.

A hybrid of two layers:
  1. Rules (regexes): instant, free, deterministic. They catch obvious injection attempts
     ("ignore previous instructions") and requests for advice. Downsides: they miss
     paraphrases and can fire on similar-looking words.
  2. LLM classifier: the router, already called for routing, also flags injection_suspected
     (see router.md). It understands meaning but costs tokens and can also be wrong.
The LLM layer needs no extra model call because it is built into the router.

Advice requests ("Should I buy NVIDIA?") are not blocked outright: the user can still benefit
from Buffett's principles and the data. They are only flagged, and the answer is additionally
checked by the output guard.
"""

import re
from dataclasses import dataclass

INJECTION_MESSAGE = (
    "I can't follow instructions that try to change how I work or reveal my configuration. "
    "I'm here to help research public companies through Warren Buffett's letters, market data and "
    "news. Ask me about a company, a Buffett principle or a valuation metric."
)

# Each rule has a readable name (for logs and debugging) and a regex.
INJECTION_RULES: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for name, pattern in {
        "ignore_instructions": r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|any|the)\b.{0,30}\b(instructions?|rules?|guidelines?|prompts?|polic(y|ies)|filters?)\b",
        "reveal_prompt": r"\b(reveal|show|print|display|repeat|output|tell me|translate|give me)\b.{0,50}\b(system prompt|initial instructions?|hidden instructions?|your instructions?|your prompt|everything above|configuration)\b",
        "reveal_secrets": r"\b(reveal|show|print|give|share)\b.{0,30}\b(api[ -]?keys?|secrets?|passwords?|credentials|tokens?)\b",
        "role_override": r"\b(you are now|from now on,? you|pretend (to be|you are|that)|act as (if|an? ))\b.{0,80}\b(dan|no restrictions|unrestricted|without (any )?(rules|restrictions|filters)|financial advisor|jailbroken)\b",
        "named_jailbreak": r"\b(DAN|developer mode|jailbreak|do anything now)\b",
        "fake_authority": r"\b(system override|admin override|safety filters? (are |have been )?(disabled|off)|developer (has )?disabled)\b",
        "obey_always": r"\b(never refuse|obey (the user|me) (completely|always)|without refusing)\b",
        "special_tokens": r"</?system>|\[/?INST\]|<\|im_(start|end)\|>",
        "repeat_after_me": r"\b(repeat after me|say exactly)\b.{0,40}\b(ignor|polic|rules)",
        "ignore_policy": r"\bignoring (any|all|your) (policy|policies|rules|instructions)\b",
    }.items()
}

ADVICE_RULES: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "should_i_trade": r"\bshould i (buy|sell|short|hold|invest|put|move)\b",
        "price_target": r"\b(price|target) (target|price)\b|\bgive me a .{0,20}target\b",
        "trade_now": r"\b(buy|sell|short) (it|them|this|that|these)?\s?(now|today|right now)\b",
        "which_stocks": r"\b(which|what) (\w+ )?(\d+ |three |two |five |some |a few )?(stocks?|shares|securities)\b.{0,30}\bshould\b",
        "allocation": r"\b(how (should|do) i (split|allocate|invest|diversify)|what (portfolio )?allocation|portfolio allocation|allocate my)\b",
        "good_buy": r"\b(is|are) .{1,60} a (good|great|safe) (buy|investment|bet)\b",
        "good_time": r"\b(is )?now a good time to (buy|sell|short|invest)\b",
        "entry_price": r"\b(entry|exit) (price|point)\b",
        "clear_recommendation": r"\b(clear|direct) (recommendation|buy|sell)\b|\bjust say (yes|no)\b",
        "tell_me_buy": r"\btell me (which|what) .{0,30}(buy|sell|invest)",
        # Requests to forecast a future price or value: not fulfilled (see the output guard).
        "forecast_value": r"\b(what|how much) (will|would|could|might) .{0,60}\b(be|cost|trade|worth|reach|earn|make)\b.{0,40}\b(in|by|at the end of|within|over the next) .{0,12}(20(2[7-9]|[3-9]\d)|\d+ (years?|months?)|next (year|decade))\b",
        "will_move": r"\bwill (?:[\w'.&-]+\s+){0,4}(?:double|triple|quadruple|crash|soar|collapse|rise|fall|drop|surge|reach \$?\d)",
        "where_trading": r"\bwhere do you (see|expect|think) .{0,40}\b(trading|price|stock|shares|market cap)\b",
        "predict_price": r"\b(predict|forecast|projection|outlook)\b.{0,40}\b(price|stock|share|earnings|market cap|valuation|net worth)\b",
    }.items()
}


@dataclass(frozen=True)
class InputVerdict:
    blocked: bool = False
    category: str | None = None  # "injection" | None
    rule: str | None = None  # which rule fired
    advice_seeking: bool = False


def check_input_rules(text: str) -> InputVerdict:
    """Layer 1: rules. Fast, no network access."""
    for name, pattern in INJECTION_RULES.items():
        if pattern.search(text):
            return InputVerdict(blocked=True, category="injection", rule=name)
    for name, pattern in ADVICE_RULES.items():
        if pattern.search(text):
            return InputVerdict(advice_seeking=True, rule=name)
    return InputVerdict()
