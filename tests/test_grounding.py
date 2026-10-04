"""The "grounded" decision: the core hallucination defense, so it is covered densely."""

from app.agents.nodes import (
    NO_INFO_MESSAGE,
    Evidence,
    build_evidence,
    decide_answer,
    extract_citations,
    normalize_text,
)
from app.rag.answer import LLMAnswer, decide_grounding, parse_llm_answer
from app.rag.retrieval import Retrieved, reciprocal_rank_fusion

EVIDENCE = [
    Evidence(
        "1988-042",
        "letter",
        "(letter 1988) Coke is a wonderful business.",
        {"year": 1988, "passage": "Coke is a wonderful business."},
    ),
    Evidence("M:KO", "market", "Coca-Cola (KO) Trailing P/E: 25.88", {"title": "Coca-Cola"}),
]


def test_answer_with_a_valid_citation_is_grounded():
    answer, grounded, sources = decide_answer("Coke is wonderful [1988-042].", EVIDENCE)
    assert grounded and [s.id for s in sources] == ["1988-042"]
    assert answer.startswith("Coke")


def test_answer_without_any_citation_is_rejected():
    answer, grounded, sources = decide_answer("Coke is wonderful.", EVIDENCE)
    assert (answer, grounded, sources) == (NO_INFO_MESSAGE, False, [])


def test_citation_of_an_unknown_id_does_not_count():
    _, grounded, _ = decide_answer("Made up [9999-001].", EVIDENCE)
    assert grounded is False


def test_model_refusal_is_not_grounded_even_if_it_cites_something():
    refusal = "The letters and data available do not contain enough information to answer this. [1988-042]"
    assert decide_answer(refusal, EVIDENCE)[1] is False


def test_phrase_in_the_middle_of_a_real_answer_does_not_cause_refusal():
    text = "The letters do not contain enough information on Apple itself, but price matters to Buffett [1988-042]."
    assert decide_answer(text, EVIDENCE)[1] is True


def test_regression_fullwidth_brackets_are_understood():
    """Real project bug: a reasoning model wrapped the citation in 【】 and a correct answer was discarded."""
    answer, grounded, sources = decide_answer("P/E is 25.88【M:KO】.", EVIDENCE)
    assert grounded and sources[0].id == "M:KO"
    assert "【" not in answer


def test_multiple_ids_in_one_bracket_and_duplicates():
    assert extract_citations("Both [1988-042, M:KO] and again [1988-042].", {"1988-042", "M:KO"}) == [
        "1988-042",
        "M:KO",
    ]


def test_normalize_text_fixes_exotic_characters():
    assert normalize_text("a b‑c") == "a b-c"


def test_failed_tools_are_reported_but_never_become_citable_evidence():
    market = [{"ok": False, "ticker": "ZZZ", "error": "Unknown ticker"}]
    evidence, problems = build_evidence([], market, None)
    assert evidence == [] and "ZZZ" in problems[0]


def test_rag_decision_requires_answerable_and_a_real_cited_passage():
    passages = [Retrieved("1988-042", 1988, "text", 0.7, 0.7)]
    assert decide_grounding(LLMAnswer(answerable=True, answer="x", used_ids=["1988-042"]), passages)[0] is True
    assert decide_grounding(LLMAnswer(answerable=True, answer="x", used_ids=["nope"]), passages)[0] is False
    assert decide_grounding(LLMAnswer(answerable=False, answer="x", used_ids=["1988-042"]), passages)[0] is False


def test_llm_json_parsing_tolerates_code_fences_and_rejects_garbage():
    fenced = '```json\n{"answerable": true, "answer": "ok", "used_ids": ["a"]}\n```'
    assert parse_llm_answer(fenced).answer == "ok"
    assert parse_llm_answer("not json at all") is None
    assert parse_llm_answer('{"answerable": "maybe"}') is None


def test_reciprocal_rank_fusion_rewards_documents_found_by_both_searches():
    a, b, c = (Retrieved(i, 1990, i, 0.5, 0.5) for i in "abc")
    fused = reciprocal_rank_fusion([[a, b], [c, a]], k=3)
    assert fused[0].chunk_id == "a"  # found in both lists
    assert {r.chunk_id for r in fused} == {"a", "b", "c"}


def test_best_excerpt_shows_the_supporting_sentence_not_the_beginning_of_the_chunk():
    """Regression: for source 1997-018 the beginning ("97% акций...") was shown, while the supporting text came later."""
    from app.agents.nodes import best_excerpt

    chunk = (
        "At the end of every year, about 97% of Berkshire's shares are held by the same investors who owned them at the start "
        "of the year. That makes them savers. They should therefore rejoice when markets decline and allow both us and our "
        'investees to deploy funds more advantageously. So smile when you read a headline that says "Investors lose as market '
        'falls." Though writers often forget this truism, there is a buyer for every seller and what hurts one necessarily helps the other.'
    )
    answer = "Investors should smile at market declines and use them to deploy funds more advantageously [1997-018]."
    excerpt = best_excerpt(chunk, answer)
    assert "deploy funds" in excerpt or "smile" in excerpt
    assert not excerpt.startswith("At the end of every year")
    assert len(excerpt) <= 320


def test_best_excerpt_handles_empty_and_short_passages():
    from app.agents.nodes import best_excerpt

    assert best_excerpt("", "anything") == ""
    assert best_excerpt("One short sentence.", "unrelated words") == "One short sentence."
