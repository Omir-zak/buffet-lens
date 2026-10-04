import pytest

from app.rag.retrieval import clean_query


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"financial weapons of mass destruction" Buffett', "financial weapons of mass destruction"),
        ("“Mr. Market” and Warren Buffett", "Mr. Market and"),
        (
            "Buffett's views on airlines",
            "Buffett's views on airlines",
        ),  # a word with the 's suffix does not count as matched
        ("margin of safety", "margin of safety"),
    ],
)
def test_quotes_and_corpus_wide_words_are_removed_from_queries(raw, expected):
    assert clean_query(raw) == expected


def test_a_query_made_only_of_corpus_wide_words_is_left_alone():
    assert clean_query("Berkshire letters") == "Berkshire letters"
    assert clean_query("Buffett") == "Buffett"
