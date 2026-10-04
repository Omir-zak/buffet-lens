from app.rag.chunking import (
    STRATEGIES,
    alpha_ratio,
    chunk_letter,
    clean_for_chunking,
    is_prose,
    strip_table_blocks,
)

PARAGRAPH = (
    "We prefer businesses with durable competitive advantages and honest, able managers. "
    "Price is what you pay and value is what you get. "
)


def make_text(paragraphs: int = 40) -> str:
    return "\n\n".join(PARAGRAPH * 2 for _ in range(paragraphs))


def test_chunks_never_exceed_the_configured_size():
    for strategy in STRATEGIES.values():
        chunks, _ = chunk_letter(1988, make_text(), strategy)
        assert chunks, strategy.name
        assert all(len(c.text) <= strategy.size for c in chunks), strategy.name


def test_chunk_ids_are_sequential_and_include_the_year():
    chunks, _ = chunk_letter(1988, make_text(), STRATEGIES["recursive_1000"])
    assert chunks[0].id == "1988-000"
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_recursive_splitter_keeps_paragraphs_whole_while_fixed_cuts_them():
    paragraphs = [
        f"Paragraph {i}. " + "Buffett prefers simple, understandable businesses with strong economics. " * 3
        for i in range(30)
    ]
    text = "\n\n".join(paragraphs)

    recursive, _ = chunk_letter(1988, text, STRATEGIES["recursive_500"])
    assert all(c.text.startswith("Paragraph") for c in recursive)  # every chunk starts at a paragraph boundary

    fixed, _ = chunk_letter(1988, text, STRATEGIES["fixed_1000"])
    assert any(not c.text.startswith("Paragraph") for c in fixed)  # fixed-size splitting cuts through paragraphs


def test_overlap_repeats_text_between_neighbouring_chunks():
    chunks, _ = chunk_letter(1988, make_text(), STRATEGIES["fixed_1000"])
    first, second = chunks[0].text, chunks[1].text
    assert first[-50:] in second or second[:50] in first


def test_table_blocks_are_cut_out_and_never_become_chunks():
    table = "\n".join(f"Item {i} .......... 1,234 5,678 9,012 (3,456)" for i in range(60))
    chunks, _ = chunk_letter(1988, table, STRATEGIES["recursive_1000"])
    assert chunks == []  # the text consisted entirely of a table


def test_prose_next_to_a_table_is_kept_while_the_table_rows_are_removed():
    """Regression: a "prose + table" chunk used to be dropped entirely, losing paragraphs
    (e.g. about houses in the 2010 letter)."""
    prose = "But a house can be a nightmare if the buyer's eyes are bigger than his wallet. " * 3
    table = "\n".join(f"{1980 + i}  {i + 3.5:.1f}  {100 + i:.1f}  {i * 2.2:.1f}  {40 - i:.1f}" for i in range(8))
    text = f"{prose}\n{table}\nOur goal is to keep it simple and honest."
    chunks, _ = chunk_letter(2010, text, STRATEGIES["recursive_1000"])
    joined = " ".join(c.text for c in chunks)
    assert "nightmare" in joined and "keep it simple" in joined
    assert "1983" not in joined  # table rows are stripped out


def test_a_single_numeric_line_inside_a_sentence_is_not_mistaken_for_a_table():
    text = "Book value per share has grown from\n$19.46 to $2,974.52, or\nat a rate of 23.0% compounded annually."
    assert "$19.46 to $2,974.52" in strip_table_blocks(text)


def test_two_numeric_rows_are_not_enough_to_be_a_table():
    text = "Intro sentence about results.\n1981  3.8 106.0 6.5 9.6\n1982  4.4 109.8 8.4 6.4\nClosing sentence."
    assert strip_table_blocks(text) == text


def test_dot_leaders_are_removed():
    assert "...." not in clean_for_chunking("Revenue ............... 100")


def test_alpha_ratio_separates_prose_from_numbers():
    assert alpha_ratio("The quick brown fox jumps") > 0.7
    assert alpha_ratio("1,234 5,678 (9,012) 3,456") < 0.1


def test_is_prose_rejects_number_heavy_text():
    assert is_prose("Our insurance operations produced an underwriting profit again this year.")
    assert not is_prose("1,234 5,678 (9,012) 3,456 7,890 (1,111)")


def test_newlines_inside_a_chunk_become_spaces():
    chunks, _ = chunk_letter(
        1988, "First line of a paragraph\nstill the same paragraph. " * 30, STRATEGIES["recursive_1000"]
    )
    assert all("\n" not in c.text for c in chunks)
