from app.rag.loader import FRAGMENT_LIMIT, extract_pdf_text, fragment_ratio

GOOD = "If only one variable is key to a decision, and the variable has a 90% chance of going your way, take it."
BROKEN = "If only on e variable is k ey t o a d ecision, an d th e variable has a 90% chance of go ing y our w ay, t ake i t."


def test_fragment_ratio_separates_clean_text_from_text_with_broken_spaces():
    assert fragment_ratio(GOOD) < FRAGMENT_LIMIT
    assert fragment_ratio(BROKEN) > FRAGMENT_LIMIT  # defect found in the PDFs of the 2004 and 2005 letters


def test_articles_a_and_i_are_not_counted_as_fragments():
    assert fragment_ratio("I think a lot about a few things I know") == 0.0


def test_empty_text_is_safe():
    assert fragment_ratio("") == 0.0 and fragment_ratio("12345 ...") == 0.0


class FakePage:
    def __init__(self, default: str, layout: str) -> None:
        self.default, self.layout = default, layout

    def extract_text(self, **kwargs) -> str:
        return self.layout if kwargs.get("extraction_mode") == "layout" else self.default


class FakeReader:
    def __init__(self, default: str, layout: str) -> None:
        self.pages = [FakePage(default, layout)]


def test_layout_mode_is_used_only_when_the_default_text_is_broken():
    text, mode = extract_pdf_text(FakeReader(BROKEN, GOOD))
    assert mode == "layout" and text == GOOD


def test_default_mode_is_kept_for_clean_text_without_trying_layout():
    text, mode = extract_pdf_text(FakeReader(GOOD, "SHOULD NOT BE USED"))
    assert mode == "default" and text == GOOD


def test_if_layout_is_not_better_the_default_text_is_kept():
    text, mode = extract_pdf_text(FakeReader(BROKEN, BROKEN + " even worse t o r ead"))
    assert mode == "default" and text == BROKEN
