"""Download Buffett's letters and extract their text.

The letters are copyrighted, so they are fetched into data/letters, which is git-ignored
and never committed.
"""

import logging
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # PDFs spam font warnings

INDEX_URL = "https://berkshirehathaway.com/letters/letters.html"
LETTERS_DIR = Path(__file__).resolve().parents[2] / "data" / "letters"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational research project)"}


def list_letter_links(client: httpx.Client) -> dict[int, str]:
    """Parse the index page and return {year: letter URL}."""
    page = client.get(INDEX_URL)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    links: dict[int, str] = {}
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        # Only links whose text is exactly a year.
        if re.fullmatch(r"(19|20)\d{2}", text):
            links[int(text)] = urljoin(INDEX_URL, str(a["href"]))
    return links


STUB_MARKER = b"IMPORTANT NOTE"  # start of the 1998-2003 stub pages that link to the PDF


def _find_existing(year: int) -> Path | None:
    """Find an already downloaded letter; stub pages count as not downloaded."""
    for suffix in (".pdf", ".html"):
        path = LETTERS_DIR / f"{year}{suffix}"
        if path.exists():
            if suffix == ".html" and STUB_MARKER in path.read_bytes():
                path.unlink()  # a stub we saved earlier; not a letter
                continue
            return path
    return None


def _pdf_link_from_stub(html: str, base_url: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if href.lower().endswith(".pdf"):
            return urljoin(base_url, href)
    return None


def download_letters(delay_seconds: float = 1.0) -> dict[int, Path]:
    """Download all letters, skipping files already present (safe to rerun)."""
    LETTERS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for year, url in sorted(list_letter_links(client).items()):
            existing = _find_existing(year)
            if existing:
                paths[year] = existing
                continue

            response = client.get(url)
            response.raise_for_status()
            suffix = Path(url).suffix
            if suffix == ".html" and STUB_MARKER in response.content:
                # A stub page instead of the letter: fetch the PDF it links to.
                pdf_url = _pdf_link_from_stub(response.text, url)
                if pdf_url is None:
                    raise RuntimeError(f"{year}: на странице-заглушке нет ссылки на PDF ({url})")
                time.sleep(delay_seconds)
                response = client.get(pdf_url)
                response.raise_for_status()
                suffix = ".pdf"

            path = LETTERS_DIR / f"{year}{suffix}"
            path.write_bytes(response.content)
            paths[year] = path
            time.sleep(delay_seconds)  # be polite to the server
    return paths


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse runs of blank lines
    return text.strip()


TEXT_DIR = LETTERS_DIR.parent / "text"


def load_text(path: Path) -> str:
    """Letter text with an on-disk cache, since PDF parsing is slow."""
    cached = TEXT_DIR / f"{path.stem}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    text = extract_text(path)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(text.encode("utf-8"))
    return text


FRAGMENT_LIMIT = 0.05  # more than 5% single-character "words" signals broken spacing


def fragment_ratio(text: str) -> float:
    """Share of single-character "words" (other than a and I) among all words.

    Text-extraction quality check: normal English is around 2-3%, while text with stray spaces inside
    words ("on e variable is k ey t o a d ecision") is 8% or more. This exposed a defect in the 2004
    and 2005 letter PDFs, where default pypdf extraction inserted spaces inside words.
    """
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) == 1 and w.lower() not in ("a", "i")) / len(words)


def extract_pdf_text(reader: Any) -> tuple[str, str]:
    """Return the PDF text and the mode used ("default" or "layout").

    Try default extraction first. If it yields broken spacing (see fragment_ratio), try layout mode,
    which uses character positions on the page, and keep whichever result has fewer fragments.
    """
    default = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if fragment_ratio(default) <= FRAGMENT_LIMIT:
        return default, "default"
    layout = "\n\n".join(page.extract_text(extraction_mode="layout") or "" for page in reader.pages)
    if fragment_ratio(layout) < fragment_ratio(default):
        return layout, "layout"
    return default, "default"


def extract_text(path: Path) -> str:
    """Extract clean text from an HTML or PDF letter."""
    if path.suffix.lower() == ".pdf":
        text, _ = extract_pdf_text(PdfReader(BytesIO(path.read_bytes())))
        return _clean(text)
    # Old HTML pages are not always UTF-8, so fall back to cp1252.
    raw = path.read_bytes()
    try:
        html = raw.decode("utf-8")
    except UnicodeDecodeError:
        html = raw.decode("cp1252", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return _clean(soup.get_text("\n"))
