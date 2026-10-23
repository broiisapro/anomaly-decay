"""Tests for the two-page research-assistant memo.

The memo is a Typst PDF built from the same results directory as the report, so these tests
pin the two properties a reader depends on: every number in the prose is computed at render
time (never typed), and the document fits the two-page budget. The page counter is
deliberately dependency-free: it counts page objects in the PDF bytes, with tests against a
synthetic document of known length and against rendered Typst PDFs, including the classic
traps — the ``/Pages`` tree node and Typst's ``/PageLabel`` objects.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from anomaly_decay import reporting

REPO_ROOT = reporting.REPO_ROOT
MEMO_PATH = REPO_ROOT / "report" / "memo.qmd"
FULL_DATA_PDF = REPO_ROOT / "report" / "_site" / "memo.pdf"
FIXTURE_PDF = REPO_ROOT / "data" / "processed" / "fixture" / "_site" / "memo.pdf"
FULL_DATA_INDEX_PDF = REPO_ROOT / "report" / "_site" / "index.pdf"
FIXTURE_INDEX_PDF = REPO_ROOT / "data" / "processed" / "fixture" / "_site" / "index.pdf"
RENDERED_INDEX_PDFS = (FULL_DATA_INDEX_PDF, FIXTURE_INDEX_PDF)
MAX_PAGES = 2

REPORT_URL = "https://broiisapro.github.io/anomaly-decay/"
REPOSITORY_URL = "https://github.com/broiisapro/anomaly-decay"

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
LINK_TARGET = re.compile(r"\]\([^)\s]*\)")
CITATION_YEAR = re.compile(r"\(\d{4}\)")
ANY_DIGIT = re.compile(r"\d")
PAGE_OBJECT = re.compile(rb"/Type\s*/Page(?![A-Za-z])")
PAGE_TREE_COUNT = re.compile(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)")
PAGE_GUARD = re.compile(r"counter\(page\)\.final\(\)\.first\(\)\s*<=\s*(\d+)")
SECTIONS = (
    "# Question",
    "# Data",
    "# Result",
    "# Robustness and Phase 2",
    "# Caveats",
    "# Next steps",
)


def count_pdf_pages(data: bytes) -> int:
    """Number of page objects in *data*.

    Counts ``/Type /Page`` (with or without the space) and rejects anything followed by a
    letter, so the ``/Pages`` tree node and Typst's ``/PageLabel`` objects are not counted.
    """
    return len(PAGE_OBJECT.findall(data))


def prose_without_numbers(text: str) -> str:
    """The part of *text* where a digit would be a typed result.

    Removes the front matter (the date line lives there), fenced code and inline code
    (including every ``{python}`` expression), headings, link targets and parenthetical
    citation years. Everything left is prose the reader sees.
    """
    body = FRONT_MATTER.sub("", text)
    body = CODE_FENCE.sub("", body)
    body = INLINE_CODE.sub("", body)
    body = HEADING.sub("", body)
    body = LINK_TARGET.sub("]", body)
    body = CITATION_YEAR.sub("", body)
    return body


def test_page_counter_counts_page_objects_not_tree_nodes() -> None:
    """A synthetic PDF with two pages, a page tree and two page labels counts two."""
    data = (
        b"%PDF-1.7\n"
        b"1 0 obj\n<</Type/Pages/Count 2/Kids[2 0 R 3 0 R]>>\nendobj\n"
        b"2 0 obj\n<</Type/Page/Parent 1 0 R/MediaBox[0 0 612 792]>>\nendobj\n"
        b"3 0 obj\n<</Type/Page/Parent 1 0 R/MediaBox[0 0 612 792]>>\nendobj\n"
        b"4 0 obj\n<</Type/PageLabel/S/D/St 1>>\nendobj\n"
        b"5 0 obj\n<</Type/PageLabel/S/D/St 2>>\nendobj\n"
    )
    assert count_pdf_pages(data) == 2


def test_page_counter_handles_spacing_variants() -> None:
    """The regex accepts both ``/Type/Page`` and ``/Type /Page`` spellings."""
    assert count_pdf_pages(b"<</Type/Page>>") == 1
    assert count_pdf_pages(b"<</Type /Page>>") == 1
    assert count_pdf_pages(b"<</Type/Pages/Count 0>>") == 0


@pytest.mark.parametrize("path", RENDERED_INDEX_PDFS, ids=["full-data", "fixture"])
def test_page_counter_agrees_with_the_page_tree_on_a_rendered_pdf(path: Path) -> None:
    """On a real Typst PDF the object count equals the page tree's own ``/Count``."""
    if not path.is_file():
        pytest.skip(f"{path} does not exist; run the render that produces it")
    data = path.read_bytes()
    match = PAGE_TREE_COUNT.search(data)
    assert match is not None, "no page tree found in the rendered report PDF"
    assert count_pdf_pages(data) == int(match.group(1))


def test_memo_contains_the_required_sections() -> None:
    text = MEMO_PATH.read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert f"\n{heading}\n" in text, f"missing section: {heading}"


def test_memo_header_names_the_author_and_links_the_report() -> None:
    text = MEMO_PATH.read_text(encoding="utf-8")
    front_matter = FRONT_MATTER.search(text)
    assert front_matter is not None
    header = front_matter.group(0)
    assert 'author: "Moksh Siruvani"' in header
    assert "date: today" in header
    assert "typst" in header
    assert "papersize: us-letter" in header
    assert "html" not in header
    assert REPORT_URL in text
    assert REPOSITORY_URL in text


def test_memo_prose_has_no_typed_numbers() -> None:
    """Every number in the memo body comes from an inline expression or a code cell."""
    prose = prose_without_numbers(MEMO_PATH.read_text(encoding="utf-8"))
    typed = ANY_DIGIT.findall(prose)
    assert not typed, f"typed digits in memo prose: {typed}"


def test_memo_inline_expressions_are_backticked() -> None:
    """A bare ``{python}`` renders literally, so every inline expression needs backticks."""
    text = CODE_FENCE.sub("", MEMO_PATH.read_text(encoding="utf-8"))
    leftover = re.sub(r"`\{python\} [^`]+`", "", text)
    assert "{python}" not in leftover


def test_memo_reads_results_only_through_reporting() -> None:
    text = MEMO_PATH.read_text(encoding="utf-8")
    assert "read_csv" not in text
    assert "read_parquet" not in text
    assert "reporting.load" in text


def test_memo_carries_a_render_time_page_guard() -> None:
    """The document asserts its own page budget, so a render cannot ship an overlong memo.

    The fixture render runs without this test file, so the guard lives in the document
    itself: every render (local, Pages, CI fixture) fails when the memo exceeds two pages.
    """
    text = MEMO_PATH.read_text(encoding="utf-8")
    match = PAGE_GUARD.search(text)
    assert match is not None, "the memo has no Typst page-budget assertion"
    assert int(match.group(1)) == MAX_PAGES


@pytest.mark.parametrize(
    ("name", "path"),
    [("full-data", FULL_DATA_PDF), ("fixture", FIXTURE_PDF)],
)
def test_memo_pdf_fits_the_two_page_budget(name: str, path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"{path} does not exist; run the render that produces it")
    pages = count_pdf_pages(path.read_bytes())
    assert pages <= MAX_PAGES, f"{name} memo is {pages} pages, over the {MAX_PAGES}-page budget"
