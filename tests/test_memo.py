"""Tests for the two-page research-assistant memo.

The memo is a Typst PDF built from the same results directory as the report, so these
tests pin the property a reader depends on most: every number in the prose is computed at
render time, never typed.
"""

from __future__ import annotations

import re

from anomaly_decay import reporting

REPO_ROOT = reporting.REPO_ROOT
MEMO_PATH = REPO_ROOT / "report" / "memo.qmd"

REPORT_URL = "https://broiisapro.github.io/anomaly-decay/"
REPOSITORY_URL = "https://github.com/broiisapro/anomaly-decay"

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
LINK_TARGET = re.compile(r"\]\([^)\s]*\)")
CITATION_YEAR = re.compile(r"\(\d{4}\)")
ANY_DIGIT = re.compile(r"\d")
SECTIONS = (
    "# Question",
    "# Data",
    "# Result",
    "# Robustness and Phase 2",
    "# Caveats",
    "# Next steps",
)


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
