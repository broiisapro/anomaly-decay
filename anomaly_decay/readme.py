"""Generate and verify the results block in ``README.md``.

The README's headline numbers are never typed by hand. :func:`render_block` reads
``results/main.csv``, ``decay.csv``, ``coverage.csv`` and ``bootstrap.csv`` through
:func:`anomaly_decay.io.read_results` (the same schema-typed reader the report uses) and the
paper's reference values from ``config/original_paper.yaml``, then formats each number with
the report's own helpers from :mod:`anomaly_decay.reporting`. The block lives between
``<!-- results:start -->`` and ``<!-- results:end -->``.

CLI: ``python -m anomaly_decay.readme`` rewrites the block from the committed results;
``python -m anomaly_decay.readme --check`` fails when the block is stale or when a number
has been typed anywhere else in the README. Outside the block, code spans and fenced code
blocks are ignored, and the only bare numbers allowed are four-digit years (citations).
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import polars as pl

from anomaly_decay import reporting
from anomaly_decay.io import read_results

__all__ = [
    "END_MARKER",
    "START_MARKER",
    "check_readme",
    "find_stray_numbers",
    "main",
    "render_block",
    "replace_block",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
README_PATH: Path = REPO_ROOT / "README.md"
START_MARKER: str = "<!-- results:start -->"
END_MARKER: str = "<!-- results:end -->"

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_NUMBER_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _number(df: pl.DataFrame, column: str, **conditions: str) -> float | None:
    """Numeric cell of *df* (a results file) selected by *conditions*, or ``None``."""
    value = reporting.lookup(df, column, **conditions)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{column}: expected a number, got {type(value).__name__}")
    return float(value)


def _count(df: pl.DataFrame, column: str, **conditions: str) -> int | None:
    """Integer cell of *df*, accepting the string values stored in ``coverage.csv``."""
    value = reporting.lookup(df, column, **conditions)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{column}: expected an integer, got {type(value).__name__}")


def _text(df: pl.DataFrame, column: str, **conditions: str) -> str | None:
    """Text cell of *df* selected by *conditions*, or ``None``."""
    value = reporting.lookup(df, column, **conditions)
    return None if value is None else str(value)


def _section(paper: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Sub-mapping *key* of ``config/original_paper.yaml``, validated."""
    value = paper.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config/original_paper.yaml: {key!r} must be a mapping")
    return cast("Mapping[str, Any]", value)


def _paper_number(section: Mapping[str, Any], key: str) -> float:
    """Numeric *key* of an original-paper section, validated."""
    value = section.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"config/original_paper.yaml: {key!r} must be a number")
    return float(value)


def _paper_term(section: Mapping[str, Any], term: str, column: str) -> float:
    """One coefficient or standard error from a Table II term block."""
    block = section.get(term)
    if not isinstance(block, Mapping):
        raise ValueError(f"config/original_paper.yaml: {term!r} must be a mapping")
    return _paper_number(cast("Mapping[str, Any]", block), column)


def _fmt_count(value: int | None) -> str:
    """Integer with thousands separators, or ``n/a``."""
    return reporting.NA_TEXT if value is None else f"{value:,}"


def _fmt_plain(value: int | None) -> str:
    """Integer without separators (identifiers such as seeds), or ``n/a``."""
    return reporting.NA_TEXT if value is None else str(value)


def _paragraph(text: str, width: int = 100) -> list[str]:
    """Wrap *text* to *width* columns without breaking words."""
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)


def render_block(results_dir: Path | None = None) -> str:
    """Render the results block from the results files and the paper reference values.

    *results_dir* defaults to :func:`anomaly_decay.reporting.results_dir` (``results/``
    under the repository root unless ``ANOMALY_DECAY_RESULTS`` is set). The return value
    includes both markers and no trailing newline. Missing results files raise
    :class:`FileNotFoundError`; missing values inside an existing file render ``n/a``.
    """
    base = reporting.results_dir() if results_dir is None else Path(results_dir)
    main = read_results("main.csv", base)
    decay = read_results("decay.csv", base)
    bootstrap = read_results("bootstrap.csv", base)
    coverage = read_results("coverage.csv", base)
    paper = reporting.original_paper()

    post_sample_coef = _number(main, "coef", spec="main", term="post_sample")
    post_publication_coef = _number(main, "coef", spec="main", term="post_publication")
    n_predictors = _count(main, "n_predictors", spec="main", term="post_sample")
    n_obs = _count(main, "n_obs", spec="main", term="post_sample")

    post_sample_decay = _number(
        decay, "decay_pct", spec="main", method="pooled_ratio", term="post_sample"
    )
    post_publication_decay = _number(
        decay, "decay_pct", spec="main", method="pooled_ratio", term="post_publication"
    )
    in_sample_mean = _number(
        decay, "in_sample_mean", spec="main", method="pooled_ratio", term="post_sample"
    )

    table2 = _section(paper, "table2_col1")
    headline = _section(paper, "headline")
    paper_post_sample = _paper_term(table2, "post_sample", "coef")
    paper_post_publication = _paper_term(table2, "post_publication", "coef")
    paper_predictors = _paper_number(table2, "n_predictors")
    paper_obs = _paper_number(table2, "n_obs")
    paper_decay_sample = _paper_number(headline, "post_sample_decline_pct")
    paper_decay_publication = _paper_number(headline, "post_publication_decline_pct")

    bootstrap_publication_low = _number(
        bootstrap, "ci_low", spec="bootstrap_main", statistic="decay_pct_post_publication"
    )
    bootstrap_publication_high = _number(
        bootstrap, "ci_high", spec="bootstrap_main", statistic="decay_pct_post_publication"
    )
    bootstrap_sample_low = _number(
        bootstrap, "ci_low", spec="bootstrap_main", statistic="decay_pct_post_sample"
    )
    bootstrap_sample_high = _number(
        bootstrap, "ci_high", spec="bootstrap_main", statistic="decay_pct_post_sample"
    )
    bootstrap_reps = _count(bootstrap, "reps", spec="bootstrap_main")
    bootstrap_block = _count(bootstrap, "block_length", spec="bootstrap_main")
    bootstrap_seed = _count(bootstrap, "seed", spec="bootstrap_main")

    vintage = _text(coverage, "value", item="vintage")
    first_month = reporting.lookup(coverage, "value", item="first_month")
    last_month = reporting.lookup(coverage, "value", item="last_month")
    total_predictors = _count(coverage, "value", item="n_predictors_op")
    published_by_2013 = _count(coverage, "value", item="n_published_by_2013")
    clear_predictors = _count(coverage, "value", item="n_clear")

    headline_text = (
        "**Headline.** Long-short anomaly returns are "
        f"{reporting.num(reporting.magnitude(post_sample_coef), 3)}% per month "
        f"{reporting.direction(post_sample_coef)} after a predictor's original sample ends "
        f"and {reporting.num(reporting.magnitude(post_publication_coef), 3)}% per month "
        f"{reporting.direction(post_publication_coef)} after publication "
        f"({_fmt_count(n_predictors)} predictors, {_fmt_count(n_obs)} observations)."
    )
    if in_sample_mean is None:
        decay_text = (
            "The pooled in-sample mean is unavailable, so percentage decay is not reported."
        )
    elif in_sample_mean > 0:
        decay_text = (
            f"Relative to the {reporting.num(in_sample_mean, 3)}% pooled in-sample mean, "
            f"returns {reporting.direction(post_sample_decay, positive='rise', negative='fall')} "
            f"by {reporting.pct(reporting.magnitude(post_sample_decay))} after the sample ends "
            f"and {reporting.direction(post_publication_decay, positive='rise', negative='fall')} "
            f"by {reporting.pct(reporting.magnitude(post_publication_decay))} after publication."
        )
    else:
        decay_text = (
            "The pooled in-sample mean is non-positive, so percentage decay is not reported."
        )

    rows = [
        (
            "After the original sample ends",
            _number(main, "coef", spec="main", term="post_sample"),
            _number(main, "se", spec="main", term="post_sample"),
            _number(main, "t", spec="main", term="post_sample"),
            post_sample_decay,
            paper_decay_sample,
        ),
        (
            "After publication",
            _number(main, "coef", spec="main", term="post_publication"),
            _number(main, "se", spec="main", term="post_publication"),
            _number(main, "t", spec="main", term="post_publication"),
            post_publication_decay,
            paper_decay_publication,
        ),
    ]
    table_lines = [
        "| Window | Return (% per month) | SE | t-statistic | Change vs in-sample "
        "| McLean–Pontiff (2016) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    table_lines.extend(
        f"| {label} | {reporting.num(coef, 3)} | {reporting.num(se, 3)} | "
        f"{reporting.tstat(t, 2)} | {reporting.pct(decay, 1)} | {reporting.pct(paper, 0)} |"
        for label, coef, se, t, decay, paper in rows
    )

    paper_text = (
        "The paper's column 1 reports "
        f"{reporting.direction(paper_post_sample, positive='increases', negative='declines')} "
        f"of {reporting.num(reporting.magnitude(paper_post_sample), 3)}% and "
        f"{reporting.num(reporting.magnitude(paper_post_publication), 3)}% per month on "
        f"{_fmt_count(int(paper_predictors))} predictors and {_fmt_count(int(paper_obs))} "
        f"observations with FGLS standard errors; the replication estimates "
        f"{reporting.num(post_sample_coef, 3)}% and {reporting.num(post_publication_coef, 3)}% "
        f"per month on {_fmt_count(n_predictors)} predictors and {_fmt_count(n_obs)} observations."
    )
    bootstrap_text = (
        f"A {_fmt_count(bootstrap_reps)}-replication moving-block bootstrap "
        f"(block length {_fmt_count(bootstrap_block)}, seed {_fmt_plain(bootstrap_seed)}) gives "
        f"95% confidence intervals of [{reporting.pct(bootstrap_sample_low)}, "
        f"{reporting.pct(bootstrap_sample_high)}] for the post-sample change and "
        f"[{reporting.pct(bootstrap_publication_low)}, "
        f"{reporting.pct(bootstrap_publication_high)}] for the post-publication change."
    )
    coverage_text = (
        f"The pinned OSAP vintage is {vintage or reporting.NA_TEXT}, covering "
        f"{reporting.month_label(first_month)} through {reporting.month_label(last_month)}: "
        f"{_fmt_count(total_predictors)} predictors in total, "
        f"{_fmt_count(published_by_2013)} published by 2013 and "
        f"{_fmt_count(clear_predictors)} classified as clear."
    )

    lines = [
        START_MARKER,
        *_paragraph(headline_text),
        "",
        *_paragraph(decay_text),
        "",
        *table_lines,
        "",
        *_paragraph(paper_text),
        "",
        *_paragraph(bootstrap_text),
        "",
        *_paragraph(coverage_text),
        END_MARKER,
    ]
    return "\n".join(lines)


def _extract_block(text: str) -> str | None:
    """The block from the start marker through the end marker, or ``None``."""
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + len(END_MARKER)]


def replace_block(text: str, block: str) -> str:
    """Replace the marked block in *text* with *block*; raise when markers are malformed.

    Both markers must appear exactly once and in order. Everything outside the markers is
    returned unchanged.
    """
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError(f"README must contain exactly one {START_MARKER!r} and one {END_MARKER!r}")
    start = text.index(START_MARKER)
    end = text.index(END_MARKER)
    if end < start:
        raise ValueError(f"README has {END_MARKER!r} before {START_MARKER!r}")
    return text[:start] + block + text[end + len(END_MARKER) :]


def find_stray_numbers(text: str) -> list[str]:
    """Bare digit runs outside the results block that are not four-digit years.

    Fenced code blocks and inline code spans are ignored (they hold commands and version
    numbers). The block itself is removed before scanning, because it is generated.
    """
    block = _extract_block(text)
    if block is not None:
        text = text.replace(block, "", 1)
    without_code = _FENCED_CODE_RE.sub("", text)
    without_code = _INLINE_CODE_RE.sub("", without_code)
    return [token for token in _NUMBER_RE.findall(without_code) if not _YEAR_RE.match(token)]


def check_readme(text: str, expected_block: str) -> list[str]:
    """Problems with *text* relative to *expected_block*; empty means up to date.

    Checks that both markers occur exactly once and in order, that the block equals the
    rendered one byte for byte, and that no number has been typed outside the block
    (four-digit years and code spans excepted).
    """
    problems: list[str] = []
    if text.count(START_MARKER) != 1 or text.count(END_MARKER) != 1:
        problems.append(
            f"expected exactly one {START_MARKER!r} and one {END_MARKER!r} in README.md"
        )
        return problems
    block = _extract_block(text)
    if block is None:
        problems.append(f"{END_MARKER!r} appears before {START_MARKER!r} in README.md")
        return problems
    if block != expected_block:
        problems.append(
            "the results block is out of date; run `uv run python -m anomaly_decay.readme`"
        )
    stray = find_stray_numbers(text)
    if stray:
        problems.append(
            "numbers outside the results block must not be typed by hand "
            f"(found: {', '.join(stray)})"
        )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: rewrite the README results block, or check it with ``--check``.

    ``--results-dir`` defaults to the report's results directory and ``--readme`` to
    ``README.md`` in the repository root. Writing requires valid markers; checking exits
    nonzero and prints every problem found.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.readme",
        description="Write or verify the generated results block in README.md.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the block and the rest of the README instead of rewriting",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=README_PATH,
        help=f"README path (default: {README_PATH})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="results directory (default: the report's results directory)",
    )
    args = parser.parse_args(argv)

    try:
        block = render_block(args.results_dir)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"readme: cannot render the results block: {exc}", file=sys.stderr)
        return 1

    readme_path = Path(args.readme)
    try:
        text = readme_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"readme: {readme_path} does not exist", file=sys.stderr)
        return 1

    if args.check:
        problems = check_readme(text, block)
        if problems:
            for problem in problems:
                print(f"readme: {problem}", file=sys.stderr)
            return 1
        print(f"readme: {readme_path} is up to date")
        return 0

    try:
        updated = replace_block(text, block)
    except ValueError as exc:
        print(f"readme: {exc}", file=sys.stderr)
        return 1
    readme_path.write_text(updated, encoding="utf-8")
    print(f"readme: updated the results block in {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
