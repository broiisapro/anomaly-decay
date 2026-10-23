"""Helpers the Quarto report uses to read results and format numbers.

The report reads only the results directory — never raw or processed data — and every
number in its prose is computed here or by the plain helper functions below, so no
result is ever typed by hand. The results directory is ``<repo>/results`` by default and
relative paths in ``ANOMALY_DECAY_RESULTS`` are resolved against the repository root,
never the process working directory: when Quarto renders, the kernel's working directory
is the document's directory (``report/``), which is not where results live.
"""

from __future__ import annotations

import math
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import yaml

from anomaly_decay import io as results_io
from anomaly_decay.schemas import RESULTS_SCHEMAS

if TYPE_CHECKING:
    from IPython.display import Image, Markdown

    from anomaly_decay.config import Spec

__all__ = [
    "NA_TEXT",
    "TERM_LABELS",
    "TERM_ORDER",
    "COVERAGE_ORDER",
    "block_method_label",
    "comparison_word",
    "count",
    "coverage_item_label",
    "coverage_rows",
    "coverage_value_text",
    "data_config",
    "decline_word",
    "direction",
    "exploratory_label",
    "factor",
    "figure",
    "figure_block",
    "grouped_table",
    "load",
    "load_optional",
    "lookup",
    "magnitude",
    "missing_tstat_label",
    "month_label",
    "num",
    "ordered",
    "original_paper",
    "pct",
    "phase2_preregistration",
    "pp_gap",
    "ratio",
    "ratio_pct",
    "results_dir",
    "specifications",
    "table",
    "term_label",
    "tstat",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_ENV_VAR: str = "ANOMALY_DECAY_RESULTS"
DEFAULT_RESULTS_DIRNAME: str = "results"
ORIGINAL_PAPER_PATH: Path = REPO_ROOT / "config" / "original_paper.yaml"
DATA_CONFIG_PATH: Path = REPO_ROOT / "config" / "data.toml"
PHASE2_PREREG_PATH: Path = REPO_ROOT / "config" / "phase2_preregistration.yaml"
SPECIFICATIONS_PATH: Path = REPO_ROOT / "config" / "specifications.yaml"
RASTER_EXTENSIONS: tuple[str, ...] = ("png", "jpg", "jpeg", "gif")
FIGURE_EXTENSIONS: tuple[str, ...] = (*RASTER_EXTENSIONS, "svg")
NA_TEXT: str = "n/a"

TERM_LABELS: dict[str, str] = {
    "post_sample": "Post-sample",
    "post_publication": "Post-publication",
    "post_sample_x_weak": "Post-sample × weak t",
    "post_publication_x_weak": "Post-publication × weak t",
    "post_sample_x_tstat": "Post-sample × in-sample t",
    "post_publication_x_tstat": "Post-publication × in-sample t",
    "post_publication_x_pub_after_2000": "Post-publication × published after 2000",
}
"""Prose labels for result terms; an unknown term renders its raw name."""

TERM_ORDER: tuple[str, ...] = (
    "post_sample",
    "post_sample_x_weak",
    "post_sample_x_tstat",
    "post_sample_x_pub_after_2000",
    "post_publication",
    "post_publication_x_weak",
    "post_publication_x_tstat",
    "post_publication_x_pub_after_2000",
)
"""Reading order for terms: the sample-end window first, then the publication window."""

COVERAGE_LABELS: dict[str, str] = {
    "vintage": "OSAP release",
    "first_month": "First return month",
    "last_month": "Last return month",
    "n_predictors_op": "Predictors with original-paper returns",
    "n_published_by_2013": "Predictors published by 2013",
    "n_clear": "Clear predictors",
    "n_predictors_deciles_ew": "Predictors with equal-weighted decile returns",
    "n_predictors_deciles_vw": "Predictors with value-weighted decile returns",
    "n_predictor_months_op": "Predictor-months with original-paper returns",
}
COVERAGE_ORDER: tuple[str, ...] = (
    "vintage",
    "first_month",
    "last_month",
    "n_predictors_op",
    "n_published_by_2013",
    "n_clear",
    "n_predictors_deciles_ew",
    "n_predictors_deciles_vw",
    "n_predictor_months_op",
)
MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MISSING_TSTAT_LABELS: dict[str, str] = {"exclude_and_count": "excluded and counted"}
BLOCK_METHOD_LABELS: dict[str, str] = {"moving_block_by_month": "moving block by month"}


def results_dir() -> Path:
    """Directory the report reads results from.

    ``ANOMALY_DECAY_RESULTS`` when set (relative paths resolve against the repository
    root), ``<repo>/results`` otherwise.
    """
    setting = os.environ.get(RESULTS_ENV_VAR)
    path = Path(setting) if setting else REPO_ROOT / DEFAULT_RESULTS_DIRNAME
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load(name: str, *, where: Path | None = None) -> pl.DataFrame:
    """Read one results file with its schema dtypes.

    *name* must be a key of :data:`anomaly_decay.schemas.RESULTS_SCHEMAS` (e.g.
    ``"main.csv"``). The file is read through
    :func:`anomaly_decay.io.read_results`, the same schema-typed reader every other stage
    uses. Missing files raise :class:`FileNotFoundError`.
    """
    if name not in RESULTS_SCHEMAS:
        raise ValueError(f"unknown results file: {name!r}")
    base = where if where is not None else results_dir()
    return results_io.read_results(name, base)


def load_optional(name: str, *, where: Path | None = None) -> pl.DataFrame | None:
    """Like :func:`load`, but return ``None`` when the file does not exist yet.

    The report renders a visible pending note in that case instead of failing the
    render when a stage has not produced its results file.
    """
    try:
        return load(name, where=where)
    except FileNotFoundError:
        return None


def lookup(df: pl.DataFrame, column: str, **conditions: object) -> object:
    """Value of *column* in the first row matching every keyword *condition*.

    Returns ``None`` when nothing matches, so inline report expressions can hand the
    result to :func:`num`/:func:`pct`/:func:`tstat` and render ``n/a``. Unknown column
    names raise :class:`ValueError` (a bug in the report, not missing data).
    """
    if column not in df.columns:
        raise ValueError(f"column not in frame: {column!r}; frame has {df.columns}")
    expr: pl.Expr = pl.lit(True)
    for name, value in conditions.items():
        if name not in df.columns:
            raise ValueError(f"condition column not in frame: {name!r}; frame has {df.columns}")
        expr = expr & (pl.col(name) == value)
    matches = df.filter(expr).select(column)
    if matches.height == 0:
        return None
    cell: object = matches[column].item(0)
    return cell


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def num(value: float | int | None, digits: int = 2) -> str:
    """Format *value* with *digits* decimals; ``None`` and ``NaN`` render ``n/a``."""
    if value is None or math.isnan(value):
        return NA_TEXT
    return f"{value:.{digits}f}"


def pct(value: float | int | None, digits: int = 1) -> str:
    """Format a value already in percent units (e.g. ``26.0`` → ``26.0%``)."""
    if value is None or math.isnan(value):
        return NA_TEXT
    return f"{value:.{digits}f}%"


def count(value: float | int | None) -> str:
    """Format a count with thousands separators (``101567`` → ``101,567``)."""
    if value is None or math.isnan(value):
        return NA_TEXT
    return f"{int(value):,}"


def tstat(value: float | None, digits: int = 2) -> str:
    """Format a t-statistic with *digits* decimals; ``None`` and ``NaN`` render ``n/a``."""
    if value is None or math.isnan(value):
        return NA_TEXT
    return f"{value:.{digits}f}"


def direction(
    value: float | int | None,
    *,
    positive: str = "higher",
    negative: str = "lower",
    zero: str = "unchanged",
) -> str:
    """Sign word for *value*: *positive* above zero, *negative* below, *zero* at zero.

    Prose whose wording depends on the sign of a computed number (for example "returns
    are *lower* after publication") gets that word here, so the text never hard-codes
    something the data could contradict. ``None`` and ``NaN`` render :data:`NA_TEXT`.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA_TEXT
    if value > 0:
        return positive
    if value < 0:
        return negative
    return zero


def magnitude(value: float | int | None) -> float | int | None:
    """Absolute value of *value*, passing ``None`` and ``NaN`` through.

    Paired with :func:`direction` so a sentence can say "lower by *x*" without writing
    ``abs(...)`` inside an inline expression that might receive a missing value.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return value
    return abs(value)


def ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """``numerator / denominator`` as a plain ratio.

    ``None`` when either input is missing or ``NaN``, or when *denominator* is zero, so
    inline expressions can pass the result to a formatter without crashing a render.
    """
    if numerator is None or denominator is None:
        return None
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return None
    return numerator / denominator


def ratio_pct(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """``100 * numerator / denominator`` as a percent value.

    ``None`` when either input is missing or ``NaN``, or when *denominator* is zero, so
    the result can be passed straight to :func:`pct` from an inline expression.
    """
    value = ratio(numerator, denominator)
    return None if value is None else 100.0 * value


def comparison_word(left: float | int | None, right: float | int | None) -> str:
    """Word for where *left* sits relative to *right*: above, below or equal to it.

    Comparison prose ("the replication's estimate sits *below* the paper's") gets its
    direction here, so the wording follows the numbers instead of being asserted.
    """
    if left is None or right is None:
        return NA_TEXT
    if math.isnan(left) or math.isnan(right):
        return NA_TEXT
    if left > right:
        return "above"
    if left < right:
        return "below"
    return "equal to"


def pp_gap(left: float | int | None, right: float | int | None, digits: int = 1) -> str:
    """Magnitude of ``left - right`` in percentage points, e.g. ``'4.2 pp'``.

    Used for gaps between two percent values (decay estimates); the direction comes from
    :func:`comparison_word` so the sentence never hard-codes which value is larger.
    """
    if left is None or right is None:
        return NA_TEXT
    if math.isnan(left) or math.isnan(right):
        return NA_TEXT
    return f"{abs(left - right):.{digits}f} pp"


def factor(value: float | int | None, base: float | int | None, digits: int = 2) -> str:
    """``|value / base|`` as e.g. ``'1.48×'`` — a magnitude, never a signed ratio."""
    if base is None:
        return NA_TEXT
    quotient = ratio(value, base)
    if quotient is None:
        return NA_TEXT
    return f"{abs(quotient):.{digits}f}×"


def month_label(value: object) -> str:
    """Format a ``yyyymm`` key as ``July 1926``.

    Used for month coverage read back from ``coverage.csv`` and for the README block;
    anything that is not a six-digit key with a month between 1 and 12 renders
    :data:`NA_TEXT` rather than raising inside a render.
    """
    if value is None:
        return NA_TEXT
    text = str(value)
    if len(text) != 6 or not text.isdigit():
        return NA_TEXT
    month = int(text[4:])
    if not 1 <= month <= 12:
        return NA_TEXT
    return f"{MONTH_NAMES[month - 1]} {int(text[:4])}"


def term_label(term: object) -> str:
    """Prose label for a result term (``post_sample`` → ``Post-sample``)."""
    if term is None:
        return NA_TEXT
    return TERM_LABELS.get(str(term), str(term))


def coverage_item_label(item: object) -> str:
    """Plain-language label for a ``coverage.csv`` item, raw name for unknown items."""
    if item is None:
        return NA_TEXT
    name = str(item)
    return COVERAGE_LABELS.get(name, name.replace("_", " "))


def coverage_value_text(item: object, value: object) -> str:
    """Display text for a coverage value: months spelled out, counts with separators."""
    if _is_missing(value):
        return NA_TEXT
    if item in ("first_month", "last_month"):
        return month_label(value)
    name = str(item)
    text = str(value)
    if name.startswith("n_") and text.isdigit():
        return f"{int(text):,}"
    return _format_cell(value)


def coverage_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Coverage rows as the report shows them: prose labels, month names, fixed order.

    Returns a two-column frame (``item``, ``value``) of display strings, ordered by
    :data:`COVERAGE_ORDER` so the table reads release, coverage, then predictor counts
    instead of the alphabetical order of the results file.
    """
    ordered_frame = ordered(frame, {"item": list(COVERAGE_ORDER)})
    items = ordered_frame["item"].to_list()
    values = ordered_frame["value"].to_list()
    return pl.DataFrame(
        {
            "item": [coverage_item_label(item) for item in items],
            "value": [
                coverage_value_text(item, value) for item, value in zip(items, values, strict=True)
            ],
        }
    )


def decline_word(
    left: float | int | None,
    right: float | int | None,
) -> str:
    """How large *left*'s decline is relative to *right*'s: larger, smaller, or the same.

    Both arguments are signed percent changes (negative = a decline), so a more negative
    value is a *larger* decline — the comparison is reversed relative to
    :func:`comparison_word`. Used for prose about two negative decay estimates.
    """
    if left is None or right is None:
        return NA_TEXT
    if math.isnan(left) or math.isnan(right):
        return NA_TEXT
    if left < right:
        return "larger"
    if left > right:
        return "smaller"
    return "the same"


def exploratory_label(value: object) -> str:
    """Words for a pre-registration flag: confirmatory (pre-registered) or exploratory."""
    if value is None:
        return NA_TEXT
    return "exploratory" if bool(value) else "confirmatory (pre-registered)"


def missing_tstat_label(value: object) -> str:
    """Words for the pre-registered missing-t rule (no raw enum names in the report)."""
    if value is None:
        return NA_TEXT
    return MISSING_TSTAT_LABELS.get(str(value), str(value))


def block_method_label(value: object) -> str:
    """Words for the pre-registered bootstrap block method."""
    if value is None:
        return NA_TEXT
    return BLOCK_METHOD_LABELS.get(str(value), str(value))


def ordered(frame: pl.DataFrame, order: Mapping[str, Sequence[object]]) -> pl.DataFrame:
    """Stable-sort *frame* into the given value order for each column in *order*.

    Each mapping key is a column and each value the preferred order of its values. Rows
    whose value is not listed keep their relative order after the listed ones, and ties
    within a rank keep their original order, so a frame ordered by specification first
    and term second reads main-first and post-sample-first without dropping unknown rows.
    """
    result = frame
    ranks: list[str] = []
    for column, values in order.items():
        if column not in result.columns:
            raise ValueError(f"order column not in frame: {column!r}; frame has {result.columns}")
        rank = f"__rank_{column}"
        mapping = {value: index for index, value in enumerate(values)}
        result = result.with_columns(
            pl.col(column).replace_strict(mapping, default=len(values)).alias(rank)
        )
        ranks.append(rank)
    if not ranks:
        return result
    return result.sort(ranks, maintain_order=True).drop(ranks)


def specifications(path: Path | None = None) -> list[Spec]:
    """Load ``config/specifications.yaml`` in pre-registered file order.

    The report takes the specification display order from here, so a table can never
    drift into the alphabetical CSV order or a hand-written list.
    """
    from anomaly_decay.config import load_specifications

    source = SPECIFICATIONS_PATH if path is None else path
    return load_specifications(source)


def _format_cell(value: object) -> str:
    if _is_missing(value):
        return NA_TEXT
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return num(value)
    if isinstance(value, int):
        return str(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(
    df: pl.DataFrame,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    formats: Mapping[str, Callable[[object], str]] | None = None,
) -> str:
    """Render selected *columns* of *df* as a Markdown pipe table.

    *headers* defaults to the column names and must match *columns* in length. *formats*
    maps column names to per-cell formatters; unlisted columns use the defaults
    (``n/a`` for null/NaN, plain decimals for floats, ``yes``/``no`` for booleans).
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}; frame has {df.columns}")
    if headers is None:
        headers = list(columns)
    if len(headers) != len(columns):
        raise ValueError(
            f"headers ({len(headers)}) and columns ({len(columns)}) must have equal length"
        )
    formatters = formats if formats is not None else {}
    view = df.select(columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in view.iter_rows():
        cells = []
        for column, value in zip(columns, row, strict=True):
            formatter = formatters.get(column)
            cells.append(formatter(value) if formatter is not None else _format_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def grouped_table(
    df: pl.DataFrame,
    group: str,
    columns: Sequence[str],
    headers: Sequence[str] | None = None,
    formats: Mapping[str, Callable[[object], str]] | None = None,
    labels: Mapping[object, str] | None = None,
) -> str:
    """Render *df* as a pipe table that prints each long *group* label once.

    *group* must be the first of *columns*. The label is printed in bold in the first
    column of the group's first row and left blank on the remaining rows, so a long
    specification label does not repeat (and does not push the table past the page
    width). *labels* maps group values to prose labels; unlisted values render as they
    are. Order the frame with :func:`ordered` first.
    """
    if not columns:
        raise ValueError("at least one column is required")
    if columns[0] != group:
        raise ValueError(f"group column {group!r} must be the first column")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}; frame has {df.columns}")
    if headers is None:
        headers = list(columns)
    if len(headers) != len(columns):
        raise ValueError(
            f"headers ({len(headers)}) and columns ({len(columns)}) must have equal length"
        )
    formatters = formats if formats is not None else {}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    current: object = object()
    for row in df.select(columns).iter_rows():
        values = dict(zip(columns, row, strict=True))
        key = values[group]
        cells: list[str] = []
        if key != current:
            current = key
            label = labels.get(key, str(key)) if labels is not None else str(key)
            cells.append(f"**{_format_cell(label)}**")
        else:
            cells.append("")
        for column in columns[1:]:
            formatter = formatters.get(column)
            value = values[column]
            cells.append(formatter(value) if formatter is not None else _format_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(name: str, *, where: Path | None = None) -> Path | None:
    """Path of a generated figure under ``<results>/figures``, or ``None`` if absent.

    Raster files are preferred because :func:`figure_block` can embed them in both HTML
    and Typst PDF; ``.svg`` is the fallback.
    """
    base = where if where is not None else results_dir()
    for extension in FIGURE_EXTENSIONS:
        candidate = base / "figures" / f"{name}.{extension}"
        if candidate.is_file():
            return candidate
    return None


def figure_block(name: str, caption: str, *, where: Path | None = None) -> Image | Markdown:
    """Displayable figure block: the image when it exists, else a visible note.

    Raster images are embedded with :class:`IPython.display.Image`, which renders in HTML
    and Typst PDF without depending on the output directory's relative layout; an
    SVG-only file is referenced as Markdown. A missing figure is a warning callout, so the
    report never fails to render because a figure stage has not run.
    """
    from IPython.display import Image as IpyImage
    from IPython.display import Markdown as IpyMarkdown

    markdown = cast("Callable[[str], Markdown]", IpyMarkdown)
    image = cast("Callable[..., Image]", IpyImage)

    path = figure(name, where=where)
    if path is None:
        return markdown(
            "::: {.callout-warning}\n"
            f"Figure not generated: `{name}` (expected under `figures/` in the results "
            "directory).\n"
            ":::\n"
        )
    if path.suffix.lstrip(".").lower() in RASTER_EXTENSIONS:
        return image(filename=str(path), alt=caption)
    return markdown(f"![{caption}]({path.as_posix()})\n")


def original_paper(path: Path | None = None) -> dict[str, Any]:
    """Load ``config/original_paper.yaml`` (citation and verified paper values)."""
    source = ORIGINAL_PAPER_PATH if path is None else path
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top level must be a mapping")
    return data


def data_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``config/data.toml`` (pinned OSAP vintage, Ken French download date)."""
    source = DATA_CONFIG_PATH if path is None else path
    with source.open("rb") as handle:
        return tomllib.load(handle)


def phase2_preregistration(path: Path | None = None) -> dict[str, Any] | None:
    """Load ``config/phase2_preregistration.yaml``, or ``None`` when it does not exist.

    The Phase 2 sections describe themselves from the pre-registration (reps, seed, block
    length, hurdle, vintage preference); returning ``None`` keeps the report renderable
    before the file is committed, with the sections showing their pending notes.
    """
    source = PHASE2_PREREG_PATH if path is None else path
    if not source.is_file():
        return None
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top level must be a mapping")
    return data
