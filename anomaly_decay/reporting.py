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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import yaml

from anomaly_decay.schemas import RESULTS_SCHEMAS

if TYPE_CHECKING:
    from IPython.display import Image, Markdown

__all__ = [
    "NA_TEXT",
    "figure",
    "figure_block",
    "load",
    "load_optional",
    "lookup",
    "num",
    "original_paper",
    "pct",
    "results_dir",
    "table",
    "tstat",
]

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_ENV_VAR: str = "ANOMALY_DECAY_RESULTS"
DEFAULT_RESULTS_DIRNAME: str = "results"
ORIGINAL_PAPER_PATH: Path = REPO_ROOT / "config" / "original_paper.yaml"
RASTER_EXTENSIONS: tuple[str, ...] = ("png", "jpg", "jpeg", "gif")
FIGURE_EXTENSIONS: tuple[str, ...] = (*RASTER_EXTENSIONS, "svg")
NA_TEXT: str = "n/a"


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


def _io_read_results() -> Callable[[str, Path], pl.DataFrame] | None:
    """``anomaly_decay.io.read_results`` when that module provides it, else ``None``."""
    from anomaly_decay import io

    reader = getattr(io, "read_results", None)
    if callable(reader):
        return cast("Callable[[str, Path], pl.DataFrame]", reader)
    return None


def load(name: str, *, where: Path | None = None) -> pl.DataFrame:
    """Read one results file with its schema dtypes.

    *name* must be a key of :data:`anomaly_decay.schemas.RESULTS_SCHEMAS` (e.g.
    ``"main.csv"``). Uses ``anomaly_decay.io.read_results`` when available and otherwise
    a schema-typed :func:`polars.read_csv` (until issue #21 lands). Missing files raise
    :class:`FileNotFoundError`.
    """
    if name not in RESULTS_SCHEMAS:
        raise ValueError(f"unknown results file: {name!r}")
    base = where if where is not None else results_dir()
    reader = _io_read_results()
    if reader is not None:
        return reader(name, base)
    return pl.read_csv(base / name, schema=RESULTS_SCHEMAS[name])


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


def tstat(value: float | None, digits: int = 2) -> str:
    """Format a t-statistic with *digits* decimals; ``None`` and ``NaN`` render ``n/a``."""
    if value is None or math.isnan(value):
        return NA_TEXT
    return f"{value:.{digits}f}"


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
    """Load ``config/original_paper.yaml`` (citation and pending-verification values)."""
    source = ORIGINAL_PAPER_PATH if path is None else path
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{source}: top level must be a mapping")
    return data
