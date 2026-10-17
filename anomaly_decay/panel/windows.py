"""Window classification (pure functions, no I/O).

Window rules per docs/architecture.md §4.6:

- ``in_sample``: January ``SampleStartYear`` through December ``SampleEndYear``,
  inclusive;
- ``post_publication``: from January ``Year+1`` (``default``) or January ``Year``
  (``alt``);
- ``post_sample``: January ``SampleEndYear+1`` up to the month before
  post-publication (may be empty);
- ``pre_sample``: before January ``SampleStartYear`` (dropped when the panel is built).
- If the publication start would fall inside the in-sample period, ``in_sample`` wins
  and ``post_sample`` is empty.

Every comparison is on integer ``yyyymm`` keys, never on calendar years, so January and
December of the boundary years are labelled correctly. Window labels carry no units;
returns (percent per month) are joined elsewhere.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

Window = Literal["pre_sample", "in_sample", "post_sample", "post_publication"]
Convention = Literal["default", "alt"]

__all__ = [
    "Convention",
    "Window",
    "classify_frame",
    "classify_month",
    "post_publication_start",
]


def _check_convention(convention: str) -> None:
    if convention not in ("default", "alt"):
        raise ValueError(f"unknown convention {convention!r}; expected 'default' or 'alt'")


def post_publication_start(pub_year: int, convention: Convention = "default") -> int:
    """First post-publication month as a ``yyyymm`` integer.

    *pub_year* is the calendar year of publication. *convention* is ``default``
    (January of ``Year+1``) or ``alt`` (January of ``Year``). For example,
    ``post_publication_start(1993) == 199401`` and
    ``post_publication_start(1993, "alt") == 199301``. No units apply.
    """
    _check_convention(convention)
    year = pub_year + 1 if convention == "default" else pub_year
    return year * 100 + 1


def classify_month(
    yyyymm: int,
    sample_start_year: int,
    sample_end_year: int,
    pub_year: int,
    convention: Convention = "default",
) -> Window:
    """Classify one month for one predictor.

    Inputs: *yyyymm* (integer ``YYYYMM``), the predictor's sample start/end years and
    publication year, and the publication convention. Output: one of ``pre_sample``,
    ``in_sample``, ``post_sample`` or ``post_publication`` per the module docstring.
    Every comparison is on ``yyyymm`` keys, so boundary months are classified on their
    month, not their year. No units apply (a month is a month).
    """
    _check_convention(convention)
    first_ym = sample_start_year * 100 + 1
    last_ym = sample_end_year * 100 + 12
    publication_first_ym = post_publication_start(pub_year, convention)
    if yyyymm < first_ym:
        return "pre_sample"
    if yyyymm <= last_ym:
        return "in_sample"
    if yyyymm >= publication_first_ym:
        return "post_publication"
    return "post_sample"


def classify_frame(df: pl.DataFrame, convention: Convention = "default") -> pl.Series:
    """Classify every row of *df* with Polars expressions.

    Input: *df* with columns ``yyyymm``, ``sample_start_year``, ``sample_end_year`` and
    ``pub_year``. Output: a Utf8 Series named ``window`` with one label per row. The
    years are widened to Int32 before the ``yyyymm`` arithmetic, so no intermediate
    value can overflow. Equivalent to :func:`classify_month` applied row by row.
    """
    _check_convention(convention)
    year_offset = 1 if convention == "default" else 0
    first_ym = pl.col("sample_start_year").cast(pl.Int32) * 100 + 1
    last_ym = pl.col("sample_end_year").cast(pl.Int32) * 100 + 12
    publication_first_ym = (pl.col("pub_year").cast(pl.Int32) + year_offset) * 100 + 1
    labels = (
        pl.when(pl.col("yyyymm") < first_ym)
        .then(pl.lit("pre_sample"))
        .when(pl.col("yyyymm") <= last_ym)
        .then(pl.lit("in_sample"))
        .when(pl.col("yyyymm") >= publication_first_ym)
        .then(pl.lit("post_publication"))
        .otherwise(pl.lit("post_sample"))
    )
    return df.select(labels.alias("window")).to_series()
