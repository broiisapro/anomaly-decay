"""Window classification (pure functions, no I/O).

Interface stub: implemented in Epic 01 workstream A. Window rules (architecture §4.6):

- ``in_sample``: January ``SampleStartYear`` through December ``SampleEndYear``,
  inclusive;
- ``post_publication``: from January ``Year+1`` (``default``) or January ``Year``
  (``alt``);
- ``post_sample``: January ``SampleEndYear+1`` up to the month before
  post-publication (may be empty);
- ``pre_sample``: before January ``SampleStartYear`` (dropped when the panel is built).
- If the publication start would fall inside the in-sample period, ``in_sample`` wins
  and ``post_sample`` is empty.
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


def post_publication_start(pub_year: int, convention: Convention = "default") -> int:
    """First post-publication month as ``yyyymm``.

    Input: *pub_year* (calendar year of publication), *convention* (``default``:
    January of ``Year+1``; ``alt``: January of ``Year``). Output: ``yyyymm`` Int.
    No units apply.
    """
    raise NotImplementedError


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
    ``in_sample``, ``post_sample``, ``post_publication`` per the module docstring.
    No units apply (a month is a month).
    """
    raise NotImplementedError


def classify_frame(df: pl.DataFrame, convention: Convention = "default") -> pl.Series:
    """Classify every row of *df*.

    Input: *df* with columns ``yyyymm``, ``sample_start_year``, ``sample_end_year``,
    ``pub_year``. Output: Utf8 Series named ``window`` with one value per row.
    """
    raise NotImplementedError
