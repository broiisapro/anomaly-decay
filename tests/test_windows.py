"""Tests for window classification (docs/architecture.md §4.6)."""

from __future__ import annotations

import random

import polars as pl
import pytest
from anomaly_decay.panel.windows import (
    classify_frame,
    classify_month,
    post_publication_start,
)
from hypothesis import example, given, settings
from hypothesis import strategies as st

WINDOWS: tuple[str, ...] = ("pre_sample", "in_sample", "post_sample", "post_publication")
WINDOW_ORDER: dict[str, int] = {window: rank for rank, window in enumerate(WINDOWS)}

# (sample_start_year, sample_end_year, pub_year) from the committed fixture signal doc.
NAMED_PREDICTORS: dict[str, tuple[int, int, int]] = {
    "Mom12m": (1964, 1989, 1993),
    "GP": (1963, 2010, 2013),
    "BetaFP": (1929, 2012, 2014),
}


def _months(first: int, last: int) -> list[int]:
    out: list[int] = []
    yyyymm = first
    while yyyymm <= last:
        out.append(yyyymm)
        year, month = divmod(yyyymm, 100)
        yyyymm = (year + 1) * 100 + 1 if month == 12 else yyyymm + 1
    return out


def test_post_publication_start_default_is_january_of_year_plus_one() -> None:
    assert post_publication_start(1993) == 199401
    assert post_publication_start(2013) == 201401


def test_post_publication_start_alt_is_january_of_year() -> None:
    assert post_publication_start(1993, "alt") == 199301
    assert post_publication_start(2013, "alt") == 201301


def test_unknown_convention_is_rejected() -> None:
    with pytest.raises(ValueError, match="convention"):
        post_publication_start(2000, "bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="convention"):
        classify_month(200001, 1990, 2000, 2001, "bogus")  # type: ignore[arg-type]
    frame = pl.DataFrame(
        {
            "yyyymm": pl.Series([200001], dtype=pl.Int32),
            "sample_start_year": pl.Series([1990], dtype=pl.Int16),
            "sample_end_year": pl.Series([2000], dtype=pl.Int16),
            "pub_year": pl.Series([2001], dtype=pl.Int16),
        }
    )
    with pytest.raises(ValueError, match="convention"):
        classify_frame(frame, "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("start", "end", "pub"),
    NAMED_PREDICTORS.values(),
    ids=NAMED_PREDICTORS.keys(),
)
def test_sample_end_year_is_in_sample_and_next_january_is_post_sample(
    start: int, end: int, pub: int
) -> None:
    # December of the sample end year is the last in-sample month ...
    assert classify_month(end * 100 + 12, start, end, pub) == "in_sample"
    # ... and the very next month is the first post-sample month.
    assert classify_month((end + 1) * 100 + 1, start, end, pub) == "post_sample"
    # Sample boundaries around the start year.
    assert classify_month((start - 1) * 100 + 12, start, end, pub) == "pre_sample"
    assert classify_month(start * 100 + 1, start, end, pub) == "in_sample"
    # The post-sample stretch ends right before January of Year+1.
    assert classify_month(pub * 100 + 12, start, end, pub) == "post_sample"
    assert classify_month((pub + 1) * 100 + 1, start, end, pub) == "post_publication"


@pytest.mark.parametrize(
    ("start", "end", "pub"),
    NAMED_PREDICTORS.values(),
    ids=NAMED_PREDICTORS.keys(),
)
def test_alt_convention_boundaries(start: int, end: int, pub: int) -> None:
    december_of_pub_minus_one = (pub - 1) * 100 + 12
    january_of_pub = pub * 100 + 1
    # Under the alt convention post-publication starts in January of Year itself ...
    assert classify_month(december_of_pub_minus_one, start, end, pub, "alt") == "post_sample"
    assert classify_month(january_of_pub, start, end, pub, "alt") == "post_publication"
    # ... while the default convention still calls both months post-sample.
    assert classify_month(december_of_pub_minus_one, start, end, pub, "default") == "post_sample"
    assert classify_month(january_of_pub, start, end, pub, "default") == "post_sample"
    # First month of Year+1 is post-publication under both conventions.
    assert classify_month((pub + 1) * 100 + 1, start, end, pub, "alt") == "post_publication"
    assert classify_month((pub + 1) * 100 + 1, start, end, pub, "default") == "post_publication"


def test_post_sample_is_empty_when_publication_equals_sample_end_year() -> None:
    start, end, pub = 1990, 2000, 2000
    assert classify_month(end * 100 + 12, start, end, pub) == "in_sample"
    for yyyymm in _months((end + 1) * 100 + 1, (end + 3) * 100 + 12):
        assert classify_month(yyyymm, start, end, pub) == "post_publication"
    labels = {
        classify_month(yyyymm, start, end, pub)
        for yyyymm in _months((start - 1) * 100 + 1, (end + 3) * 100 + 12)
    }
    assert "post_sample" not in labels


@pytest.mark.parametrize("pub", [1995, 2000])
def test_post_sample_is_empty_when_publication_is_inside_sample_under_alt(pub: int) -> None:
    start, end = 1990, 2000
    # In-sample wins: even months after the publication month stay in_sample.
    for yyyymm in _months(pub * 100 + 1, end * 100 + 12):
        assert classify_month(yyyymm, start, end, pub, "alt") == "in_sample"
    for yyyymm in _months((end + 1) * 100 + 1, (end + 3) * 100 + 12):
        assert classify_month(yyyymm, start, end, pub, "alt") == "post_publication"
    labels = {
        classify_month(yyyymm, start, end, pub, "alt")
        for yyyymm in _months((start - 1) * 100 + 1, (end + 3) * 100 + 12)
    }
    assert "post_sample" not in labels


@settings(max_examples=150, deadline=None)
@given(
    a=st.integers(min_value=1900, max_value=2030),
    b=st.integers(min_value=1900, max_value=2030),
    c=st.integers(min_value=1900, max_value=2030),
    month_year=st.integers(min_value=1899, max_value=2032),
    month_number=st.integers(min_value=1, max_value=12),
)
@example(a=2000, b=2000, c=2000, month_year=1999, month_number=12)  # start == end == pub
@example(a=1990, b=2000, c=2001, month_year=2000, month_number=12)  # end + 1 == pub
@example(a=1990, b=2000, c=2000, month_year=2001, month_number=1)  # pub == end
@example(a=1990, b=2000, c=1995, month_year=1995, month_number=1)  # pub inside sample
def test_property_each_month_in_exactly_one_window(
    a: int, b: int, c: int, month_year: int, month_number: int
) -> None:
    start, end, pub = sorted((a, b, c))
    month = month_year * 100 + month_number

    label = classify_month(month, start, end, pub)
    assert label in WINDOWS
    assert sum(label == window for window in WINDOWS) == 1

    months = _months((start - 1) * 100 + 1, (pub + 1) * 100 + 12)
    default_labels = [classify_month(m, start, end, pub) for m in months]
    ranks = [WINDOW_ORDER[item] for item in default_labels]
    assert ranks == sorted(ranks), f"labels not ordered in time: {default_labels}"

    alt_labels = [classify_month(m, start, end, pub, "alt") for m in months]
    default_publication = {
        m for m, item in zip(months, default_labels, strict=True) if item == "post_publication"
    }
    alt_publication = {
        m for m, item in zip(months, alt_labels, strict=True) if item == "post_publication"
    }
    assert default_publication <= alt_publication
    assert min(alt_publication) <= min(default_publication)


def test_classify_frame_matches_classify_month_on_10000_random_rows() -> None:
    rng = random.Random(20261017)
    rows: list[tuple[int, int, int, int]] = []
    for _ in range(10_000):
        start = rng.randint(1900, 2030)
        end = rng.randint(start, 2030)
        pub = rng.randint(end, 2035)
        yyyymm = rng.randint(1899, 2036) * 100 + rng.randint(1, 12)
        rows.append((yyyymm, start, end, pub))

    frame = pl.DataFrame(
        {
            "yyyymm": pl.Series([row[0] for row in rows], dtype=pl.Int32),
            "sample_start_year": pl.Series([row[1] for row in rows], dtype=pl.Int16),
            "sample_end_year": pl.Series([row[2] for row in rows], dtype=pl.Int16),
            "pub_year": pl.Series([row[3] for row in rows], dtype=pl.Int16),
        }
    )
    for convention in ("default", "alt"):
        result = classify_frame(frame, convention)
        assert result.name == "window"
        assert result.dtype == pl.Utf8
        expected = [
            classify_month(yyyymm, start, end, pub, convention) for yyyymm, start, end, pub in rows
        ]
        assert result.to_list() == expected
