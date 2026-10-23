"""Tests for the per-predictor information and event-year profiles (decision O45)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate.predictor_profiles import (
    EVENT_YEAR_HI,
    EVENT_YEAR_LO,
    main,
    predictor_event_time,
    predictor_info,
)
from anomaly_decay.io import read_results
from anomaly_decay.schemas import PREDICTORS_SCHEMA, RESULTS_SCHEMAS


def _metadata() -> pl.DataFrame:
    """A small metadata table in schema order, deliberately not sorted by signalname."""
    return pl.DataFrame(
        {
            "signalname": ["Size", "AM", "Beta"],
            "sample_start_year": [1926, 1968, 1963],
            "sample_end_year": [1975, 1990, 1993],
            "pub_year": [1981, 1992, 1998],
            "journal": ["JFE", "JF", "JF"],
            "authors": ["Banz", "Amihud", "Fama"],
            "cat_economic": ["size", "liquidity", "risk"],
            "cat_data": ["Price", "Price", "Price"],
            "clear": [True, False, True],
            "date_source": ["signaldoc", "signaldoc", "supplement:x"],
        },
        schema=PREDICTORS_SCHEMA,
    )


def _panel(
    signalname: list[str],
    portfolio: list[str],
    ret: list[float | None],
    sample_end: list[int | None],
    publication: list[int | None],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": signalname,
            "portfolio": portfolio,
            "ret": pl.Series(ret, dtype=pl.Float64),
            "event_time_sample_end": pl.Series(sample_end, dtype=pl.Int32),
            "event_time_pub": pl.Series(publication, dtype=pl.Int32),
        }
    )


def _two_predictor_panel() -> pl.DataFrame:
    return _panel(
        signalname=["A"] * 11 + ["B"] * 2,
        portfolio=["op"] * 13,
        ret=[100.0, 2.0, 4.0, -1.0, 1.0, 0.5, 9.0, 100.0, 100.0, 0.25, 0.75, None, 2.0],
        sample_end=[-13, -12, -1, 0, 11, 12, 119, 120, None, None, None, 0, None],
        publication=[None] * 9 + [-1, 0, None, 5],
    )


def test_predictor_info_casts_years_and_sorts_by_signalname() -> None:
    info = predictor_info(_metadata())

    assert info.columns == list(RESULTS_SCHEMAS["predictor_info.csv"])
    assert info.schema == RESULTS_SCHEMAS["predictor_info.csv"]
    assert info["signalname"].to_list() == ["AM", "Beta", "Size"]
    assert info["pub_year"].to_list() == [1992, 1998, 1981]
    assert info["sample_start_year"].to_list() == [1968, 1963, 1926]
    assert info["sample_end_year"].to_list() == [1990, 1993, 1975]
    assert info["clear"].to_list() == [False, True, True]
    assert info["authors"].to_list() == ["Amihud", "Fama", "Banz"]
    assert info["journal"].to_list() == ["JF", "JF", "JFE"]
    assert info["cat_economic"].to_list() == ["liquidity", "risk", "size"]


def test_predictor_info_rejects_a_frame_that_is_not_metadata() -> None:
    with pytest.raises(ValueError, match="columns differ"):
        predictor_info(_metadata().drop("clear"))


@pytest.mark.parametrize(
    ("event_month", "expected_year"),
    [
        (-60, -5),
        (-13, -2),
        (-12, -1),
        (-1, -1),
        (0, 0),
        (11, 0),
        (12, 1),
        (119, 9),
    ],
)
def test_event_year_uses_floor_division_for_negative_months(
    event_month: int, expected_year: int
) -> None:
    panel = _panel(
        signalname=["A"],
        portfolio=["op"],
        ret=[1.0],
        sample_end=[event_month],
        publication=[None],
    )
    profile = predictor_event_time(panel)
    assert profile.height == 1
    assert profile["anchor"].to_list() == ["sample_end"]
    assert profile["event_year"].to_list() == [expected_year]


def test_event_year_bounds_cover_the_documented_window() -> None:
    assert (EVENT_YEAR_LO, EVENT_YEAR_HI) == (-5, 9)


def test_event_time_aggregates_months_into_years_per_anchor() -> None:
    profile = predictor_event_time(_two_predictor_panel())

    assert profile.columns == list(RESULTS_SCHEMAS["predictor_event_time.csv"])
    assert profile.schema == RESULTS_SCHEMAS["predictor_event_time.csv"]

    rows = profile.iter_rows(named=True)
    by_key = {(row["signalname"], row["anchor"], row["event_year"]): row for row in rows}

    # months -12 and -1 both fall in year -1; -13 is inside the window and lands in
    # year -2, so its deliberately odd 100.0 return must appear there on its own.
    assert by_key[("A", "sample_end", -2)]["mean_ret"] == 100.0
    assert by_key[("A", "sample_end", -2)]["n_months"] == 1
    assert by_key[("A", "sample_end", -1)]["mean_ret"] == 3.0
    assert by_key[("A", "sample_end", -1)]["n_months"] == 2
    assert by_key[("A", "sample_end", 0)]["mean_ret"] == 0.0
    assert by_key[("A", "sample_end", 0)]["n_months"] == 2
    assert by_key[("A", "sample_end", 1)]["n_months"] == 1
    assert by_key[("A", "sample_end", 9)]["mean_ret"] == 9.0

    # months 120 and null are outside the window and contribute no rows
    assert ("A", "sample_end", 10) not in by_key
    assert max(key[2] for key in by_key if key[:2] == ("A", "sample_end")) == 9

    assert by_key[("A", "publication", -1)]["mean_ret"] == 0.25
    assert by_key[("A", "publication", 0)]["mean_ret"] == 0.75
    assert by_key[("B", "publication", 0)]["mean_ret"] == 2.0
    assert ("B", "sample_end", 0) not in by_key  # ret is null, so no usable month


def test_event_time_ignores_non_op_portfolios() -> None:
    panel = _panel(
        signalname=["A"],
        portfolio=["deciles_ew"],
        ret=[999.0],
        sample_end=[0],
        publication=[0],
    )
    assert predictor_event_time(panel).height == 0


def test_event_time_without_anchor_columns_is_empty_and_typed() -> None:
    panel = pl.DataFrame(
        {
            "signalname": ["A"],
            "portfolio": ["op"],
            "ret": [1.0],
        }
    )
    profile = predictor_event_time(panel)
    assert profile.height == 0
    assert profile.schema == RESULTS_SCHEMAS["predictor_event_time.csv"]


def test_event_time_output_is_sorted_by_its_key() -> None:
    profile = predictor_event_time(_two_predictor_panel())
    keys = profile.select("signalname", "anchor", "event_year")
    assert keys.equals(keys.sort(["signalname", "anchor", "event_year"]))


def test_cli_writes_both_results_files_and_is_byte_identical(tmp_path: Path) -> None:
    metadata = tmp_path / "predictors.parquet"
    panel = tmp_path / "panel.parquet"
    _metadata().write_parquet(metadata)
    _two_predictor_panel().write_parquet(panel)

    first = tmp_path / "first"
    second = tmp_path / "second"
    for out in (first, second):
        assert (
            main(["--panel", str(panel), "--metadata", str(metadata), "--results-dir", str(out)])
            == 0
        )

    for name, frame in (
        ("predictor_info.csv", predictor_info(_metadata())),
        ("predictor_event_time.csv", predictor_event_time(_two_predictor_panel())),
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert read_results(name, first).equals(frame)

    assert sorted(path.name for path in first.iterdir()) == [
        "predictor_event_time.csv",
        "predictor_info.csv",
    ]
