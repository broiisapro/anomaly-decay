"""Tests for the multiple-testing lens (issue #67).

Everything here runs on hand-built panels and signal docs: the parse rules for
the signal doc's text ``T-Stat`` column, the weak-flag boundary and sign rules,
the replicated t-statistic formula, and the summary counts. The regression
specifications (issue #68) are covered further down in the module.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate.multiple_testing import (
    DEFAULT_HURDLE_T,
    TSTAT_SUMMARY_SCHEMA,
    TSTATS_SCHEMA,
    TstatError,
    tstat_summary,
    tstats,
)

from test_estimate import make_panel


def write_signal_doc(
    tmp_path: Path, rows: list[tuple[str, str | None]], *, columns: list[str] | None = None
) -> Path:
    """Write a small signal doc CSV with the given ``(Acronym, T-Stat)`` rows."""
    path = tmp_path / "signal_doc.csv"
    header = columns if columns is not None else ["Acronym", "T-Stat"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    return path


def one_row_panel(*signalnames: str) -> pl.DataFrame:
    """A schema-valid one-month panel with the given predictors (ret = 1.0)."""
    return make_panel([{"signalname": name, "yyyymm": 196301, "ret": 1.0} for name in signalnames])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.69", 5.69),
        (" 2.5 ", 2.5),
        ("+1.5", 1.5),
        ("-4.2", -4.2),
        ("3", 3.0),
        ("1e1", 10.0),
        (".5", 0.5),
        ("", None),
        ("   ", None),
        ("n/a", None),
        ("1,5", None),
        ("(3.0)", None),
        ("nan", None),
        ("inf", None),
        ("-inf", None),
        ("abc3", None),
    ],
)
def test_tstat_parsing_rules(tmp_path: Path, raw: str, expected: float | None) -> None:
    doc = write_signal_doc(tmp_path, [("S", raw)])
    out = tstats(one_row_panel("S"), doc)
    value = out["t_original"].item()
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


def test_weak_flag_boundary_and_sign_are_preserved(tmp_path: Path) -> None:
    doc = write_signal_doc(
        tmp_path,
        [
            ("AtHurdle", "3.0"),
            ("Below", "2.99"),
            ("Negative", "-8.0"),
            ("Above", "12.0"),
            ("Empty", None),
        ],
    )
    out = tstats(one_row_panel("AtHurdle", "Below", "Negative", "Above", "Empty"), doc)
    weak = {row["signalname"]: row["weak_original"] for row in out.iter_rows(named=True)}
    assert weak == {
        "AtHurdle": False,
        "Below": True,
        "Negative": True,
        "Above": False,
        "Empty": None,
    }


def test_hurdle_parameter_is_respected(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", "2.5")])
    panel = one_row_panel("S")
    assert tstats(panel, doc, hurdle_t=2.0)["weak_original"].item() is False
    assert tstats(panel, doc, hurdle_t=DEFAULT_HURDLE_T)["weak_original"].item() is True


def test_predictors_absent_from_the_signal_doc_are_null_and_counted(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("InPanel", "4.0"), ("NotInPanel", "9.0")])
    out = tstats(one_row_panel("InPanel", "Absent"), doc)

    assert out["signalname"].to_list() == ["Absent", "InPanel"]
    values = {row["signalname"]: row["t_original"] for row in out.iter_rows(named=True)}
    assert values == {"Absent": None, "InPanel": 4.0}

    summary = tstat_summary(out)
    original = summary.filter(pl.col("tstat_source") == "original_paper")
    assert original["n_predictors"].item() == 2
    assert original["n_missing"].item() == 1


def test_signal_doc_without_tstat_column_gives_all_nulls(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", "1.0")], columns=["Acronym", "Year"])
    out = tstats(one_row_panel("S"), doc)
    assert out["t_original"].item() is None

    empty_doc = write_signal_doc(tmp_path, [("S", None)], columns=["Acronym", "Year"])
    assert tstats(one_row_panel("S"), empty_doc)["t_original"].item() is None


def test_non_predictor_rows_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "signal_doc.csv"
    path.write_text(
        "Acronym,Cat.Signal,T-Stat\nP,Predictor,2.0\nQ,Placebo,0.1\nR,Drop,0.2\n",
        encoding="utf-8",
    )
    out = tstats(one_row_panel("P", "Q", "R"), path)
    values = {row["signalname"]: row["t_original"] for row in out.iter_rows(named=True)}
    assert values == {"P": 2.0, "Q": None, "R": None}


def test_duplicate_doc_predictors_raise(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", "1.0"), ("S", "2.0")])
    with pytest.raises(TstatError, match="appears twice"):
        tstats(one_row_panel("S"), doc)


def test_doc_without_acronym_column_raises(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", "1.0")], columns=["Name", "T-Stat"])
    with pytest.raises(TstatError, match="Acronym"):
        tstats(one_row_panel("S"), doc)


def test_missing_signal_doc_raises(tmp_path: Path) -> None:
    with pytest.raises(TstatError, match="not found"):
        tstats(one_row_panel("S"), tmp_path / "does_not_exist.csv")


def test_replicated_tstat_matches_hand_computation(tmp_path: Path) -> None:
    values = [1.0, 2.0, 4.0]
    records: list[dict[str, object]] = [
        {"signalname": "S", "yyyymm": 196301 + index, "ret": value}
        for index, value in enumerate(values)
    ]
    # distractors: a post-sample month and a row of another portfolio must not enter
    records.append({"signalname": "S", "yyyymm": 196312, "ret": 100.0, "post_sample": 1})
    records.append({"signalname": "S", "yyyymm": 196304, "ret": -500.0, "portfolio": "deciles_ew"})
    panel = make_panel(records)
    doc = write_signal_doc(tmp_path, [("S", None)])

    out = tstats(panel, doc)
    expected = statistics.fmean(values) / (statistics.stdev(values) / math.sqrt(len(values)))
    assert out["t_replicated"].item() == pytest.approx(expected, rel=1e-12)
    assert out["t_original"].item() is None


def test_replicated_tstat_null_for_short_or_constant_samples(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {"signalname": "One", "yyyymm": 196301, "ret": 3.0},
        {"signalname": "Constant", "yyyymm": 196301, "ret": 2.0},
        {"signalname": "Constant", "yyyymm": 196302, "ret": 2.0},
        {"signalname": "Constant", "yyyymm": 196303, "ret": 2.0},
    ]
    panel = make_panel(records)
    doc = write_signal_doc(tmp_path, [("One", "9.0"), ("Constant", "9.0")])

    out = tstats(panel, doc).sort("signalname")
    values = {row["signalname"]: row["t_replicated"] for row in out.iter_rows(named=True)}
    assert values == {"Constant": None, "One": None}
    assert out["weak_replicated"].to_list() == [None, None]

    summary = tstat_summary(out)
    replicated = summary.filter(pl.col("tstat_source") == "in_sample_replicated")
    assert replicated["n_missing"].item() == 2


def test_replicated_tstat_uses_the_sign_of_the_return(tmp_path: Path) -> None:
    records: list[dict[str, object]] = [
        {"signalname": "Down", "yyyymm": 196301, "ret": -1.0},
        {"signalname": "Down", "yyyymm": 196302, "ret": -2.0},
        {"signalname": "Down", "yyyymm": 196303, "ret": -3.0},
    ]
    panel = make_panel(records)
    doc = write_signal_doc(tmp_path, [("Down", None)])
    out = tstats(panel, doc)
    expected = -2.0 / (1.0 / math.sqrt(3.0))
    assert out["t_replicated"].item() == pytest.approx(expected, rel=1e-12)
    assert out["weak_replicated"].item() is True


def test_tstats_output_schema(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", "2.0")])
    out = tstats(one_row_panel("S"), doc)

    assert list(TSTATS_SCHEMA) == [
        "signalname",
        "t_original",
        "t_replicated",
        "weak_original",
        "weak_replicated",
    ]
    assert out.columns == list(TSTATS_SCHEMA)
    assert out.schema == TSTATS_SCHEMA

    assert {
        "tstat_source": pl.Utf8,
        "n_predictors": pl.Int64,
        "n_missing": pl.Int64,
        "n_below_hurdle": pl.Int64,
        "median_t": pl.Float64,
        "hurdle_t": pl.Float64,
    } == TSTAT_SUMMARY_SCHEMA
    summary = tstat_summary(out)
    assert summary.columns == list(TSTAT_SUMMARY_SCHEMA)
    assert summary.schema == TSTAT_SUMMARY_SCHEMA


def test_replicated_tstat_uses_the_sample_standard_deviation(tmp_path: Path) -> None:
    """The ddof=1 convention matters: 0, 1, 3 gives 4/sqrt(7) with ddof=1."""
    records = [
        {"signalname": "S", "yyyymm": 196301 + index, "ret": value}
        for index, value in enumerate([0.0, 1.0, 3.0])
    ]
    panel = make_panel(records)
    out = tstats(panel, write_signal_doc(tmp_path, [("S", None)]))
    assert out["t_replicated"].item() == pytest.approx(4.0 / math.sqrt(7.0), rel=1e-12)


def test_tstat_summary_counts_median_and_hurdle(tmp_path: Path) -> None:
    doc = write_signal_doc(
        tmp_path,
        [("Down", "-4.0"), ("Exactly", "3.0"), ("Missing", None), ("Up", "5.0")],
    )
    records: list[dict[str, object]] = [
        {"signalname": "Down", "yyyymm": 196301, "ret": -1.0},
        {"signalname": "Down", "yyyymm": 196302, "ret": -2.0},
        {"signalname": "Down", "yyyymm": 196303, "ret": -3.0},
        {"signalname": "Exactly", "yyyymm": 196301, "ret": 0.0},
        {"signalname": "Missing", "yyyymm": 196301, "ret": 1.0},
        {"signalname": "Up", "yyyymm": 196301, "ret": 2.0},
    ]
    panel = make_panel(records)
    out = tstats(panel, doc)
    summary = tstat_summary(out)

    assert summary["tstat_source"].to_list() == ["in_sample_replicated", "original_paper"]

    original = summary.filter(pl.col("tstat_source") == "original_paper").row(0, named=True)
    assert original["n_predictors"] == 4
    assert original["n_missing"] == 1
    assert original["n_below_hurdle"] == 1
    assert original["median_t"] == pytest.approx(statistics.median([-4.0, 3.0, 5.0]))
    assert original["hurdle_t"] == DEFAULT_HURDLE_T

    replicated = summary.filter(pl.col("tstat_source") == "in_sample_replicated").row(0, named=True)
    assert replicated["n_predictors"] == 4
    assert replicated["n_missing"] == 3
    assert replicated["n_below_hurdle"] == 1
    assert replicated["median_t"] == pytest.approx(-2.0 / (1.0 / math.sqrt(3.0)))


def test_tstat_summary_median_is_null_when_everything_is_missing(tmp_path: Path) -> None:
    doc = write_signal_doc(tmp_path, [("S", None)])
    summary = tstat_summary(tstats(one_row_panel("S"), doc))
    assert summary["median_t"].unique().to_list() == [None]
    assert summary["n_below_hurdle"].unique().to_list() == [0]


def test_tstat_summary_rejects_a_frame_without_the_t_columns() -> None:
    with pytest.raises(TstatError, match="missing columns"):
        tstat_summary(pl.DataFrame({"signalname": ["S"]}))
