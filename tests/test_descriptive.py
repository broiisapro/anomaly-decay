"""Tests for the descriptive results (predictor summary, event time, coverage)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate.descriptive import coverage, event_time, main, predictor_summary
from anomaly_decay.io import read_results
from anomaly_decay.schemas import PANEL_SCHEMA, RESULTS_SCHEMAS, validate_frame


def _yyyymm(first: int, count: int) -> list[int]:
    year, month = divmod(first, 100)
    out = []
    for _ in range(count):
        out.append(year * 100 + month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _panel(
    *,
    values: dict[str, list[float]],
    portfolios: dict[str, tuple[str, ...]] | None = None,
    anchor: int = 202101,
    windows: tuple[str, ...] = ("in_sample", "post_sample", "post_publication"),
) -> pl.DataFrame:
    """Hand-built panel per PANEL_SCHEMA: month ``anchor + i`` carries ``values[sig][i]``.

    Every signal has the same months. ``windows`` repeats to cover the length of the
    longest series (the tail repeats the last window). ``portfolios`` maps a signal to
    the portfolios it appears in; other signals get ``("op",)`` only.
    """
    n_rows = len(next(iter(values.values())))
    months = _yyyymm(anchor, n_rows)
    splits = [0, 1, max(1, n_rows - 1)]
    records = []
    for signalname, series in values.items():
        for index, (yyyymm, ret) in enumerate(zip(months, series, strict=True)):
            window = (
                windows[0] if index < splits[1] else windows[1] if index < splits[2] else windows[2]
            )
            for portfolio in (portfolios or {}).get(signalname, ("op",)):
                records.append(
                    {
                        "signalname": signalname,
                        "portfolio": portfolio,
                        "yyyymm": yyyymm,
                        "month": f"{yyyymm // 100:04d}-{yyyymm % 100:02d}-01",
                        "ret": ret,
                        "window": window,
                        "post_sample": int(window == "post_sample"),
                        "post_publication": int(window == "post_publication"),
                        "window_alt": window,
                        "post_sample_alt": int(window == "post_sample"),
                        "post_publication_alt": int(window == "post_publication"),
                        "event_time_sample_end": index,
                        "event_time_pub": index - 12,
                        "is_mean": 1.0,
                        "ret_norm": ret,
                        "sample_start_year": 1980,
                        "sample_end_year": 2020,
                        "pub_year": 2021,
                        "clear": True,
                        "mkt_rf": 0.1,
                        "smb": 0.2,
                        "hml": 0.3,
                        "rmw": 0.4,
                        "cma": 0.5,
                        "umd": 0.6,
                        "rf": 0.01,
                    }
                )
    drops = [
        "event_time_sample_end",
        "event_time_pub",
        "post_sample",
        "post_sample_alt",
        "post_publication",
        "post_publication_alt",
        "is_mean",
        "ret_norm",
        "sample_start_year",
        "sample_end_year",
        "pub_year",
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "umd",
        "rf",
    ]
    frame = pl.DataFrame(records).with_columns(
        pl.col("month").str.to_date("%Y-%m-%d"),
        *(pl.col(column).cast(PANEL_SCHEMA[column]) for column in drops),
        pl.col("yyyymm").cast(pl.Int32),
    )
    return frame.select(list(PANEL_SCHEMA)).sort(["signalname", "portfolio", "yyyymm"])


def _metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": ["A", "B", "C", "D"],
            "sample_start_year": pl.Series([1980, 1985, 1990, 2000], dtype=pl.Int16),
            "sample_end_year": pl.Series([2020, 2015, 2012, 2019], dtype=pl.Int16),
            "pub_year": pl.Series([2021, 2013, 2014, 2013], dtype=pl.Int16),
            "journal": ["JF", "JFE", "RFS", "JF"],
            "authors": ["Alpha", "Beta", "Gamma", "Delta"],
            "cat_economic": ["x", "y", "z", "w"],
            "cat_data": ["Price", "Price", "Accounting", "Other"],
            "clear": [True, True, False, True],
            "date_source": ["signaldoc"] * 4,
        }
    )


def test_predictor_summary_by_window() -> None:
    panel = _panel(values={"A": [1.0, 2.0, 3.0, 4.0]})
    summary = predictor_summary(panel)
    validate_frame(summary, RESULTS_SCHEMAS["predictor_summary.csv"])
    assert summary["signalname"].to_list() == ["A", "A", "A"]
    assert summary["window"].to_list() == ["in_sample", "post_publication", "post_sample"]
    assert summary["n_months"].to_list() == [1, 1, 2]
    assert summary["mean_ret"].to_list() == [1.0, 4.0, 2.5]
    assert summary["se_mean"].to_list()[0] is None  # n = 1
    assert summary["se_mean"].to_list()[2] == pytest.approx(0.5)  # sd([2, 3]) / sqrt(2)


def test_predictor_summary_uses_op_only() -> None:
    panel = _panel(
        values={"A": [1.0, 2.0, 3.0, 4.0]},
        portfolios={"A": ("op", "deciles_ew")},
    )
    summary = predictor_summary(panel)
    assert len(summary) == 3
    assert summary.filter(pl.col("window") == "post_sample")["mean_ret"].to_list() == [2.5]


def test_event_time_cross_predictor_means() -> None:
    panel = _panel(values={"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]})
    epochs = event_time(panel, lo=0, hi=2)
    validate_frame(epochs, RESULTS_SCHEMAS["event_time.csv"])
    assert epochs["anchor"].to_list() == ["sample_end"] * 3
    assert epochs["event_month"].to_list() == [0, 1, 2]
    assert epochs["mean_ret"].to_list() == [2.0, 3.0, 4.0]
    assert epochs["n_predictors"].to_list() == [2, 2, 2]
    assert epochs["se"].to_list()[0] == pytest.approx(1.0)  # sd([1, 3]) / sqrt(2)

    single = event_time(panel.filter(pl.col("signalname") == "A"), lo=0, hi=0)
    assert single["se"].to_list() == [None]  # n_predictors = 1
    assert single["n_predictors"].to_list() == [1]


def test_event_time_publication_anchor_and_bounds() -> None:
    panel = _panel(values={"A": [1.0] * 24, "B": [2.0] * 24})
    epochs = event_time(panel, lo=-2, hi=1)
    validate_frame(epochs, RESULTS_SCHEMAS["event_time.csv"])
    publication = epochs.filter(pl.col("anchor") == "publication")
    assert publication["event_month"].to_list() == [-2, -1, 0, 1]
    assert publication["mean_ret"].to_list() == [1.5, 1.5, 1.5, 1.5]
    assert publication["n_predictors"].to_list() == [2, 2, 2, 2]

    full = event_time(panel)
    assert set(full["anchor"].unique().to_list()) == {"sample_end", "publication"}
    assert full.equals(full.sort(["anchor", "event_month"]))
    assert full.filter(pl.col("anchor") == "publication")["event_month"].min() == -12
    assert full.filter(pl.col("anchor") == "sample_end")["event_month"].min() == 0
    # both signals cover every sample-end event month of the 24-month window
    assert full.filter(pl.col("anchor") == "sample_end")["n_predictors"].to_list() == [2] * 24


def test_coverage_items() -> None:
    panel = _panel(
        values={"A": [1.0] * 3, "B": [2.0] * 3},
        portfolios={"A": ("op", "deciles_ew"), "B": ("op", "deciles_ew", "deciles_vw")},
    )
    result = coverage(panel, _metadata(), 202510)
    validate_frame(result, RESULTS_SCHEMAS["coverage.csv"])
    values = dict(result.iter_rows())
    assert values == {
        "vintage": "202510",
        "first_month": "202101",
        "last_month": "202103",
        "n_predictors_op": "2",
        "n_predictors_deciles_ew": "2",
        "n_predictors_deciles_vw": "1",
        "n_clear": "3",
        "n_published_by_2013": "2",
        "n_predictor_months_op": "6",
    }
    assert result["item"].to_list() == sorted(values)


def test_coverage_is_text_and_sorted() -> None:
    panel = _panel(values={"A": [1.0] * 3})
    result = coverage(panel, _metadata(), 202510)
    assert result["value"].dtype == pl.Utf8
    assert result["item"].to_list() == sorted(result["item"].to_list())


def test_cli_writes_three_files_and_is_byte_identical(tmp_path: Path) -> None:
    panel = _panel(values={"A": [1.0, 2.0, 3.0], "B": [3.0, 4.0, 5.0]})
    panel_path = tmp_path / "panel.parquet"
    meta_path = tmp_path / "predictors.parquet"
    panel.write_parquet(panel_path)
    _metadata().write_parquet(meta_path)
    out_one = tmp_path / "results-one"
    out_two = tmp_path / "results-two"
    assert (
        main(
            [
                "--panel",
                str(panel_path),
                "--metadata",
                str(meta_path),
                "--vintage",
                "202510",
                "--results-dir",
                str(out_one),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--panel",
                str(panel_path),
                "--metadata",
                str(meta_path),
                "--vintage",
                "202510",
                "--results-dir",
                str(out_two),
            ]
        )
        == 0
    )
    for name in ("predictor_summary.csv", "event_time.csv", "coverage.csv"):
        assert (out_one / name).read_bytes() == (out_two / name).read_bytes()
        assert read_results(name, out_one).schema == RESULTS_SCHEMAS[name]


def test_cli_rejects_wrong_metadata_schema(tmp_path: Path) -> None:
    panel = _panel(values={"A": [1.0] * 3})
    panel_path = tmp_path / "panel.parquet"
    meta_path = tmp_path / "predictors.parquet"
    panel.write_parquet(panel_path)
    _metadata().drop("clear").write_parquet(meta_path)
    with pytest.raises(Exception, match="columns"):
        main(
            [
                "--panel",
                str(panel_path),
                "--metadata",
                str(meta_path),
                "--vintage",
                "202510",
                "--results-dir",
                str(tmp_path / "results"),
            ]
        )


def test_empty_event_time_frame_has_schema() -> None:
    panel = _panel(values={"A": [1.0] * 3}).drop("event_time_pub", "event_time_sample_end")
    epochs = event_time(panel)
    assert epochs.height == 0
    assert epochs.schema == {
        "anchor": pl.Utf8,
        "event_month": pl.Int64,
        "mean_ret": pl.Float64,
        "se": pl.Float64,
        "n_predictors": pl.Int64,
    }
