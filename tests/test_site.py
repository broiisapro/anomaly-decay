"""Tests for the companion page's JSON data (Phase 3 definition of done).

Two variants:

- the full-data variant runs against the committed ``results/`` once #112 has produced
  ``predictor_info.csv`` and ``predictor_event_time.csv``; it asserts that every
  predictor has a file and that every value equals the results table it came from.
- the fixture variant builds a small results directory from the committed fictional
  fixture plus a hand-built metadata and event table, so the site path is exercised in
  CI before #112 lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate.predictor_profiles import predictor_info
from anomaly_decay.io import read_results, write_results
from anomaly_decay.schemas import PREDICTORS_SCHEMA, RESULTS_SCHEMAS
from anomaly_decay.site import SiteError, round_float, write_site_data

REPO_ROOT = Path(__file__).parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIXTURE_RESULTS = REPO_ROOT / "tests" / "fixtures" / "results_fixture"
PROFILE_INPUTS = ("predictor_info.csv", "predictor_summary.csv", "predictor_event_time.csv")

requires_full_results = pytest.mark.skipif(
    not all((RESULTS_DIR / name).is_file() for name in PROFILE_INPUTS),
    reason="results/predictor_info.csv and results/predictor_event_time.csv land in #112",
)


def _index(out: Path) -> list[dict[str, object]]:
    return json.loads((out / "predictors.json").read_text(encoding="utf-8"))


def _profile(out: Path, signalname: str) -> dict[str, object]:
    path = out / "predictors" / f"{signalname}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": ["AM", "Size", "Mom12m"],
            "sample_start_year": [1963, 1926, 1965],
            "sample_end_year": [1990, 1975, 1989],
            "pub_year": [1992, 1981, 1993],
            "journal": ["JF", "JFE", "JF"],
            "authors": ["Fama and French", "Banz", "Jegadeesh and Titman"],
            "cat_economic": ["valuation", "size", "momentum"],
            "cat_data": ["Price", "Price", "Price"],
            "clear": [True, True, True],
            "date_source": ["signaldoc", "signaldoc", "signaldoc"],
        },
        schema=PREDICTORS_SCHEMA,
    )


def _fixture_event_time() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "signalname": "AM",
                "anchor": "sample_end",
                "event_year": -1,
                "mean_ret": 0.5,
                "n_months": 12,
            },
            {
                "signalname": "AM",
                "anchor": "sample_end",
                "event_year": 0,
                "mean_ret": -0.25,
                "n_months": 12,
            },
            {
                "signalname": "AM",
                "anchor": "publication",
                "event_year": -1,
                "mean_ret": 0.75,
                "n_months": 12,
            },
            {
                "signalname": "AM",
                "anchor": "publication",
                "event_year": 0,
                "mean_ret": 0.1,
                "n_months": 12,
            },
            {
                "signalname": "Size",
                "anchor": "sample_end",
                "event_year": 0,
                "mean_ret": 0.2,
                "n_months": 6,
            },
            {
                "signalname": "Mom12m",
                "anchor": "publication",
                "event_year": 2,
                "mean_ret": -0.3,
                "n_months": 3,
            },
        ],
        schema=RESULTS_SCHEMAS["predictor_event_time.csv"],
    )


def _fixture_results(tmp_path: Path) -> Path:
    """A results directory with the three inputs and three predictors."""
    names = ["AM", "Size", "Mom12m"]
    results = tmp_path / "fixture-results"
    results.mkdir()
    write_results(predictor_info(_fixture_metadata()), "predictor_info.csv", results)
    summary = read_results("predictor_summary.csv", FIXTURE_RESULTS).filter(
        pl.col("signalname").is_in(names)
    )
    write_results(summary, "predictor_summary.csv", results)
    write_results(_fixture_event_time(), "predictor_event_time.csv", results)
    return results


def test_round_float_is_fixed_and_normalises_zero() -> None:
    assert round_float(0.123456789) == 0.123457
    assert round_float(2.0) == 2.0
    assert round_float(-0.0000001) == 0.0
    assert round_float(None) is None
    assert round_float(float("nan")) is None
    assert round_float(float("inf")) is None


def test_fixture_results_build_every_profile(tmp_path: Path) -> None:
    results = _fixture_results(tmp_path)
    out = tmp_path / "out"
    build = write_site_data(results, out)

    assert build.n_predictors == 3
    assert build.out_dir == out
    assert build.n_bytes > 0
    assert sorted(path.stem for path in (out / "predictors").glob("*.json")) == [
        "AM",
        "Mom12m",
        "Size",
    ]
    assert [row["signalname"] for row in _index(out)] == ["AM", "Mom12m", "Size"]
    assert _index(out)[0] == {
        "signalname": "AM",
        "authors": "Fama and French",
        "journal": "JF",
        "pub_year": 1992,
        "sample_start_year": 1963,
        "sample_end_year": 1990,
        "clear": True,
    }

    am = _profile(out, "AM")
    assert am["signalname"] == "AM"
    assert am["authors"] == "Fama and French"
    assert am["journal"] == "JF"
    assert am["pub_year"] == 1992
    assert am["sample_start_year"] == 1963
    assert am["sample_end_year"] == 1990
    assert am["clear"] is True
    assert am["cat_economic"] == "valuation"

    # values match the committed fictional fixture predictor_summary.csv exactly
    assert am["windows"]["in_sample"] == {"mean_ret": 0.333, "se_mean": 0.222, "n_months": 300}
    assert am["windows"]["post_sample"] == {"mean_ret": 0.111, "se_mean": 0.333, "n_months": 20}
    assert am["windows"]["post_publication"] == {
        "mean_ret": -0.111,
        "se_mean": 0.3108,
        "n_months": 60,
    }

    assert am["event_time"]["sample_end"] == [
        {"event_year": -1, "mean_ret": 0.5, "n_months": 12},
        {"event_year": 0, "mean_ret": -0.25, "n_months": 12},
    ]
    assert am["event_time"]["publication"] == [
        {"event_year": -1, "mean_ret": 0.75, "n_months": 12},
        {"event_year": 0, "mean_ret": 0.1, "n_months": 12},
    ]

    # a predictor with no rows for an anchor has an empty series, not an error
    assert _profile(out, "Size")["event_time"]["publication"] == []
    assert _profile(out, "Mom12m")["event_time"]["sample_end"] == []


def test_missing_window_is_null_with_zero_months(tmp_path: Path) -> None:
    results = _fixture_results(tmp_path)
    summary = read_results("predictor_summary.csv", results)
    write_results(
        summary.filter(~((pl.col("signalname") == "AM") & (pl.col("window") == "post_sample"))),
        "predictor_summary.csv",
        results,
    )
    out = tmp_path / "out"
    write_site_data(results, out)
    assert _profile(out, "AM")["windows"]["post_sample"] == {
        "mean_ret": None,
        "se_mean": None,
        "n_months": 0,
    }


def test_rebuild_is_byte_identical_and_removes_stale_files(tmp_path: Path) -> None:
    results = _fixture_results(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_site_data(results, first)
    (first / "predictors" / "STALE.json").write_text("{}", encoding="utf-8")
    write_site_data(results, second)
    assert not (second / "predictors" / "STALE.json").exists()
    for path in sorted(first.rglob("*.json")):
        relative = path.relative_to(first)
        if relative.name == "STALE.json":
            continue
        assert path.read_bytes() == (second / relative).read_bytes()


def test_missing_results_file_fails_instead_of_writing_an_empty_site(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        write_site_data(tmp_path / "empty", out)
    assert not out.exists()


def test_cli_reports_a_missing_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from anomaly_decay.site import main

    assert main(["--results-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "out")]) == 1
    assert "site: missing results file" in capsys.readouterr().err


def test_predictor_absent_from_the_index_is_rejected(tmp_path: Path) -> None:
    results = _fixture_results(tmp_path)
    summary = read_results("predictor_summary.csv", results)
    extra = pl.DataFrame(
        [
            {
                "signalname": "NOPE",
                "window": "in_sample",
                "n_months": 12,
                "mean_ret": 0.1,
                "se_mean": 0.1,
            }
        ],
        schema=RESULTS_SCHEMAS["predictor_summary.csv"],
    )
    write_results(pl.concat([summary, extra]), "predictor_summary.csv", results)
    with pytest.raises(SiteError, match="absent from predictor_info.csv"):
        write_site_data(results, tmp_path / "out")


def test_signalname_that_is_not_a_file_name_is_rejected(tmp_path: Path) -> None:
    results = _fixture_results(tmp_path)
    metadata = _fixture_metadata().with_columns(
        pl.when(pl.col("signalname") == "AM")
        .then(pl.lit("../escape"))
        .otherwise(pl.col("signalname"))
        .alias("signalname")
    )
    write_results(predictor_info(metadata), "predictor_info.csv", results)
    with pytest.raises(SiteError, match="cannot be used as file names"):
        write_site_data(results, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@requires_full_results
def test_every_predictor_has_a_data_file(tmp_path: Path) -> None:
    """The Phase 3 definition of done: all 212 predictors have a site profile."""
    info = read_results("predictor_info.csv", RESULTS_DIR)
    summary = read_results("predictor_summary.csv", RESULTS_DIR)
    out = tmp_path / "site-data"
    build = write_site_data(RESULTS_DIR, out)

    names = sorted(info["signalname"].to_list())
    assert build.n_predictors == len(names) == 212
    assert sorted(set(summary["signalname"].to_list())) == names
    assert sorted(path.stem for path in (out / "predictors").glob("*.json")) == names
    assert [row["signalname"] for row in _index(out)] == names

    for entry in _index(out):
        assert set(entry) == {
            "signalname",
            "authors",
            "journal",
            "pub_year",
            "sample_start_year",
            "sample_end_year",
            "clear",
        }
        row = info.filter(pl.col("signalname") == entry["signalname"]).row(0, named=True)
        assert entry["authors"] == row["authors"]
        assert entry["journal"] == row["journal"]
        assert entry["pub_year"] == row["pub_year"]
        assert entry["sample_start_year"] == row["sample_start_year"]
        assert entry["sample_end_year"] == row["sample_end_year"]
        assert entry["clear"] == row["clear"]


@requires_full_results
def test_profile_values_match_the_results_exactly(tmp_path: Path) -> None:
    out = tmp_path / "site-data"
    write_site_data(RESULTS_DIR, out)

    info = read_results("predictor_info.csv", RESULTS_DIR)
    summary = read_results("predictor_summary.csv", RESULTS_DIR)
    event_time = read_results("predictor_event_time.csv", RESULTS_DIR)
    summary_rows = {
        (row["signalname"], row["window"]): row for row in summary.iter_rows(named=True)
    }
    event_rows: dict[str, list[dict[str, object]]] = {}
    for row in event_time.iter_rows(named=True):
        event_rows.setdefault(str(row["signalname"]), []).append(row)

    for info_row in info.iter_rows(named=True):
        name = str(info_row["signalname"])
        profile = _profile(out, name)
        for column in (
            "authors",
            "journal",
            "pub_year",
            "sample_start_year",
            "sample_end_year",
            "clear",
        ):
            assert profile[column] == info_row[column], (name, column)
        for window in ("in_sample", "post_sample", "post_publication"):
            cell = profile["windows"][window]
            expected = summary_rows.get((name, window))
            if expected is None:
                assert cell == {"mean_ret": None, "se_mean": None, "n_months": 0}
            else:
                assert cell["mean_ret"] == round_float(expected["mean_ret"]), (name, window)
                assert cell["se_mean"] == round_float(expected["se_mean"]), (name, window)
                assert cell["n_months"] == expected["n_months"], (name, window)
        for anchor in ("sample_end", "publication"):
            expected_rows = [row for row in event_rows.get(name, []) if row["anchor"] == anchor]
            assert profile["event_time"][anchor] == [
                {
                    "event_year": row["event_year"],
                    "mean_ret": round_float(row["mean_ret"]),
                    "n_months": row["n_months"],
                }
                for row in expected_rows
            ], (name, anchor)


@requires_full_results
def test_full_site_files_are_valid_json_with_no_nan(tmp_path: Path) -> None:
    out = tmp_path / "site-data"
    write_site_data(RESULTS_DIR, out)
    for path in sorted(out.rglob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "Infinity" not in text
        json.loads(text)
