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
from anomaly_decay.config import load_phase2_preregistration, load_specifications
from anomaly_decay.estimate.multiple_testing import (
    DEFAULT_HURDLE_T,
    TSTATS_SCHEMA,
    TstatError,
    run_multiple_testing,
    tstat_summary,
    tstats,
)
from anomaly_decay.estimate.multiple_testing import (
    main as mt_cli,
)
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

from test_estimate import SPECS_PATH, make_panel

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"


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

    summary = tstat_summary(out)
    assert summary.columns == list(RESULTS_SCHEMAS["tstat_summary.csv"])
    assert summary.schema == RESULTS_SCHEMAS["tstat_summary.csv"]


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


PREREG_TEXT = """version: 1

bootstrap:
  id: bootstrap_main
  base_spec: main
  method: moving_block_by_month
  block_length_months: 12
  reps: 2000
  seed: 2027
  statistics: [coef_post_sample, coef_post_publication,
               decay_pct_post_sample, decay_pct_post_publication]
  interval: percentile
  level: 0.95
  exploratory: false

multiple_testing:
  hurdle_t: 3.0
  weak_rule: tstat_below_hurdle
  missing_tstat: exclude_and_count
  tstat_sources:
    original_paper: "OSAP SignalDoc column T-Stat"
    in_sample_replicated: "mean / (sd / sqrt(n)) in-sample"
  specs:
    - id: mt_weak_original
      base_spec: main
      tstat_source: original_paper
      extra_terms: [post_sample_x_weak, post_publication_x_weak]
      exploratory: false
    - id: mt_weak_replicated
      base_spec: main
      tstat_source: in_sample_replicated
      extra_terms: [post_sample_x_weak, post_publication_x_weak]
      exploratory: false
    - id: mt_tstat_continuous
      base_spec: main
      tstat_source: in_sample_replicated
      extra_terms: [post_sample_x_tstat, post_publication_x_tstat]
      exploratory: false

vintages:
  primary: 202510
  compare_preference: [202410]
  predictor_set: intersection
  specs: [main, full_extension]
  metrics: [coef, se, decay_pct, n_predictors, n_obs]
  exploratory: false
"""


def write_prereg(tmp_path: Path, text: str = PREREG_TEXT) -> Path:
    path = tmp_path / "phase2_preregistration.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def write_mt_signal_doc(tmp_path: Path, tstats_map: dict[str, float | None]) -> Path:
    path = tmp_path / "signal_doc_mt.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Acronym", "Cat.Signal", "T-Stat"])
        for name, value in tstats_map.items():
            writer.writerow([name, "Predictor", "" if value is None else f"{value}"])
    return path


def weak_decay_panel(amplitudes: dict[str, float] | None = None) -> pl.DataFrame:
    """Four predictors, in-sample level 1.0, weak predictors decay more.

    1963-1964 in-sample (level 1.0), 1965 post-sample, 1966 post-publication.
    Strong predictors drop to 0.8 / 0.6 and weak ones to 0.4 / 0.1, so the exact
    coefficients are post_sample -0.2, post_publication -0.4, post_sample_x_weak
    -0.4 and post_publication_x_weak -0.5. Each predictor's alternating monthly
    wiggle averages zero, so recovery is exact; its amplitude sets the
    replicated t (small amplitude -> large t). The W amplitudes 1.7 and 1.9 give
    replicated t 2.82 and 2.52, below the 3.0 hurdle, so the W group is weak
    under both t-statistic sources.
    """
    if amplitudes is None:
        amplitudes = {"W0": 1.7, "W1": 1.9, "S0": 0.10, "S1": 0.20}
    records: list[dict[str, object]] = []
    for signalname, amplitude in amplitudes.items():
        weak = signalname.startswith("W")
        for month_index in range(48):
            year = 1963 + month_index // 12
            if year <= 1964:
                level, post_sample, post_publication = 1.0, 0, 0
            elif year == 1965:
                level, post_sample, post_publication = (0.4 if weak else 0.8), 1, 0
            else:
                level, post_sample, post_publication = (0.1 if weak else 0.6), 0, 1
            wiggle = amplitude * (1 if month_index % 2 == 0 else -1)
            records.append(
                {
                    "signalname": signalname,
                    "yyyymm": year * 100 + month_index % 12 + 1,
                    "ret": level + wiggle,
                    "post_sample": post_sample,
                    "post_publication": post_publication,
                }
            )
    return make_panel(records)


def mt_table(panel: pl.DataFrame, doc: Path, prereg_path: Path) -> pl.DataFrame:
    stats = tstats(panel, doc)
    prereg = load_phase2_preregistration(prereg_path)
    return run_multiple_testing(
        panel, stats, prereg.multiple_testing, load_specifications(SPECS_PATH)
    )


def test_weak_original_recovers_known_interactions(tmp_path: Path) -> None:
    panel = weak_decay_panel()
    doc = write_mt_signal_doc(tmp_path, {"W0": 1.0, "W1": 2.0, "S0": 5.0, "S1": 9.0})
    table = mt_table(panel, doc, write_prereg(tmp_path))
    rows = {
        row["term"]: row
        for row in table.filter(pl.col("spec") == "mt_weak_original").iter_rows(named=True)
    }
    assert set(rows) == {
        "post_sample",
        "post_publication",
        "post_sample_x_weak",
        "post_publication_x_weak",
    }
    assert rows["post_sample"]["coef"] == pytest.approx(-0.2, abs=1e-10)
    assert rows["post_publication"]["coef"] == pytest.approx(-0.4, abs=1e-10)
    assert rows["post_sample_x_weak"]["coef"] == pytest.approx(-0.4, abs=1e-10)
    assert rows["post_publication_x_weak"]["coef"] == pytest.approx(-0.5, abs=1e-10)
    assert rows["post_sample"]["n_predictors"] == 4
    assert rows["post_sample"]["n_weak"] == 2
    assert rows["post_sample"]["n_excluded_missing_t"] == 0
    assert rows["post_sample"]["n_obs"] == 4 * 48
    assert rows["post_sample"]["tstat_source"] == "original_paper"

    # the same group is weak under the replicated source, so the replicated
    # weak specification recovers the same known interactions
    replicated = {
        row["term"]: row
        for row in table.filter(pl.col("spec") == "mt_weak_replicated").iter_rows(named=True)
    }
    assert replicated["post_sample_x_weak"]["coef"] == pytest.approx(-0.4, abs=1e-10)
    assert replicated["post_publication_x_weak"]["coef"] == pytest.approx(-0.5, abs=1e-10)
    assert replicated["post_sample"]["n_weak"] == 2


def test_missing_t_predictors_are_excluded_and_counted(tmp_path: Path) -> None:
    panel = weak_decay_panel()
    doc = write_mt_signal_doc(tmp_path, {"W0": 1.0, "W1": 2.0, "S0": 5.0})
    table = mt_table(panel, doc, write_prereg(tmp_path))
    rows = {
        row["term"]: row
        for row in table.filter(pl.col("spec") == "mt_weak_original").iter_rows(named=True)
    }
    assert rows["post_sample"]["n_predictors"] == 3
    assert rows["post_sample"]["n_excluded_missing_t"] == 1
    assert rows["post_sample"]["n_obs"] == 3 * 48
    assert rows["post_sample"]["n_weak"] == 2


def continuous_panel() -> pl.DataFrame:
    """Four predictors whose post-window changes are exactly linear in the replicated t.

    Post-sample change is ``-0.5 + 0.005 * t`` and post-publication change is
    ``-0.8 + 0.01 * t``, so the weaker the original evidence (smaller t) the
    larger the decay, and the continuous interaction coefficients are exactly
    0.005 and 0.01. The alternating wiggle amplitude ``sqrt(23) / t`` makes the
    in-sample replicated t equal the intended value.
    """
    t_values = {"P0": 60.0, "P1": 30.0, "P2": 12.0, "P3": 2.5}
    records: list[dict[str, object]] = []
    for signalname, t in t_values.items():
        amplitude = math.sqrt(23.0) / t
        post_sample_change = -0.5 + 0.005 * t
        post_publication_change = -0.8 + 0.01 * t
        for month_index in range(48):
            year = 1963 + month_index // 12
            if year <= 1964:
                level, post_sample, post_publication = 1.0, 0, 0
            elif year == 1965:
                level, post_sample, post_publication = 1.0 + post_sample_change, 1, 0
            else:
                level, post_sample, post_publication = 1.0 + post_publication_change, 0, 1
            wiggle = amplitude * (1 if month_index % 2 == 0 else -1)
            records.append(
                {
                    "signalname": signalname,
                    "yyyymm": year * 100 + month_index % 12 + 1,
                    "ret": level + wiggle,
                    "post_sample": post_sample,
                    "post_publication": post_publication,
                }
            )
    return make_panel(records)


def test_continuous_spec_recovers_the_linear_t_relation(tmp_path: Path) -> None:
    panel = continuous_panel()
    doc = write_mt_signal_doc(tmp_path, {"P0": 5.0, "P1": 1.0, "P2": 5.0, "P3": 1.0})
    table = mt_table(panel, doc, write_prereg(tmp_path))
    rows = {
        row["term"]: row
        for row in table.filter(pl.col("spec") == "mt_tstat_continuous").iter_rows(named=True)
    }
    assert rows["post_sample"]["coef"] == pytest.approx(-0.5, abs=1e-8)
    assert rows["post_sample_x_tstat"]["coef"] == pytest.approx(0.005, abs=1e-10)
    assert rows["post_publication"]["coef"] == pytest.approx(-0.8, abs=1e-8)
    assert rows["post_publication_x_tstat"]["coef"] == pytest.approx(0.01, abs=1e-10)
    # positive interaction: smaller t (weaker evidence) means a larger decline
    assert rows["post_sample_x_tstat"]["coef"] > 0.0
    assert rows["post_publication_x_tstat"]["coef"] > 0.0
    assert rows["post_sample"]["n_weak"] == 1


def test_fixture_panel_runs_all_three_specs(tmp_path: Path) -> None:
    panel = pl.read_parquet(FIXTURE_PANEL)
    names = panel["signalname"].unique().sort().to_list()
    doc = write_mt_signal_doc(
        tmp_path, {name: 1.0 if index % 3 == 0 else 5.0 for index, name in enumerate(names)}
    )
    table = mt_table(panel, doc, write_prereg(tmp_path))

    validate_frame(table, RESULTS_SCHEMAS["multiple_testing.csv"])
    assert table["spec"].unique().sort().to_list() == [
        "mt_tstat_continuous",
        "mt_weak_original",
        "mt_weak_replicated",
    ]
    assert table.height == 3 * 4
    assert table["n_predictors"].unique().to_list() == [12]
    assert table["n_obs"].unique().to_list() == [8574]
    assert table["n_excluded_missing_t"].unique().to_list() == [0]
    assert table["coef"].is_finite().all()
    assert table["se"].is_finite().all()


def test_cli_writes_schema_valid_files_byte_identically(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    weak_decay_panel().write_parquet(panel_path)
    doc = write_mt_signal_doc(tmp_path, {"W0": 1.0, "W1": 2.0, "S0": 5.0, "S1": 9.0})
    prereg_path = write_prereg(tmp_path)

    dirs = [tmp_path / "results_a", tmp_path / "results_b"]
    for results_dir in dirs:
        code = mt_cli(
            [
                "--panel",
                str(panel_path),
                "--signal-doc",
                str(doc),
                "--prereg",
                str(prereg_path),
                "--specs",
                str(SPECS_PATH),
                "--results-dir",
                str(results_dir),
            ]
        )
        assert code == 0

    for name in ("multiple_testing.csv", "tstat_summary.csv"):
        assert (dirs[0] / name).read_bytes() == (dirs[1] / name).read_bytes()

    table = read_results("multiple_testing.csv", dirs[0])
    validate_frame(table, RESULTS_SCHEMAS["multiple_testing.csv"])
    assert table.height == 12

    summary = read_results("tstat_summary.csv", dirs[0])
    validate_frame(summary, RESULTS_SCHEMAS["tstat_summary.csv"])
    assert summary.height == 2
    assert summary["n_predictors"].unique().to_list() == [4]


def test_cli_records_a_failing_spec_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = PREREG_TEXT.replace(
        "extra_terms: [post_sample_x_weak, post_publication_x_weak]",
        "extra_terms: [bogus_x_weak, post_publication_x_weak]",
        1,
    )
    panel_path = tmp_path / "panel.parquet"
    weak_decay_panel().write_parquet(panel_path)
    doc = write_mt_signal_doc(tmp_path, {"W0": 1.0, "W1": 2.0, "S0": 5.0, "S1": 9.0})
    results_dir = tmp_path / "results"
    code = mt_cli(
        [
            "--panel",
            str(panel_path),
            "--signal-doc",
            str(doc),
            "--prereg",
            str(write_prereg(tmp_path, broken)),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(results_dir),
        ]
    )
    assert code == 1
    assert "failed" in capsys.readouterr().err

    table = read_results("multiple_testing.csv", results_dir)
    assert table["spec"].unique().sort().to_list() == ["mt_tstat_continuous", "mt_weak_replicated"]
    assert table.height == 8
    assert (results_dir / "tstat_summary.csv").is_file()


def test_committed_preregistration_loads_with_the_expected_specs() -> None:
    path = REPO_ROOT / "config" / "phase2_preregistration.yaml"
    if not path.is_file():
        pytest.skip("config/phase2_preregistration.yaml not merged yet")
    prereg = load_phase2_preregistration(path).multiple_testing
    assert prereg.hurdle_t == 3.0
    assert [spec.id for spec in prereg.specs] == [
        "mt_weak_original",
        "mt_weak_replicated",
        "mt_tstat_continuous",
    ]
    assert [spec.tstat_source for spec in prereg.specs] == [
        "original_paper",
        "in_sample_replicated",
        "in_sample_replicated",
    ]
