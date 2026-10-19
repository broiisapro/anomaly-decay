"""Tests for the Phase 2 results schemas and the pre-registration loader.

The Phase 2 files (``bootstrap.csv``, ``multiple_testing.csv``, ``tstat_summary.csv``,
``vintage_comparison.csv``) and the pre-registration text are frozen contracts: the
tested column order, dtypes, sort keys and loader values are written out literally here
so that a change to the schema modules cannot silently change the contract.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from anomaly_decay.config import (
    ConfigError,
    Phase2Prereg,
    load_phase2_preregistration,
)
from anomaly_decay.io import RESULTS_SORT_KEYS, read_results, write_results
from anomaly_decay.schemas import RESULTS_SCHEMAS
from polars.datatypes import DataTypeClass

REPO_ROOT = Path(__file__).parents[1]
COMMITTED_PREREG = REPO_ROOT / "config" / "phase2_preregistration.yaml"
COMMITTED_SKIP_REASON = "config/phase2_preregistration.yaml is not merged yet (issue #57)"

PHASE2_SCHEMAS: dict[str, tuple[tuple[str, DataTypeClass], ...]] = {
    "bootstrap.csv": (
        ("spec", pl.Utf8),
        ("statistic", pl.Utf8),
        ("estimate", pl.Float64),
        ("boot_mean", pl.Float64),
        ("boot_se", pl.Float64),
        ("ci_low", pl.Float64),
        ("ci_high", pl.Float64),
        ("reps", pl.Int64),
        ("block_length", pl.Int64),
        ("seed", pl.Int64),
    ),
    "multiple_testing.csv": (
        ("spec", pl.Utf8),
        ("tstat_source", pl.Utf8),
        ("term", pl.Utf8),
        ("coef", pl.Float64),
        ("se", pl.Float64),
        ("t", pl.Float64),
        ("n_predictors", pl.Int64),
        ("n_obs", pl.Int64),
        ("n_weak", pl.Int64),
        ("n_excluded_missing_t", pl.Int64),
    ),
    "tstat_summary.csv": (
        ("tstat_source", pl.Utf8),
        ("n_predictors", pl.Int64),
        ("n_missing", pl.Int64),
        ("n_below_hurdle", pl.Int64),
        ("median_t", pl.Float64),
        ("hurdle_t", pl.Float64),
    ),
    "vintage_comparison.csv": (
        ("spec", pl.Utf8),
        ("vintage", pl.Int64),
        ("term", pl.Utf8),
        ("coef", pl.Float64),
        ("se", pl.Float64),
        ("decay_pct", pl.Float64),
        ("n_predictors", pl.Int64),
        ("n_obs", pl.Int64),
        ("predictor_set", pl.Utf8),
    ),
}

PHASE2_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "bootstrap.csv": ("spec", "statistic"),
    "multiple_testing.csv": ("spec", "term"),
    "tstat_summary.csv": ("tstat_source",),
    "vintage_comparison.csv": ("spec", "vintage", "term"),
}

PREREG_TEXT = """\
# Phase 2 analyses for anomaly-decay, pre-registered before the first full-data estimation.
# Frozen together with config/specifications.yaml by the annotated tag `specs-frozen`.
# Anything added after the tag must set `exploratory: true` and is reported as such.
version: 1

bootstrap:
  id: bootstrap_main
  base_spec: main
  method: moving_block_by_month   # contiguous blocks of calendar months
  block_length_months: 12
  reps: 2000
  seed: 2027
  statistics:
    - coef_post_sample
    - coef_post_publication
    - decay_pct_post_sample
    - decay_pct_post_publication
  interval: percentile
  level: 0.95
  exploratory: false

multiple_testing:
  hurdle_t: 3.0                   # Harvey, Liu and Zhu (2016) suggested hurdle
  weak_rule: tstat_below_hurdle   # weak = t-statistic strictly below hurdle_t
  missing_tstat: exclude_and_count
  tstat_sources:
    original_paper: "OSAP SignalDoc column T-Stat (the original paper's reported t-statistic)"
    in_sample_replicated: "mean / (sd / sqrt(n)) of the op long-short return over the
      predictor's in-sample months"
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
      extra_terms: [post_sample_x_tstat, post_publication_x_tstat]   # MP Table II col. 4
      exploratory: false

vintages:
  primary: 202510
  compare_preference: [202410, 202408, 2023]   # first downloadable release passing validation
  predictor_set: intersection   # in both signal docs and op LS rows in both vintages
  specs: [main, full_extension]
  metrics: [coef, se, decay_pct, n_predictors, n_obs]
  exploratory: false
"""

Mutate = Callable[[dict[str, Any]], None]


def _write_prereg(tmp_path: Path, mutate: Mutate | None = None) -> Path:
    """The supplied pre-registration text (optionally mutated) as a YAML file."""
    data: dict[str, Any] = yaml.safe_load(PREREG_TEXT)
    if mutate is not None:
        mutate(data)
    path = tmp_path / "phase2_preregistration.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _phase2_frame(name: str) -> pl.DataFrame:
    """A multi-row frame matching the Phase 2 schema *name*, with descending sort keys."""
    schema = RESULTS_SCHEMAS[name]
    rows: list[dict[str, object]] = []
    for index in (0, 1):
        row: dict[str, object] = {}
        for column, dtype in schema.items():
            if dtype == pl.Utf8:
                row[column] = f"x{index}"
            elif dtype == pl.Float64:
                row[column] = 0.5 + (index + 1) / 3
            else:
                row[column] = index + 1
        for column in RESULTS_SORT_KEYS[name]:
            row[column] = ["b", "a"][index] if schema[column] == pl.Utf8 else [2, 1][index]
        rows.append(row)
    return pl.DataFrame(rows, schema=schema)


@pytest.mark.parametrize("name", sorted(PHASE2_SCHEMAS))
def test_phase2_schema_columns_and_dtypes_are_exact(name: str) -> None:
    assert tuple(RESULTS_SCHEMAS[name].items()) == PHASE2_SCHEMAS[name]


@pytest.mark.parametrize("name", sorted(PHASE2_SORT_KEYS))
def test_phase2_sort_keys_are_exact(name: str) -> None:
    assert RESULTS_SORT_KEYS[name] == PHASE2_SORT_KEYS[name]
    for column in PHASE2_SORT_KEYS[name]:
        assert column in RESULTS_SCHEMAS[name]


@pytest.mark.parametrize("name", sorted(PHASE2_SCHEMAS))
def test_phase2_results_round_trip_through_write_and_read(name: str, tmp_path: Path) -> None:
    frame = _phase2_frame(name)
    path = write_results(frame, name, tmp_path)
    back = read_results(name, tmp_path)
    assert back.schema == frame.schema
    expected = frame.sort(list(PHASE2_SORT_KEYS[name]))
    assert back.equals(expected)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == [column for column, _ in PHASE2_SCHEMAS[name]]


@pytest.mark.parametrize("name", sorted(PHASE2_SCHEMAS))
def test_phase2_results_rewrite_is_byte_identical(name: str, tmp_path: Path) -> None:
    frame = _phase2_frame(name)
    first = write_results(frame, name, tmp_path / "first")
    second = write_results(frame.reverse(), name, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()


def _assert_supplied_values(prereg: Phase2Prereg) -> None:
    assert prereg.version == 1

    bootstrap = prereg.bootstrap
    assert bootstrap.id == "bootstrap_main"
    assert bootstrap.base_spec == "main"
    assert bootstrap.method == "moving_block_by_month"
    assert bootstrap.block_length_months == 12
    assert bootstrap.reps == 2000
    assert bootstrap.seed == 2027
    assert bootstrap.statistics == (
        "coef_post_sample",
        "coef_post_publication",
        "decay_pct_post_sample",
        "decay_pct_post_publication",
    )
    assert bootstrap.interval == "percentile"
    assert bootstrap.level == 0.95
    assert bootstrap.exploratory is False

    multiple_testing = prereg.multiple_testing
    assert multiple_testing.hurdle_t == 3.0
    assert multiple_testing.weak_rule == "tstat_below_hurdle"
    assert multiple_testing.missing_tstat == "exclude_and_count"
    assert multiple_testing.tstat_sources == {
        "original_paper": (
            "OSAP SignalDoc column T-Stat (the original paper's reported t-statistic)"
        ),
        "in_sample_replicated": (
            "mean / (sd / sqrt(n)) of the op long-short return over the predictor's "
            "in-sample months"
        ),
    }
    assert [spec.id for spec in multiple_testing.specs] == [
        "mt_weak_original",
        "mt_weak_replicated",
        "mt_tstat_continuous",
    ]
    assert [spec.tstat_source for spec in multiple_testing.specs] == [
        "original_paper",
        "in_sample_replicated",
        "in_sample_replicated",
    ]
    assert multiple_testing.specs[0].extra_terms == (
        "post_sample_x_weak",
        "post_publication_x_weak",
    )
    assert multiple_testing.specs[2].extra_terms == (
        "post_sample_x_tstat",
        "post_publication_x_tstat",
    )
    assert all(spec.base_spec == "main" for spec in multiple_testing.specs)
    assert all(spec.exploratory is False for spec in multiple_testing.specs)

    vintages = prereg.vintages
    assert vintages.primary == 202510
    assert vintages.compare_preference == (202410, 202408, 2023)
    assert vintages.predictor_set == "intersection"
    assert vintages.specs == ("main", "full_extension")
    assert vintages.metrics == ("coef", "se", "decay_pct", "n_predictors", "n_obs")
    assert vintages.exploratory is False


def test_loader_reads_the_supplied_prereg_text(tmp_path: Path) -> None:
    _assert_supplied_values(load_phase2_preregistration(_write_prereg(tmp_path)))


def test_committed_prereg_file_loads_with_expected_values() -> None:
    if not COMMITTED_PREREG.is_file():
        pytest.skip(COMMITTED_SKIP_REASON)
    _assert_supplied_values(load_phase2_preregistration(COMMITTED_PREREG))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_phase2_preregistration(tmp_path / "nope.yaml")


def test_non_mapping_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "phase2_preregistration.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top level must be a mapping"):
        load_phase2_preregistration(path)


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data.update(extra=1))
    with pytest.raises(ConfigError, match=r"unknown keys \['extra'\]"):
        load_phase2_preregistration(path)


def test_unknown_bootstrap_key_raises(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["bootstrap"].update(blocks=12))
    with pytest.raises(ConfigError, match=r"unknown keys \['blocks'\]"):
        load_phase2_preregistration(path)


def test_unknown_vintages_key_raises(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["vintages"].update(latest=202510))
    with pytest.raises(ConfigError, match=r"unknown keys \['latest'\]"):
        load_phase2_preregistration(path)


def test_unknown_multiple_testing_spec_key_raises(tmp_path: Path) -> None:
    path = _write_prereg(
        tmp_path, lambda data: data["multiple_testing"]["specs"][0].update(note="x")
    )
    with pytest.raises(ConfigError, match=r"unknown keys \['note'\]"):
        load_phase2_preregistration(path)


def test_missing_key_raises(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        del data["bootstrap"]["reps"]

    path = _write_prereg(tmp_path, mutate)
    with pytest.raises(ConfigError, match=r"\[bootstrap\]: missing key 'reps'"):
        load_phase2_preregistration(path)


def test_method_must_be_the_registered_method(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["bootstrap"].update(method="iid"))
    with pytest.raises(ConfigError, match="method must be one of"):
        load_phase2_preregistration(path)


def test_statistics_must_be_registered_values(tmp_path: Path) -> None:
    path = _write_prereg(
        tmp_path,
        lambda data: data["bootstrap"].update(statistics=["coef_post_sample", "sharpe"]),
    )
    with pytest.raises(ConfigError, match="statistics entry 'sharpe' must be one of"):
        load_phase2_preregistration(path)


def test_level_must_be_between_zero_and_one(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["bootstrap"].update(level=1.5))
    with pytest.raises(ConfigError, match="level must be strictly between 0 and 1"):
        load_phase2_preregistration(path)


@pytest.mark.parametrize("key", ["block_length_months", "reps", "seed"])
def test_non_positive_replicate_settings_raise(tmp_path: Path, key: str) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["bootstrap"][key] = -1

    path = _write_prereg(tmp_path, mutate)
    with pytest.raises(ConfigError, match="must be >= 1|must be >= 0"):
        load_phase2_preregistration(path)


def test_non_integer_reps_raise(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["bootstrap"].update(reps="many"))
    with pytest.raises(ConfigError, match="reps must be an integer"):
        load_phase2_preregistration(path)


def test_exploratory_must_be_boolean(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["bootstrap"].update(exploratory="no"))
    with pytest.raises(ConfigError, match="exploratory must be a boolean"):
        load_phase2_preregistration(path)


def test_tstat_source_must_be_declared(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["multiple_testing"]["specs"][0]["tstat_source"] = "winsorized"

    path = _write_prereg(tmp_path, mutate)
    with pytest.raises(ConfigError, match="tstat_source 'winsorized' is not in tstat_sources"):
        load_phase2_preregistration(path)


def test_duplicate_multiple_testing_spec_ids_raise(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        specs = data["multiple_testing"]["specs"]
        specs[1]["id"] = specs[0]["id"]

    path = _write_prereg(tmp_path, mutate)
    with pytest.raises(ConfigError, match="duplicate spec id"):
        load_phase2_preregistration(path)


def test_compare_preference_must_be_integers(tmp_path: Path) -> None:
    path = _write_prereg(
        tmp_path, lambda data: data["vintages"].update(compare_preference=["202410"])
    )
    with pytest.raises(ConfigError, match="compare_preference must contain only integers"):
        load_phase2_preregistration(path)


def test_predictor_set_must_be_the_registered_rule(tmp_path: Path) -> None:
    path = _write_prereg(tmp_path, lambda data: data["vintages"].update(predictor_set="union"))
    with pytest.raises(ConfigError, match="predictor_set must be one of"):
        load_phase2_preregistration(path)
