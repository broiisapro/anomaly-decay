"""Tests for the committed fictional results fixture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest
from anomaly_decay.config import load_specifications
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
RESULTS_DIR = FIXTURES / "results_fixture"
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"

FIXTURE_FILES = (
    "coverage.csv",
    "decay.csv",
    "event_time.csv",
    "extension.csv",
    "main.csv",
    "predictor_summary.csv",
    "robustness.csv",
    "specs_run.csv",
)
"""The files the committed fictional fixture covers (the eight Phase 1 results files)."""

COVERAGE_ITEMS = {
    "vintage",
    "first_month",
    "last_month",
    "n_predictors_op",
    "n_predictors_deciles_ew",
    "n_predictors_deciles_vw",
    "n_clear",
    "n_published_by_2013",
    "n_predictor_months_op",
}


def _builder() -> ModuleType:
    path = FIXTURES / "make_results_fixture.py"
    spec = importlib.util.spec_from_file_location("make_results_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(name: str) -> pl.DataFrame:
    return read_results(name, RESULTS_DIR)


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_file_validates_against_results_schema(name: str) -> None:
    path = RESULTS_DIR / name
    assert path.is_file()
    schema = RESULTS_SCHEMAS[name]
    frame = _read(name)
    assert frame.columns == list(schema)
    validate_frame(frame, schema)

    raw = path.read_bytes()
    assert b"FICTIONAL TEST DATA" not in raw
    assert raw.split(b"\n", 1)[0].decode("utf-8") == ",".join(schema)


def test_committed_fixture_matches_deterministic_rebuild(tmp_path: Path) -> None:
    builder = _builder()
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert builder.main(["--out-dir", str(first)]) == 0
    assert builder.main(["--out-dir", str(second)]) == 0
    for name in FIXTURE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (RESULTS_DIR / name).read_bytes() == (first / name).read_bytes()


def test_every_specification_appears_in_specs_run() -> None:
    specs = load_specifications(SPECS_PATH)
    frame = _read("specs_run.csv")
    assert sorted(frame["spec"].to_list()) == sorted(spec.id for spec in specs)
    assert set(frame["status"].to_list()) == {"ok"}
    by_id = {spec.id: spec for spec in specs}
    for row in frame.iter_rows(named=True):
        assert row["label"] == by_id[row["spec"]].label
        assert row["exploratory"] == by_id[row["spec"]].exploratory


def test_main_file_has_both_terms() -> None:
    frame = _read("main.csv")
    assert set(frame["spec"].to_list()) == {"main"}
    assert set(frame["term"].to_list()) == {"post_sample", "post_publication"}


def test_robustness_file_covers_main_and_every_robustness_spec() -> None:
    specs = load_specifications(SPECS_PATH)
    extension_ids = {spec.id for spec in specs if spec.group == "extension"}
    expected = {"main", *(spec.id for spec in specs if spec.id not in extension_ids | {"main"})}
    frame = _read("robustness.csv")
    assert set(frame["spec"].to_list()) == expected
    assert set(frame["term"].to_list()) == {"post_sample", "post_publication"}
    labels = dict(zip(frame["spec"].to_list(), frame["label"].to_list(), strict=True))
    by_id = {spec.id: spec for spec in specs}
    for spec_id, label in labels.items():
        assert label == by_id[spec_id].label


def test_extension_file_has_three_terms_for_the_interaction_spec() -> None:
    specs = load_specifications(SPECS_PATH)
    expected = {spec.id for spec in specs if spec.group == "extension"}
    frame = _read("extension.csv")
    assert set(frame["spec"].to_list()) == expected
    terms = {
        spec_id: set(frame.filter(pl.col("spec") == spec_id)["term"].to_list())
        for spec_id in expected
    }
    assert terms["full_extension"] == {"post_sample", "post_publication"}
    assert terms["ext_pub_after_2000"] == {
        "post_sample",
        "post_publication",
        "post_publication_x_pub_after_2000",
    }


def test_decay_file_has_both_methods_and_matching_arithmetic() -> None:
    decay = _read("decay.csv")
    main = _read("main.csv")
    assert set(decay["method"].to_list()) == {"pooled_ratio", "normalized"}
    assert set(decay["term"].to_list()) == {"post_sample", "post_publication"}
    assert set(decay.filter(pl.col("method") == "pooled_ratio")["spec"].to_list()) == {"main"}
    assert set(decay.filter(pl.col("method") == "normalized")["spec"].to_list()) == {"normalized"}

    pooled = decay.filter(pl.col("method") == "pooled_ratio")
    for row in pooled.iter_rows(named=True):
        coef = main.filter(pl.col("term") == row["term"])["coef"].item()
        expected = 100.0 * coef / row["in_sample_mean"]
        assert row["decay_pct"] == pytest.approx(expected, abs=1e-9)
        assert row["decay_pct"] < 0
    assert all(
        value == pytest.approx(14.0)
        for value in pooled.filter(pl.col("term") == "post_sample")["se_pct"].to_list()
    )

    normalized = _read("robustness.csv").filter(pl.col("spec") == "normalized")
    norm_decay = decay.filter(pl.col("method") == "normalized")
    for row in norm_decay.iter_rows(named=True):
        coef = normalized.filter(pl.col("term") == row["term"])["coef"].item()
        se = normalized.filter(pl.col("term") == row["term"])["se"].item()
        assert row["decay_pct"] == pytest.approx(100.0 * coef, abs=1e-9)
        assert row["se_pct"] == pytest.approx(100.0 * se, abs=1e-9)


def test_predictor_summary_has_fifteen_predictors_in_three_windows() -> None:
    frame = _read("predictor_summary.csv")
    assert frame["signalname"].n_unique() == 15
    assert frame.height == 45
    assert set(frame["window"].to_list()) == {"in_sample", "post_sample", "post_publication"}
    assert frame.select("signalname", "window").n_unique() == 45
    assert (frame["n_months"] > 0).all()


def test_event_time_covers_both_anchors_and_full_month_range() -> None:
    frame = _read("event_time.csv")
    assert set(frame["anchor"].to_list()) == {"sample_end", "publication"}
    expected_months = list(range(-60, 121))
    for anchor in ("sample_end", "publication"):
        anchor_frame = frame.filter(pl.col("anchor") == anchor)
        assert anchor_frame["event_month"].to_list() == expected_months
        assert anchor_frame.height == 181
        assert (anchor_frame["n_predictors"] > 0).all()


def test_coverage_has_every_documented_item() -> None:
    frame = _read("coverage.csv")
    assert set(frame["item"].to_list()) == COVERAGE_ITEMS
    assert frame.height == len(COVERAGE_ITEMS)
    assert frame.filter(pl.col("item") == "vintage")["value"].item() == "202510"


def test_coefficients_are_obviously_fictional_multiples() -> None:
    unit = _builder().FICTIONAL_UNIT
    for name in ("main.csv", "robustness.csv", "extension.csv"):
        frame = _read(name)
        for row in frame.iter_rows(named=True):
            multiple = row["coef"] / unit
            assert multiple == pytest.approx(round(multiple), abs=1e-9), (name, row)
            assert row["t"] == pytest.approx(row["coef"] / row["se"], abs=1e-9)

    summary = _read("predictor_summary.csv")
    for value in summary["mean_ret"].to_list():
        multiple = value / unit
        assert multiple == pytest.approx(round(multiple), abs=1e-9)
