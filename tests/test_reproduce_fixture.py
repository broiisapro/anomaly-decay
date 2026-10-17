"""Fixture-gate expectations: the built panel and results must match what is committed.

``make reproduce-fixture`` runs the production pipeline on the committed fixture inputs
into ``data/processed/fixture/`` and then executes this module. In a plain test run those
outputs do not exist and the two end-to-end checks skip; the comparison helper itself is
covered by hand-built frames either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from anomaly_decay.schemas import RESULTS_SCHEMAS
from polars.testing import assert_frame_equal

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_PROCESSED = REPO_ROOT / "data" / "processed" / "fixture"
FIXTURE_PANEL = FIXTURE_PROCESSED / "panel.parquet"
FIXTURE_MAIN = FIXTURE_PROCESSED / "results" / "main.csv"
REFERENCE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"
EXPECTED_MAIN = REPO_ROOT / "tests" / "fixtures" / "expected_main_fixture.csv"
TOLERANCE = 1e-10
FLOAT_COLUMNS = ("coef", "se", "t")


def assert_main_matches_expected(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    """Compare two ``main.csv`` frames: float columns to 1e-10, others exactly."""
    assert actual.columns == expected.columns, (actual.columns, expected.columns)
    assert actual.height == expected.height, (actual.height, expected.height)
    for column in actual.columns:
        if column in FLOAT_COLUMNS:
            np.testing.assert_allclose(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
                rtol=0,
                atol=TOLERANCE,
            )
        else:
            assert actual[column].equals(expected[column]), column


def _main_frame(coef: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "spec": ["main"],
            "term": ["post_sample"],
            "coef": [coef],
            "se": [0.05],
            "t": [2.0],
            "n_predictors": [12],
            "n_obs": [100],
        },
        schema=RESULTS_SCHEMAS["main.csv"],
    )


def test_main_comparison_accepts_a_change_below_the_tolerance() -> None:
    base = _main_frame(0.1)
    assert_main_matches_expected(base.with_columns(pl.col("coef") + 1e-11), base)


def test_main_comparison_rejects_a_change_above_the_tolerance() -> None:
    base = _main_frame(0.1)
    with pytest.raises(AssertionError):
        assert_main_matches_expected(base.with_columns(pl.col("coef") + 1e-8), base)


def test_main_comparison_rejects_a_changed_integer() -> None:
    base = _main_frame(0.1)
    with pytest.raises(AssertionError):
        assert_main_matches_expected(base.with_columns(pl.col("n_obs") + 1), base)


@pytest.mark.skipif(not FIXTURE_PANEL.is_file(), reason="run `make reproduce-fixture` first")
def test_built_fixture_panel_equals_the_reference_panel_exactly() -> None:
    built = pl.read_parquet(FIXTURE_PANEL)
    reference = pl.read_parquet(REFERENCE_PANEL)
    assert_frame_equal(built, reference, check_exact=True)


@pytest.mark.skipif(not FIXTURE_MAIN.is_file(), reason="run `make reproduce-fixture` first")
def test_fixture_main_matches_the_expected_file_to_1e_10() -> None:
    actual = pl.read_csv(FIXTURE_MAIN, schema=RESULTS_SCHEMAS["main.csv"])
    expected = pl.read_csv(EXPECTED_MAIN, schema=RESULTS_SCHEMAS["main.csv"])
    assert_main_matches_expected(actual, expected)
