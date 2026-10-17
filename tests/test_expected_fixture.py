"""The committed expected main.csv must equal what the CLI computes on the fixture.

``tests/fixtures/expected_main_fixture.csv`` is produced by the estimation CLI
(``python -m anomaly_decay.estimate.main --only main``) from
``tests/fixtures/panel_fixture.parquet``; this module recomputes it and compares
every cell. A second, independent estimator — statsmodels OLS with predictor
dummies — cross-checks the two coefficients.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.api as sm
from anomaly_decay.estimate.main import main as cli_main
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"
EXPECTED_MAIN = REPO_ROOT / "tests" / "fixtures" / "expected_main_fixture.csv"
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"

TOLERANCE = 1e-10
FLOAT_COLUMNS = ("coef", "se", "t")


@pytest.fixture(scope="module")
def expected_main() -> pl.DataFrame:
    frame = pl.read_csv(EXPECTED_MAIN, schema_overrides={"spec": pl.Utf8, "term": pl.Utf8})
    return frame.cast(RESULTS_SCHEMAS["main.csv"])


def test_expected_main_matches_cli_output(tmp_path: Path, expected_main: pl.DataFrame) -> None:
    panel_path = tmp_path / "panel.parquet"
    pl.read_parquet(FIXTURE_PANEL).write_parquet(panel_path)
    results_dir = tmp_path / "results"
    code = cli_main(
        [
            "--panel",
            str(panel_path),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(results_dir),
            "--only",
            "main",
        ]
    )
    assert code == 0
    actual = read_results("main.csv", results_dir)

    assert actual.columns == expected_main.columns
    assert actual.height == expected_main.height
    for column in actual.columns:
        if column in FLOAT_COLUMNS:
            np.testing.assert_allclose(
                actual[column].to_numpy(),
                expected_main[column].to_numpy(),
                rtol=0,
                atol=TOLERANCE,
            )
        else:
            assert actual[column].equals(expected_main[column]), column


def test_main_coefficients_match_statsmodels_with_predictor_dummies(
    expected_main: pl.DataFrame,
) -> None:
    panel = pl.read_parquet(FIXTURE_PANEL)
    sample = panel.filter(
        (pl.col("portfolio") == "op")
        & (pl.col("pub_year") <= 2013)
        & (pl.col("yyyymm") <= 201312)
        & pl.col("ret").is_not_null()
    )
    frame = sample.select("signalname", "ret", "post_sample", "post_publication").to_pandas()
    # One dummy per predictor and no constant, mirroring PanelOLS entity effects
    # without an explicit constant; dropping one dummy and fitting no constant
    # instead would impose a zero intercept for that predictor and change the
    # slopes on an unbalanced panel.
    dummies = pd.get_dummies(frame["signalname"]).astype(float)
    design = pd.concat([frame[["post_sample", "post_publication"]], dummies], axis=1).astype(float)
    results = sm.OLS(frame["ret"].astype(float), design).fit()

    coefficients = dict(zip(design.columns, results.params, strict=True))
    committed = {row["term"]: row["coef"] for row in expected_main.iter_rows(named=True)}
    np.testing.assert_allclose(
        coefficients["post_sample"], committed["post_sample"], rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        coefficients["post_publication"], committed["post_publication"], rtol=0, atol=1e-8
    )


def test_expected_main_is_the_fixture_main_spec(expected_main: pl.DataFrame) -> None:
    assert expected_main["spec"].unique().to_list() == ["main"]
    assert expected_main["term"].to_list() == ["post_publication", "post_sample"]
