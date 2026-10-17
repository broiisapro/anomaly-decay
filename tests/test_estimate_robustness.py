"""Tests for the robustness and extension runners (issue #19).

Covers the row builders themselves (``main`` included in robustness.csv, the
extension interaction term, per-portfolio predictor counts) and the CLI's
robustness/extension files, which must equal what the public runners produce.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.config import Spec, load_specifications
from anomaly_decay.estimate.extension import run_extension
from anomaly_decay.estimate.main import main as cli_main
from anomaly_decay.estimate.main import run_spec
from anomaly_decay.estimate.robustness import run_robustness
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

from test_estimate import REPO_ROOT, SPECS_PATH, grid_panel, grid_records, load_spec, make_panel

FIXTURE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"


@pytest.fixture(scope="module")
def specs() -> list[Spec]:
    return load_specifications(SPECS_PATH)


@pytest.fixture(scope="module")
def fixture_panel() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE_PANEL)


def _run_cli(tmp_path: Path, panel: pl.DataFrame) -> Path:
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)
    results_dir = tmp_path / "results"
    code = cli_main(
        [
            "--panel",
            str(panel_path),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(results_dir),
        ]
    )
    assert code == 0
    return results_dir


# The two-way CGM covariance can be non-PSD on the deterministic synthetic panel
# and warn while computing a NaN standard error (see test_estimate.py).
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_run_robustness_includes_main_and_robustness_specs() -> None:
    panel = grid_panel(n_signals=4)
    chosen = [load_spec(spec_id) for spec_id in ("main", "main_twoway", "main_dk", "pubconv_alt")]
    result = run_robustness(panel, chosen)

    validate_frame(result, RESULTS_SCHEMAS["robustness.csv"])
    assert result["spec"].unique(maintain_order=True).to_list() == [
        "main",
        "main_twoway",
        "main_dk",
        "pubconv_alt",
    ]
    assert result.height == 8
    labels = {
        row["spec"]: row["label"]
        for row in result.select("spec", "label").unique().iter_rows(named=True)
    }
    assert labels["main"] == load_spec("main").label
    assert labels["main_dk"] == load_spec("main_dk").label


def test_run_robustness_covers_every_group_member_on_the_fixture(
    fixture_panel: pl.DataFrame, specs: list[Spec]
) -> None:
    result = run_robustness(fixture_panel, specs)

    validate_frame(result, RESULTS_SCHEMAS["robustness.csv"])
    expected = [spec.id for spec in specs if spec.group in ("main", "robustness")]
    assert result["spec"].unique(maintain_order=True).to_list() == expected
    assert result["spec"].n_unique() == 10


def test_run_extension_reports_three_terms_for_the_interaction_spec(
    fixture_panel: pl.DataFrame, specs: list[Spec]
) -> None:
    result = run_extension(fixture_panel, specs)

    validate_frame(result, RESULTS_SCHEMAS["extension.csv"])
    assert result["spec"].unique(maintain_order=True).to_list() == [
        "full_extension",
        "ext_pub_after_2000",
    ]
    extension_terms = result.filter(pl.col("spec") == "ext_pub_after_2000")["term"].to_list()
    assert set(extension_terms) == {
        "post_sample",
        "post_publication",
        "post_publication_x_pub_after_2000",
    }
    assert set(result.filter(pl.col("spec") == "full_extension")["term"].to_list()) == {
        "post_sample",
        "post_publication",
    }


def test_factor_adj_runs_on_the_fixture_panel(fixture_panel: pl.DataFrame) -> None:
    result = run_spec(fixture_panel, load_spec("factor_adj"))

    validate_frame(result, RESULTS_SCHEMAS["main.csv"])
    assert result["term"].to_list() == ["post_sample", "post_publication"]
    assert result["coef"].is_finite().all()
    assert result["se"].is_finite().all()
    assert result["n_predictors"].unique().to_list() == [12]


def test_decile_specs_report_their_own_predictor_counts() -> None:
    records = [dict(record, portfolio="op") for record in grid_records(3)]
    records += [dict(record, portfolio="deciles_ew") for record in grid_records(2)]
    panel = make_panel(records)

    op = run_spec(panel, load_spec("main"))
    ew = run_spec(panel, load_spec("ew_deciles"))
    assert op["n_predictors"].to_list() == [3, 3]
    assert ew["n_predictors"].to_list() == [2, 2]
    assert ew["n_obs"].to_list() == [2 * 48, 2 * 48]


def test_decile_counts_on_the_fixture_match_the_decile_rows(fixture_panel: pl.DataFrame) -> None:
    for spec_id, portfolio in (("ew_deciles", "deciles_ew"), ("vw_deciles", "deciles_vw")):
        result = run_spec(fixture_panel, load_spec(spec_id))
        covered = fixture_panel.filter(
            (pl.col("portfolio") == portfolio)
            & (pl.col("pub_year") <= 2013)
            & (pl.col("yyyymm") <= 201312)
        )["signalname"].n_unique()
        assert result["n_predictors"].unique().to_list() == [covered]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cli_robustness_and_extension_match_the_runners(tmp_path: Path, specs: list[Spec]) -> None:
    panel = grid_panel(n_signals=4)
    results_dir = _run_cli(tmp_path, panel)

    # the CLI estimates each specification once and assembles the rows from
    # those estimates; the public runners must produce exactly the same rows
    statuses = {
        row["spec"]: row["status"]
        for row in read_results("specs_run.csv", results_dir).iter_rows(named=True)
    }
    assert statuses["ew_deciles"].startswith("failed:")
    succeeded = [spec for spec in specs if statuses[spec.id] == "ok"]

    written_robustness = read_results("robustness.csv", results_dir).sort(["spec", "term"])
    expected_robustness = run_robustness(panel, succeeded).sort(["spec", "term"])
    assert written_robustness.equals(expected_robustness)
    assert "main" in written_robustness["spec"].to_list()
    assert "ew_deciles" not in written_robustness["spec"].to_list()

    written_extension = read_results("extension.csv", results_dir).sort(["spec", "term"])
    expected_extension = run_extension(panel, succeeded).sort(["spec", "term"])
    assert written_extension.equals(expected_extension)
    assert written_extension["spec"].unique().to_list() == ["full_extension"]
