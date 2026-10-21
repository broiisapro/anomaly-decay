"""Tests for the estimation CLI (issue #18).

The CLI is exercised in-process; the fixtures are the deterministic panels from
``test_estimate``. Covered: schema-valid outputs, the headline format, failure
isolation across specifications, and the ``--only`` restriction.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate import main as main_module
from anomaly_decay.estimate.main import build_parser
from anomaly_decay.estimate.main import main as cli_main
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

from test_estimate import SPECS_PATH, decay_panel, grid_panel


def run_cli(
    tmp_path: Path, panel: pl.DataFrame, *extra: str, specs: Path = SPECS_PATH
) -> tuple[int, Path]:
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)
    results_dir = tmp_path / "results"
    argv = [
        "--panel",
        str(panel_path),
        "--specs",
        str(specs),
        "--results-dir",
        str(results_dir),
        *extra,
    ]
    return cli_main(argv), results_dir


def spec_statuses(results_dir: Path) -> dict[str, str]:
    status = read_results("specs_run.csv", results_dir)
    validate_frame(status, RESULTS_SCHEMAS["specs_run.csv"])
    return {row["spec"]: row["status"] for row in status.iter_rows(named=True)}


# The two-way CGM covariance can be non-PSD on the deterministic synthetic panel
# and warn while computing a NaN standard error (see test_estimate.py).
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cli_runs_every_spec_and_writes_schema_valid_files(tmp_path: Path) -> None:
    code, results_dir = run_cli(tmp_path, grid_panel(n_signals=4))
    assert code == 0

    main = read_results("main.csv", results_dir)
    validate_frame(main, RESULTS_SCHEMAS["main.csv"])
    assert main["spec"].unique().to_list() == ["main"]
    assert main["term"].to_list() == ["post_publication", "post_sample"]
    assert main["n_predictors"].unique().to_list() == [4]

    decay = read_results("decay.csv", results_dir)
    validate_frame(decay, RESULTS_SCHEMAS["decay.csv"])
    assert set(decay["method"].to_list()) == {"pooled_ratio", "normalized"}

    statuses = spec_statuses(results_dir)
    assert len(statuses) == 12
    assert statuses["main"] == "ok"
    assert statuses["main_twoway"] == "ok"
    assert statuses["main_dk"] == "ok"
    assert statuses["normalized"] == "ok"
    assert statuses["full_extension"] == "ok"
    assert statuses["factor_adj"].startswith("failed:")
    assert statuses["ew_deciles"].startswith("failed:")
    assert statuses["vw_deciles"].startswith("failed:")
    assert statuses["ext_pub_after_2000"].startswith("failed:")


def test_cli_print_headline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, results_dir = run_cli(tmp_path, decay_panel(), "--print-headline")
    assert code == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert re.fullmatch(
        r"post-sample: -30\.0% \(SE \d+\.\d%\), N predictors = 4, N obs = 192", lines[0]
    )
    assert re.fullmatch(
        r"post-publication: -60\.0% \(SE \d+\.\d%\), N predictors = 4, N obs = 192", lines[1]
    )

    pooled = read_results("decay.csv", results_dir).filter(pl.col("method") == "pooled_ratio")
    assert pooled.height == 2
    assert float(pooled["in_sample_mean"].unique().item()) == pytest.approx(1.6, abs=1e-12)


def test_cli_headline_unavailable_returns_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run_cli(tmp_path, decay_panel(), "--only", "normalized", "--print-headline")
    assert code == 1
    assert "headline unavailable" in capsys.readouterr().err


def test_cli_records_failures_and_keeps_going(tmp_path: Path) -> None:
    specs = tmp_path / "specifications.yaml"
    specs.write_text(
        """
version: 1
defaults:
  group: robustness
  portfolio: op
  dep_var: ret
  convention: default
  cov: clustered_time
  cov_params: {}
  universe: {pub_year_max: null, sample_start_year_min: null, clear_only: false}
  months: {start_yyyymm: null, end_yyyymm: null}
  extra_terms: []
  exploratory: false
specifications:
  - id: main
    label: "Main"
    group: main
  - id: bogus
    label: "Bogus extra term"
    extra_terms: [does_not_exist]
  - id: normalized
    label: "Normalized"
    dep_var: ret_norm
""",
        encoding="utf-8",
    )
    code, results_dir = run_cli(tmp_path, grid_panel(n_signals=4), specs=specs)
    assert code == 0

    statuses = spec_statuses(results_dir)
    assert statuses["main"] == "ok"
    assert statuses["normalized"] == "ok"
    assert statuses["bogus"].startswith("failed:")
    assert "unknown extra term" in statuses["bogus"]
    assert (results_dir / "main.csv").is_file()
    assert (results_dir / "decay.csv").is_file()


def test_cli_only_restricts_the_run(tmp_path: Path) -> None:
    code, results_dir = run_cli(tmp_path, grid_panel(n_signals=4), "--only", "main")
    assert code == 0

    statuses = spec_statuses(results_dir)
    assert statuses["main"] == "ok"
    assert statuses["main_twoway"] == "skipped"
    assert statuses["normalized"] == "skipped"

    decay = read_results("decay.csv", results_dir)
    assert decay["method"].unique().to_list() == ["pooled_ratio"]
    assert decay["spec"].unique().to_list() == ["main"]


def test_cli_rejects_unknown_only_id(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    grid_panel(n_signals=4).write_parquet(panel_path)
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "--panel",
                str(panel_path),
                "--specs",
                str(SPECS_PATH),
                "--results-dir",
                str(tmp_path / "results"),
                "--only",
                "does_not_exist",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_path_defaults_point_at_the_repository_files() -> None:
    args = build_parser().parse_args([])
    assert (
        args.panel
        == main_module.DEFAULT_PANEL
        == main_module.REPO_ROOT / "data/processed/panel.parquet"
    )
    assert (
        args.specs
        == main_module.DEFAULT_SPECS
        == main_module.REPO_ROOT / "config/specifications.yaml"
    )
    assert args.results_dir == main_module.DEFAULT_RESULTS_DIR == main_module.REPO_ROOT / "results"


def test_cli_runs_with_no_path_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The PRD command works as written: all paths come from the CLI defaults."""
    panel_path = tmp_path / "panel.parquet"
    grid_panel(n_signals=4).write_parquet(panel_path)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(main_module, "DEFAULT_PANEL", panel_path)
    monkeypatch.setattr(main_module, "DEFAULT_SPECS", SPECS_PATH)
    monkeypatch.setattr(main_module, "DEFAULT_RESULTS_DIR", results_dir)

    assert cli_main(["--only", "main"]) == 0
    assert (results_dir / "main.csv").is_file()
    assert read_results("main.csv", results_dir)["n_predictors"].unique().to_list() == [4]
