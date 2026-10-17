"""Running the estimation CLI twice must produce byte-identical results files."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.estimate.main import main as cli_main

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"
RESULTS_FILES = ("main.csv", "decay.csv", "robustness.csv", "extension.csv", "specs_run.csv")


@pytest.fixture(scope="module")
def run_dirs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Two full CLI runs on the fixture panel, each into its own directory."""
    dirs: list[Path] = []
    panel_path = tmp_path_factory.mktemp("source") / "panel.parquet"
    pl.read_parquet(FIXTURE_PANEL).write_parquet(panel_path)
    for run in range(2):
        run_dir = tmp_path_factory.mktemp(f"results-{run}")
        code = cli_main(
            [
                "--panel",
                str(panel_path),
                "--specs",
                str(SPECS_PATH),
                "--results-dir",
                str(run_dir),
            ]
        )
        assert code == 0
        dirs.append(run_dir)
    return dirs


def test_cli_outputs_are_byte_identical_across_runs(run_dirs: list[Path]) -> None:
    for name in RESULTS_FILES:
        first = (run_dirs[0] / name).read_bytes()
        second = (run_dirs[1] / name).read_bytes()
        assert first == second, f"{name} differs between runs"
