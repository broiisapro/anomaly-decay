"""Tests for the panel build CLI (``python -m anomaly_decay.panel.build``)."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.panel import build as panel_build
from anomaly_decay.panel.build import build_panel, main
from anomaly_decay.schemas import PREDICTORS_SCHEMA

REPO_ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_OSAP = FIXTURES / "data" / "osap"
FIXTURE_PINNED = FIXTURES / "data" / "pinned"
VINTAGE = "202510"
ALL_PORTFOLIOS = "op,deciles_ew,deciles_vw"


def _write_predictors_parquet(tmp_path: Path) -> Path:
    """Write the fixture metadata per PREDICTORS_SCHEMA (until #11 provides the CLI)."""
    doc = pl.read_csv(FIXTURE_OSAP / VINTAGE / "signal_doc.csv", infer_schema_length=500).filter(
        pl.col("Cat.Signal") == "Predictor"
    )
    height = doc.height
    frame = pl.DataFrame(
        {
            "signalname": doc["Acronym"],
            "sample_start_year": doc["SampleStartYear"].cast(pl.Int16),
            "sample_end_year": doc["SampleEndYear"].cast(pl.Int16),
            "pub_year": doc["Year"].cast(pl.Int16),
            "journal": doc["Journal"],
            "authors": doc["Authors"],
            "cat_economic": doc["Cat.Economic"],
            "cat_data": doc["Cat.Data"],
            "clear": doc["Predictability in OP"] == "1_clear",
            "date_source": ["signaldoc"] * height,
        }
    ).select(list(PREDICTORS_SCHEMA))
    path = tmp_path / "predictors.parquet"
    frame.write_parquet(path)
    return path


def _base_args(metadata_path: Path, out_path: Path) -> list[str]:
    return [
        "--data-dir",
        str(FIXTURE_OSAP),
        "--pinned-dir",
        str(FIXTURE_PINNED),
        "--vintage",
        VINTAGE,
        "--metadata",
        str(metadata_path),
        "--portfolios",
        ALL_PORTFOLIOS,
        "--out",
        str(out_path),
    ]


def _counts(stdout: str, label: str) -> dict[str, int]:
    line = next(line for line in stdout.splitlines() if line.startswith(f"{label}:"))
    return {key: int(value) for key, value in re.findall(r"([\w]+)=(\d+)", line)}


def test_cli_writes_byte_identical_parquet(tmp_path: Path) -> None:
    metadata_path = _write_predictors_parquet(tmp_path)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    assert main(_base_args(metadata_path, first)) == 0
    assert main(_base_args(metadata_path, second)) == 0
    assert first.read_bytes() == second.read_bytes()


def test_cli_dry_run_prints_counts_only_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = _write_predictors_parquet(tmp_path)
    out = tmp_path / "panel.parquet"
    assert main([*_base_args(metadata_path, out), "--dry-run"]) == 0
    assert not out.exists()

    stdout = capsys.readouterr().out
    assert "dry-run: counts only, nothing written" in stdout
    for label in (
        "raw LS rows",
        "kept rows",
        "rows per portfolio",
        "predictors per portfolio",
        "rows per window",
        "dropped rows without metadata",
        "dropped predictor-portfolios without in-sample returns",
        "dropped pre-sample rows",
        "dropped rows with null ret",
        "predictors without metadata",
    ):
        assert f"{label}:" in stdout
    for forbidden in ("is_mean", "ret_norm", "median", "decay", "coef", "mean_ret"):
        assert forbidden not in stdout


def test_cli_dry_run_counts_match_full_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = _write_predictors_parquet(tmp_path)
    out = tmp_path / "panel.parquet"
    assert main([*_base_args(metadata_path, out), "--dry-run"]) == 0
    stdout = capsys.readouterr().out

    panel = build_panel(
        FIXTURE_OSAP,
        pl.read_parquet(metadata_path),
        tuple(ALL_PORTFOLIOS.split(",")),
        pinned_dir=FIXTURE_PINNED,
    )
    kept = re.search(r"kept rows: (\d+)", stdout)
    assert kept is not None and int(kept.group(1)) == panel.height
    assert _counts(stdout, "rows per portfolio") == dict(
        panel.group_by("portfolio").agg(pl.len().alias("n")).sort("portfolio").iter_rows()
    )
    assert _counts(stdout, "predictors per portfolio") == dict(
        panel.group_by("portfolio")
        .agg(pl.col("signalname").n_unique().alias("n"))
        .sort("portfolio")
        .iter_rows()
    )
    assert _counts(stdout, "rows per window") == dict(
        panel.group_by("window").agg(pl.len().alias("n")).sort("window").iter_rows()
    )


def test_cli_default_pinned_dir_is_data_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    metadata_path = _write_predictors_parquet(tmp_path)
    out = tmp_path / "panel.parquet"
    args = [
        "--data-dir",
        str(FIXTURE_OSAP),
        "--vintage",
        VINTAGE,
        "--metadata",
        str(metadata_path),
        "--portfolios",
        "op",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    assert out.is_file()


def test_panel_docstrings_state_units_and_window_rules() -> None:
    module_doc = panel_build.__doc__ or ""
    build_doc = panel_build.build_panel.__doc__ or ""
    for text in (module_doc, build_doc):
        assert "percent per month" in text
    for window in ("in_sample", "post_sample", "post_publication"):
        assert window in module_doc
    assert "SampleStartYear" in module_doc and "SampleEndYear" in module_doc
