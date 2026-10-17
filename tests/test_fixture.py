"""Tests for the committed fixture: manifest, size, predictor list, panel."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from anomaly_decay.data.fetch import verify_manifest

REPO_ROOT = Path(__file__).parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FIXTURE_PREDICTORS = {
    "Mom12m",
    "Size",
    "BM",
    "Accruals",
    "AM",
    "AssetGrowth",
    "GP",
    "IdioVol3F",
    "STreversal",
    "EP",
    "Beta",
    "BetaFP",
    "TrendFactor",
    "dVolPut",
    "MaxRet",
}
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def test_fixture_manifest_verifies() -> None:
    result = verify_manifest(FIXTURES / "manifest.lock", REPO_ROOT)
    assert result.checked > 0
    assert result.mismatches == ()
    assert result.missing == ()
    assert result.not_cached == ()
    assert result.ok


def test_fixture_manifest_covers_only_fixture_data() -> None:
    lines = (FIXTURES / "manifest.lock").read_text(encoding="utf-8").splitlines()
    paths = [line.split("  ")[1] for line in lines]
    assert paths
    assert all(path.startswith("tests/fixtures/data/") for path in paths)
    assert "tests/fixtures/panel_fixture.parquet" not in paths


def test_fixture_size_under_ten_megabytes() -> None:
    assert _dir_size(FIXTURES) < MAX_FIXTURE_BYTES


def test_fixture_predictor_list_matches() -> None:
    doc = pl.read_csv(FIXTURES / "data" / "osap" / "202510" / "signal_doc.csv")
    assert set(doc["Acronym"].to_list()) == FIXTURE_PREDICTORS
    assert doc.height == 15
    for name in ("op", "deciles_ew", "deciles_vw"):
        ports = pl.read_csv(FIXTURES / "data" / "osap" / "202510" / f"port_{name}.csv")
        assert set(ports["signalname"].to_list()) == FIXTURE_PREDICTORS
        assert set(ports["port"].unique().to_list()) == {"LS"}


def test_fixture_source_file_flags_synthetic_or_osap() -> None:
    first_line = (
        (FIXTURES / "data" / "FIXTURE_SOURCE.txt").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first_line in {"synthetic", "osap-202510"}


def test_panel_fixture_exists_and_covers_all_portfolios() -> None:
    panel = pl.read_parquet(FIXTURES / "panel_fixture.parquet")
    assert panel.height > 0
    assert set(panel["portfolio"].unique().to_list()) == {"op", "deciles_ew", "deciles_vw"}
    assert set(panel["signalname"].unique().to_list()) == FIXTURE_PREDICTORS


def test_panel_fixture_keeps_null_ret_rows_out_and_marks_negative_mean_nulls() -> None:
    panel = pl.read_parquet(FIXTURES / "panel_fixture.parquet")
    assert panel["ret"].null_count() == 0
    negative = panel.filter(pl.col("is_mean") <= 0)
    assert negative.height > 0
    assert negative["ret_norm"].null_count() == negative.height
    positive = panel.filter(pl.col("is_mean") > 0)
    assert positive["ret_norm"].null_count() == 0
