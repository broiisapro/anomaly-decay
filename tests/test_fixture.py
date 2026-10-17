"""Tests for the committed fixture: manifest, size, predictor list, panel.

The fixture is a real subset of the OSAP 202510 canonical cache (LS rows for the 15
pre-registered predictors, all months kept). Edge cases the retired synthetic fixture
injected on purpose (null returns, a predictor-portfolio with a non-positive in-sample
mean) are covered here with hand-built frames run through the independent reference
builder in ``tests/fixtures/build_fixture_panel.py``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import polars as pl
from anomaly_decay.data.fetch import sha256_file, verify_manifest
from anomaly_decay.io import write_csv_canonical

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
PORT_FILES = ("op", "deciles_ew", "deciles_vw")
PORT_COLUMNS = ("signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort")
SIGNAL_DOC_COLUMNS = (
    "Acronym",
    "Predictability in OP",
    "SampleStartYear",
    "SampleEndYear",
    "Year",
)


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _read_port(name: str) -> pl.DataFrame:
    return pl.read_csv(
        FIXTURES / "data" / "osap" / "202510" / f"port_{name}.csv",
        schema_overrides={"port": pl.Utf8, "date": pl.Utf8},
    )


def _load_reference_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_fixture_panel", FIXTURES / "build_fixture_panel.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    for name in PORT_FILES:
        ports = _read_port(name)
        # Every predictor has long-short rows in all three portfolios in the 202510
        # vintage; a deciles file may legitimately drop a predictor in other vintages
        # (discrete signals carry no deciles), so this is pinned per vintage, not padded.
        assert set(ports["signalname"].to_list()) == FIXTURE_PREDICTORS, name
        assert set(ports["port"].unique().to_list()) == {"LS"}


def test_fixture_keeps_months_outside_the_sample_period() -> None:
    # The real op series start well before SampleStartYear; the fixture keeps them all
    # because the production panel is what drops pre-sample months.
    mom = _read_port("op").filter(pl.col("signalname") == "Mom12m")
    assert mom["date"].min() == "1927-01-31"
    assert mom.height > 1000


def test_fixture_signal_doc_covers_predictors_with_metadata() -> None:
    doc = pl.read_csv(FIXTURES / "data" / "osap" / "202510" / "signal_doc.csv")
    assert doc.columns == list(
        (
            "Acronym",
            "Cat.Signal",
            "Predictability in OP",
            "Authors",
            "Year",
            "Journal",
            "Cat.Economic",
            "Cat.Data",
            "SampleStartYear",
            "SampleEndYear",
        )
    )
    assert set(doc["Cat.Signal"].unique().to_list()) == {"Predictor"}
    assert doc["SampleStartYear"].min() is not None
    assert doc["SampleEndYear"].max() is not None


def test_fixture_source_records_osap_vintage_and_source_hashes() -> None:
    lines = (FIXTURES / "data" / "FIXTURE_SOURCE.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "osap-202510"
    assert "predictors: 15" in lines
    sources = [line.split() for line in lines if line.startswith("source-sha256:")]
    assert len(sources) == 6
    for _, relative_path, digest in sources:
        assert re.fullmatch(r"[0-9a-f]{64}", digest), relative_path
        assert relative_path.startswith(("data/raw/202510/", "data/pinned/"))
        source = REPO_ROOT / relative_path
        if source.is_file():  # the gitignored raw cache is absent in CI
            assert sha256_file(source) == digest


def test_panel_fixture_exists_and_covers_all_portfolios() -> None:
    panel = pl.read_parquet(FIXTURES / "panel_fixture.parquet")
    assert panel.height > 0
    assert set(panel["portfolio"].unique().to_list()) == {"op", "deciles_ew", "deciles_vw"}
    assert set(panel["signalname"].unique().to_list()) == FIXTURE_PREDICTORS


def test_panel_fixture_has_no_null_rets_and_nulls_normalized_returns_for_negative_means() -> None:
    panel = pl.read_parquet(FIXTURES / "panel_fixture.parquet")
    assert panel["ret"].null_count() == 0
    negative = panel.filter(pl.col("is_mean") <= 0)
    # Real data quirk: BetaFP has a non-positive in-sample long-short mean in 202510.
    assert set(negative["signalname"].unique().to_list()) == {"BetaFP"}
    assert negative["ret_norm"].null_count() == negative.height
    positive = panel.filter(pl.col("is_mean") > 0)
    assert positive["ret_norm"].null_count() == 0


def test_reference_builder_drops_null_rets_and_nulls_normalized_returns(tmp_path: Path) -> None:
    # Hand-built frames replace the retired synthetic fixture's deliberate edge cases;
    # the real 202510 cache has no null LS returns, so the drop rule needs its own input.
    fixtures = tmp_path / "fixtures"
    (fixtures / "data" / "osap" / "202510").mkdir(parents=True)
    (fixtures / "data" / "pinned").mkdir(parents=True)
    doc = pl.DataFrame(
        {
            "Acronym": ["NullRet", "Negative"],
            "Predictability in OP": ["1_clear", "1_clear"],
            "SampleStartYear": [1990, 1990],
            "SampleEndYear": [1992, 1992],
            "Year": [1993, 1993],
        }
    )
    write_csv_canonical(doc, fixtures / "data" / "osap" / "202510" / "signal_doc.csv", ["Acronym"])
    sample_months = [y * 100 + m for y in (1990, 1991, 1992) for m in range(1, 13)]
    months = [198912, *sample_months, 199301, 199302]
    dates = [f"{value // 100:04d}-{value % 100:02d}-01" for value in months]
    null_months = {199103, 199106}
    op = pl.DataFrame(
        {
            "signalname": ["NullRet"] * len(months) + ["Negative"] * len(months),
            "port": ["LS"] * (2 * len(months)),
            "date": dates * 2,
            "ret": [None if value in null_months else 1.0 for value in months]
            + [-1.0] * len(months),
            "signallag": [None] * (2 * len(months)),
            "Nlong": [100] * (2 * len(months)),
            "Nshort": [100] * (2 * len(months)),
        }
    )
    write_csv_canonical(
        op,
        fixtures / "data" / "osap" / "202510" / "port_op.csv",
        ["signalname", "port", "date"],
    )
    for name in ("deciles_ew", "deciles_vw"):
        empty = pl.DataFrame(
            {
                "signalname": pl.Series([], dtype=pl.Utf8),
                "port": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Utf8),
                "ret": pl.Series([], dtype=pl.Float64),
                "signallag": pl.Series([], dtype=pl.Int8),
                "Nlong": pl.Series([], dtype=pl.Int64),
                "Nshort": pl.Series([], dtype=pl.Int64),
            }
        )
        write_csv_canonical(
            empty,
            fixtures / "data" / "osap" / "202510" / f"port_{name}.csv",
            ["signalname", "port", "date"],
        )
    write_csv_canonical(
        pl.DataFrame({"yyyymm": [199301], "mkt_rf": [0.5]}),
        fixtures / "data" / "pinned" / "ff5_monthly.csv",
        ["yyyymm"],
    )
    write_csv_canonical(
        pl.DataFrame({"yyyymm": [199301], "umd": [-0.5]}),
        fixtures / "data" / "pinned" / "umd_monthly.csv",
        ["yyyymm"],
    )

    builder = _load_reference_builder()
    panel = builder.build_fixture_panel(fixtures)
    assert not (panel["portfolio"] != "op").any()  # empty deciles files drop out

    null_ret = panel.filter(pl.col("signalname") == "NullRet")
    # Publication year 1993 => post_publication starts 1994-01; the two 1993 months
    # are post_sample, and the 1989-12 pre-sample month is dropped by the builder.
    assert set(null_ret["window"].unique().to_list()) == {"in_sample", "post_sample"}
    assert null_ret.filter(pl.col("window") == "in_sample").height == 34  # 36 - 2 nulls
    assert "1989-12" not in null_ret["month"].dt.strftime("%Y-%m").to_list()
    assert null_ret["is_mean"].unique().to_list() == [1.0]
    assert null_ret["ret_norm"].null_count() == 0

    negative = panel.filter(pl.col("signalname") == "Negative")
    assert negative.filter(pl.col("window") == "in_sample").height == 36
    assert negative["is_mean"].unique().to_list() == [-1.0]
    negative_post_sample = negative.filter(pl.col("window") == "post_sample")
    assert negative_post_sample.height == 2
    assert negative_post_sample["ret_norm"].null_count() == negative_post_sample.height
