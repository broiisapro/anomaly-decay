"""Opt-in invariants over the real OSAP 202510 canonical cache.

The whole module is skipped with a clear reason unless ``data/raw/202510`` exists
(``make fetch``). The checks are structural only -- coverage, keys, units, date sanity
-- because nothing on the full data may compute returns by window, regressions or decay
before the ``specs-frozen`` tag (pre-registration guard, O26). The single return
statistic here is the median absolute long-short return, a percent-per-month unit check.

``tests/fixtures/realdata_expectations.csv`` records the per-predictor first/last
``yyyymm`` for every portfolio; regenerate it explicitly with::

    ANOMALY_DECAY_REGENERATE_REALDATA_EXPECTATIONS=1 uv run pytest -q tests/test_realdata.py

and commit the result.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Mapping
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.io import write_csv_canonical
from polars.testing import assert_frame_equal

REPO_ROOT = Path(__file__).parents[1]
VINTAGE = 202510
RAW_DIR = REPO_ROOT / "data" / "raw" / str(VINTAGE)
PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")
EXPECTATIONS_PATH = REPO_ROOT / "tests" / "fixtures" / "realdata_expectations.csv"
REGENERATE_ENV = "ANOMALY_DECAY_REGENERATE_REALDATA_EXPECTATIONS"
CSV_OVERRIDES: dict[str, pl.DataType] = {"port": pl.Utf8, "date": pl.Utf8}
LS_COLUMNS = ("signalname", "port", "date", "ret", "signallag")
EXPECTATION_COLUMNS = ("signalname", "portfolio", "first_yyyymm", "last_yyyymm")
EXPECTED_PREDICTORS = 212
EXPECTED_DECILES_PREDICTORS = 179

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not RAW_DIR.is_dir(),
        reason=f"OSAP cache absent under {RAW_DIR.relative_to(REPO_ROOT)} (run make fetch)",
    ),
]


@pytest.fixture(scope="module")
def port_frames() -> Mapping[str, pl.DataFrame]:
    return {
        name: pl.read_csv(
            RAW_DIR / f"port_{name}.csv",
            columns=list(LS_COLUMNS),
            schema_overrides=CSV_OVERRIDES,
        )
        for name in PORTFOLIOS
    }


@pytest.fixture(scope="module")
def predictor_names() -> list[str]:
    doc = pl.read_csv(RAW_DIR / "signal_doc.csv", infer_schema_length=0)
    return sorted(doc.filter(pl.col("Cat.Signal") == "Predictor")["Acronym"].to_list())


def _ls(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("port") == "LS")


def _with_yyyymm(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("date").str.slice(0, 7).str.replace("-", "").cast(pl.Int32).alias("yyyymm")
    )


def _build_expectations(port_frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    parts = []
    for portfolio in PORTFOLIOS:
        grouped = (
            _with_yyyymm(_ls(port_frames[portfolio]))
            .group_by("signalname")
            .agg(
                pl.col("yyyymm").min().alias("first_yyyymm"),
                pl.col("yyyymm").max().alias("last_yyyymm"),
            )
            .with_columns(pl.lit(portfolio).alias("portfolio"))
        )
        parts.append(grouped)
    return pl.concat(parts).select(list(EXPECTATION_COLUMNS)).sort(["signalname", "portfolio"])


def _read_expectations() -> pl.DataFrame:
    frame = pl.read_csv(EXPECTATIONS_PATH)
    return frame.select(
        pl.col("signalname").cast(pl.Utf8),
        pl.col("portfolio").cast(pl.Utf8),
        pl.col("first_yyyymm").cast(pl.Int32),
        pl.col("last_yyyymm").cast(pl.Int32),
    )


@pytest.fixture(scope="module", autouse=True)
def regenerate_expectations_when_requested(port_frames: Mapping[str, pl.DataFrame]) -> None:
    if os.environ.get(REGENERATE_ENV) == "1":
        write_csv_canonical(
            _build_expectations(port_frames), EXPECTATIONS_PATH, ["signalname", "portfolio"]
        )


def test_every_predictor_has_op_ls_rows(
    port_frames: Mapping[str, pl.DataFrame], predictor_names: list[str]
) -> None:
    assert len(predictor_names) == EXPECTED_PREDICTORS
    op_signals = set(_ls(port_frames["op"])["signalname"].unique().to_list())
    missing = sorted(set(predictor_names) - op_signals)
    assert not missing, f"predictors without op long-short rows: {missing}"


def test_deciles_coverage_matches_the_recorded_set(
    port_frames: Mapping[str, pl.DataFrame], predictor_names: list[str]
) -> None:
    recorded = _read_expectations()
    universe = set(predictor_names)
    for portfolio in PORTFOLIOS:
        actual = set(_ls(port_frames[portfolio])["signalname"].unique().to_list())
        committed = set(recorded.filter(pl.col("portfolio") == portfolio)["signalname"].to_list())
        assert actual == committed, (
            f"{portfolio}: coverage differs from {EXPECTATIONS_PATH.name}; regenerate it if "
            "the vintage changed"
        )
        assert actual <= universe, portfolio
    for portfolio in ("deciles_ew", "deciles_vw"):
        committed = set(recorded.filter(pl.col("portfolio") == portfolio)["signalname"].to_list())
        assert len(committed) == EXPECTED_DECILES_PREDICTORS, portfolio


def test_no_duplicate_signal_portfolio_date_keys(port_frames: Mapping[str, pl.DataFrame]) -> None:
    for name, frame in port_frames.items():
        keys = frame.select(["signalname", "port", "date"])
        assert keys.n_unique() == frame.height, f"duplicate keys in port_{name}.csv"


def test_ls_returns_have_no_nulls_and_a_percent_monthly_median(
    port_frames: Mapping[str, pl.DataFrame],
) -> None:
    for name, frame in port_frames.items():
        ls = _ls(frame)
        assert ls["ret"].null_count() == 0, name
        median_abs = ls["ret"].abs().median()
        assert median_abs is not None
        assert 0.5 <= median_abs <= 10.0, (
            f"{name}: median |LS return| = {median_abs} -- returns look like decimals, "
            "not percent per month"
        )


def test_ls_dates_are_valid_calendar_days_with_no_modern_weekends(
    port_frames: Mapping[str, pl.DataFrame],
) -> None:
    # OSAP dates are the last trading day of the month. The NYSE still held Saturday
    # sessions until mid-1952, so weekend dates are legitimate only as Saturdays
    # before then -- the real cache has 4,084 (op) and 3,332 (each deciles file) of
    # them, all 1926-01-30..1951-03-31, and no Sunday ever.
    saturday_trading_ended = datetime.date(1952, 6, 1)
    for name, frame in port_frames.items():
        ls = _ls(frame)
        dates = ls["date"].str.to_date("%Y-%m-%d", strict=False)
        assert dates.null_count() == 0, f"{name}: malformed or out-of-month dates"
        assert (dates.dt.weekday() <= 6).all(), f"{name}: Sunday date"
        assert (dates.dt.strftime("%Y-%m-%d") == ls["date"]).all(), name
        modern_saturdays = ls.filter((dates.dt.weekday() == 6) & (dates >= saturday_trading_ended))
        assert modern_saturdays.height == 0, (
            f"{name}: {modern_saturdays.height} Saturday dates after Saturday trading ended"
        )


def test_ls_signallag_is_null(port_frames: Mapping[str, pl.DataFrame]) -> None:
    for name, frame in port_frames.items():
        ls = _ls(frame)
        assert ls["signallag"].null_count() == ls.height, name


def test_first_and_last_months_match_recorded_expectations(
    port_frames: Mapping[str, pl.DataFrame],
) -> None:
    actual = _build_expectations(port_frames)
    expected = _read_expectations()
    assert_frame_equal(actual, expected, check_row_order=True, check_dtypes=True)
