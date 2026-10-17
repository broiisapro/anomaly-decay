"""Tests for the panel build (docs/architecture.md §4.6 and §5.2)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.panel.build import _prepare, build_panel
from anomaly_decay.panel.windows import classify_month
from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, PREDICTORS_SCHEMA, validate_frame

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_DATA = FIXTURES / "data"
FIXTURE_OSAP = FIXTURE_DATA / "osap"
FIXTURE_PINNED = FIXTURE_DATA / "pinned"
VINTAGE = 202510
PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")

# (signalname, sample_start_year, sample_end_year, pub_year) from the fixture signal doc.
MOM12M = ("Mom12m", 1964, 1989, 1993)
GP = ("GP", 1963, 2010, 2013)


def _local_metadata() -> pl.DataFrame:
    """Build the fixture metadata frame without agent A's metadata module (issue #11)."""
    doc = pl.read_csv(
        FIXTURE_OSAP / str(VINTAGE) / "signal_doc.csv", infer_schema_length=500
    ).filter(pl.col("Cat.Signal") == "Predictor")
    height = doc.height
    return pl.DataFrame(
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


def _hand_metadata(rows: list[tuple[str, int, int, int]]) -> pl.DataFrame:
    height = len(rows)
    return pl.DataFrame(
        {
            "signalname": [row[0] for row in rows],
            "sample_start_year": pl.Series([row[1] for row in rows], dtype=pl.Int16),
            "sample_end_year": pl.Series([row[2] for row in rows], dtype=pl.Int16),
            "pub_year": pl.Series([row[3] for row in rows], dtype=pl.Int16),
            "journal": ["J"] * height,
            "authors": ["A"] * height,
            "cat_economic": ["valuation"] * height,
            "cat_data": ["Price"] * height,
            "clear": [True] * height,
            "date_source": ["test"] * height,
        }
    ).select(list(PREDICTORS_SCHEMA))


def _write_port_file(
    tmp_path: Path,
    portfolio: str,
    rows: list[tuple[str, str, float | None]],
    *,
    port: str = "LS",
) -> Path:
    data_dir = tmp_path / str(VINTAGE)
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "signalname": [row[0] for row in rows],
            "port": [port] * len(rows),
            "date": [row[1] for row in rows],
            "ret": pl.Series([row[2] for row in rows], dtype=pl.Float64),
            "signallag": [1] * len(rows),
            "Nlong": [100] * len(rows),
            "Nshort": [100] * len(rows),
        }
    )
    path = data_dir / f"port_{portfolio}.csv"
    frame.write_csv(path)
    return path


def _raw_rows(predictor: str, portfolio: str = "op") -> pl.DataFrame:
    raw = pl.read_csv(
        FIXTURE_OSAP / str(VINTAGE) / f"port_{portfolio}.csv",
        schema_overrides={"port": pl.Utf8, "date": pl.Utf8},
    )
    return raw.filter((pl.col("signalname") == predictor) & (pl.col("port") == "LS"))


@pytest.fixture(scope="module")
def metadata() -> pl.DataFrame:
    return _local_metadata()


@pytest.fixture(scope="module")
def panel(metadata: pl.DataFrame) -> pl.DataFrame:
    return build_panel(FIXTURE_OSAP, metadata, PORTFOLIOS, pinned_dir=FIXTURE_PINNED)


def test_panel_matches_schema_and_key(panel: pl.DataFrame) -> None:
    assert panel.columns == list(PANEL_SCHEMA)
    assert panel.schema == PANEL_SCHEMA
    validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)
    assert set(panel["window"].unique().to_list()) <= {
        "in_sample",
        "post_sample",
        "post_publication",
    }


def test_returns_unit_is_percent_monthly(panel: pl.DataFrame) -> None:
    raw = _raw_rows("Mom12m")
    raw_row = raw.filter(pl.col("date").str.starts_with("1964-01")).row(0, named=True)
    raw_ret = float(raw_row["ret"])
    panel_ret = panel.filter(
        (pl.col("signalname") == "Mom12m")
        & (pl.col("portfolio") == "op")
        & (pl.col("yyyymm") == 196401)
    )["ret"].item()
    # The raw value passes through unchanged: OSAP percent-per-month units are kept.
    assert panel_ret == raw_ret

    median_absolute = panel.filter(pl.col("portfolio") == "op")["ret"].abs().median()
    assert median_absolute is not None
    assert 0.5 < median_absolute < 10.0

    ff5 = pl.read_csv(FIXTURE_PINNED / "ff5_monthly.csv")
    mkt_rf_196308 = ff5.filter(pl.col("yyyymm") == 196308)["mkt_rf"].item()
    assert mkt_rf_196308 == 5.08
    panel_mkt_rf = panel.filter(pl.col("yyyymm") == 196308)["mkt_rf"].unique().to_list()
    assert panel_mkt_rf == [mkt_rf_196308]


def test_factors_join_on_yyyymm_not_raw_date(panel: pl.DataFrame) -> None:
    # The raw OSAP date is the last day of January 1963 ("1963-01-31"); UMD covers
    # 196301 while FF5 starts in 196307, so a correct yyyymm join leaves umd filled
    # and mkt_rf null for that month.
    raw_row = _raw_rows("AM").filter(pl.col("date").str.starts_with("1963-01")).row(0, named=True)
    assert raw_row["date"] == "1963-01-31"
    row = panel.filter(
        (pl.col("signalname") == "AM")
        & (pl.col("portfolio") == "op")
        & (pl.col("yyyymm") == 196301)
    ).row(0, named=True)
    assert row["umd"] == -2.14
    assert row["mkt_rf"] is None

    # A month covered by both files keeps the pinned values.
    umd = pl.read_csv(FIXTURE_PINNED / "umd_monthly.csv")
    umd_199501 = umd.filter(pl.col("yyyymm") == 199501)["umd"].item()
    ff5 = pl.read_csv(FIXTURE_PINNED / "ff5_monthly.csv")
    mkt_rf_199501 = ff5.filter(pl.col("yyyymm") == 199501)["mkt_rf"].item()
    row_199501 = panel.filter(
        (pl.col("signalname") == "AM")
        & (pl.col("portfolio") == "op")
        & (pl.col("yyyymm") == 199501)
    ).row(0, named=True)
    assert row_199501["umd"] == umd_199501
    assert row_199501["mkt_rf"] == mkt_rf_199501


def test_pre_sample_rows_are_absent(panel: pl.DataFrame) -> None:
    # The fixture raw data starts one year before SampleStartYear, so pre-sample rows
    # exist in the inputs and must be dropped at build time.
    for signalname, start, _end, _pub in (MOM12M, GP):
        months = panel.filter((pl.col("signalname") == signalname) & (pl.col("portfolio") == "op"))[
            "yyyymm"
        ]
        assert months.min() == start * 100 + 1
    first_month = pl.col("sample_start_year").cast(pl.Int32) * 100 + 1
    assert panel.filter(pl.col("yyyymm") < first_month).height == 0


def test_is_mean_equals_independent_in_sample_mean(panel: pl.DataFrame) -> None:
    raw = _raw_rows("Mom12m")
    rows = sorted(
        (int(row["date"][0:4]) * 100 + int(row["date"][5:7]), row["ret"])
        for row in raw.iter_rows(named=True)
    )
    in_sample = [
        float(ret)
        for yyyymm, ret in rows
        if classify_month(yyyymm, 1964, 1989, 1993) == "in_sample" and ret is not None
    ]
    independent_mean = sum(in_sample) / len(in_sample)
    mom = panel.filter((pl.col("signalname") == "Mom12m") & (pl.col("portfolio") == "op"))
    assert mom["is_mean"].unique().item() == independent_mean

    expected_norm = [ret / independent_mean for ret in mom["ret"].to_list()]
    assert mom["ret_norm"].to_list() == expected_norm


def test_ret_norm_null_when_is_mean_is_non_positive(tmp_path: Path) -> None:
    signal = ("NegMean", 1990, 1992, 1993)
    rows = [
        ("NegMean", "1990-01-15", 0.5),
        ("NegMean", "1990-02-15", -1.5),
        ("NegMean", "1990-03-15", 0.25),
        ("NegMean", "1993-01-15", 2.0),
    ]
    _write_port_file(tmp_path, "op", rows)
    frame = build_panel(tmp_path, _hand_metadata([signal]), ("op",), pinned_dir=FIXTURE_PINNED)
    assert frame["is_mean"].unique().to_list() == [-0.25]
    assert frame["ret_norm"].null_count() == frame.height
    assert frame.filter(pl.col("window") == "in_sample").height == 3


def test_ret_norm_is_ret_over_is_mean_when_positive(tmp_path: Path) -> None:
    rows = [("PosMean", "1990-01-15", 1.0), ("PosMean", "1990-02-15", 2.0)]
    _write_port_file(tmp_path, "op", rows)
    frame = build_panel(
        tmp_path,
        _hand_metadata([("PosMean", 1990, 1990, 1993)]),
        ("op",),
        pinned_dir=FIXTURE_PINNED,
    )
    assert frame["is_mean"].unique().to_list() == [1.5]
    expected = [ret / 1.5 for ret in frame["ret"].to_list()]
    assert frame["ret_norm"].to_list() == expected


def test_event_time_anchors(panel: pl.DataFrame) -> None:
    mom = panel.filter((pl.col("signalname") == "Mom12m") & (pl.col("portfolio") == "op"))
    sample_end = dict(zip(mom["yyyymm"], mom["event_time_sample_end"], strict=True))
    # Anchor: January SampleEndYear+1 == 0; December SampleEndYear == -1.
    assert sample_end[199001] == 0
    assert sample_end[198912] == -1
    assert sample_end[199012] == 11

    pub = dict(zip(mom["yyyymm"], mom["event_time_pub"], strict=True))
    # Anchor: first default post-publication month (January Year+1) == 0.
    assert pub[199401] == 0
    assert pub[199312] == -1
    assert pub[199501] == 12
    # In-sample months still carry the anchor.
    assert pub[198912] == -49


def test_window_flags_and_alt_convention(panel: pl.DataFrame) -> None:
    for window, flag in (
        ("post_sample", "post_sample"),
        ("post_publication", "post_publication"),
    ):
        assert panel.filter(pl.col("window") == window)[flag].unique().to_list() == [1]
        assert panel.filter(pl.col("window") != window)[flag].unique().to_list() == [0]
        assert panel.filter(pl.col("window_alt") == window)[f"{flag}_alt"].unique().to_list() == [1]

    mom = panel.filter((pl.col("signalname") == "Mom12m") & (pl.col("portfolio") == "op"))
    alt = dict(zip(mom["yyyymm"], mom["window_alt"], strict=True))
    # Publication 1993: alt convention starts post-publication in January 1993.
    assert alt[199212] == "post_sample"
    assert alt[199301] == "post_publication"
    assert alt[199412] == "post_publication"


def test_predictor_without_metadata_is_dropped(tmp_path: Path) -> None:
    rows = [
        ("Known", "1990-01-15", 1.0),
        ("Ghost", "1990-01-15", 2.0),
        ("Ghost", "1990-02-15", 3.0),
    ]
    _write_port_file(tmp_path, "op", rows)
    frame = build_panel(
        tmp_path, _hand_metadata([("Known", 1990, 1990, 1993)]), ("op",), pinned_dir=FIXTURE_PINNED
    )
    assert frame["signalname"].unique().to_list() == ["Known"]


def test_group_without_in_sample_returns_is_dropped(tmp_path: Path) -> None:
    # The sample period is 1970-1971 but data only exists from 1990: is_mean would be
    # undefined, so the whole predictor-portfolio group is dropped.
    rows = [("Late", "1990-01-15", 1.0), ("Late", "1990-02-15", 2.0)]
    _write_port_file(tmp_path, "op", rows)
    frame = build_panel(
        tmp_path, _hand_metadata([("Late", 1970, 1971, 1993)]), ("op",), pinned_dir=FIXTURE_PINNED
    )
    assert frame.height == 0
    assert frame.columns == list(PANEL_SCHEMA)


def test_null_ret_rows_are_dropped(panel: pl.DataFrame) -> None:
    # The fixture has three null-ret months for MaxRet (1995-01..03, in-sample).
    raw = _raw_rows("MaxRet")
    null_months = {
        int(row["date"][0:4]) * 100 + int(row["date"][5:7])
        for row in raw.iter_rows(named=True)
        if row["ret"] is None
    }
    assert null_months == {199501, 199502, 199503}
    panel_months = set(
        panel.filter((pl.col("signalname") == "MaxRet") & (pl.col("portfolio") == "op"))[
            "yyyymm"
        ].to_list()
    )
    assert panel_months.isdisjoint(null_months)


def test_non_long_short_rows_are_ignored(tmp_path: Path) -> None:
    rows = [
        ("OnlyLS", "1990-01-31", 1.0),
        ("OnlyLS", "1990-02-28", 2.0),
        ("OnlyLS", "1990-03-31", -1.0),
    ]
    _write_port_file(tmp_path, "op", rows, port="01")
    frame = build_panel(
        tmp_path, _hand_metadata([("OnlyLS", 1990, 1990, 1993)]), ("op",), pinned_dir=FIXTURE_PINNED
    )
    assert frame.height == 0
    assert frame.columns == list(PANEL_SCHEMA)

    _write_port_file(tmp_path, "op", rows)
    frame = build_panel(
        tmp_path, _hand_metadata([("OnlyLS", 1990, 1990, 1993)]), ("op",), pinned_dir=FIXTURE_PINNED
    )
    assert frame.height == 3


def test_build_stats_count_dropped_rows(tmp_path: Path) -> None:
    rows = [
        ("Good", "1989-12-15", 0.5),
        ("Good", "1990-01-15", 1.0),
        ("Good", "1990-02-15", None),
        ("Good", "1990-03-15", 2.0),
        ("Ghost", "1990-01-15", 1.0),
        ("Late", "1995-01-15", 1.0),
    ]
    _write_port_file(tmp_path, "op", rows)
    metadata = _hand_metadata([("Good", 1990, 1990, 1993), ("Late", 1970, 1971, 1993)])
    _, stats = _prepare(tmp_path, metadata, ("op",), vintage=VINTAGE)
    assert stats.raw_rows == 6
    assert stats.kept_rows == 2
    assert stats.dropped_no_metadata_predictors == 1
    assert stats.dropped_no_metadata_rows == 1
    assert stats.dropped_no_in_sample_predictor_portfolios == 1
    assert stats.dropped_no_in_sample_rows == 1
    assert stats.dropped_pre_sample_rows == 1
    assert stats.dropped_null_ret_rows == 1
    assert stats.rows_per_portfolio == {"op": 2}
    assert stats.predictors_per_portfolio == {"op": 1}
    assert stats.rows_per_window == {"in_sample": 2}


def test_duplicate_metadata_signalname_is_rejected(tmp_path: Path) -> None:
    metadata = _hand_metadata([("Dup", 1990, 1990, 1993), ("Dup", 1990, 1990, 1993)])
    with pytest.raises(ValueError, match="duplicate signalname"):
        build_panel(tmp_path, metadata, ("op",), pinned_dir=FIXTURE_PINNED)


def test_missing_portfolio_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="port_op.csv"):
        build_panel(
            tmp_path, _hand_metadata([("X", 1990, 1990, 1993)]), ("op",), pinned_dir=FIXTURE_PINNED
        )
