"""Merge long-short returns, predictor metadata and Ken French factors into the panel.

All join keys are integer ``yyyymm`` values (never the raw OSAP ``date``). Returns are
**percent per month** and are never converted. Window labels follow
docs/architecture.md §4.6: ``in_sample`` is January ``SampleStartYear`` through December
``SampleEndYear``, ``post_publication`` starts January ``Year+1`` (default) or January
``Year`` (alt), ``post_sample`` fills the gap and may be empty, and ``in_sample`` wins
when publication would start inside the sample. Pre-sample months are dropped at build
time, so every panel row is in exactly one window.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from anomaly_decay.panel.windows import classify_frame
from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, validate_frame

__all__ = ["BuildStats", "build_panel", "main"]

_METADATA_COLUMNS: tuple[str, ...] = (
    "signalname",
    "sample_start_year",
    "sample_end_year",
    "pub_year",
    "clear",
)
_FACTOR_COLUMNS: tuple[str, ...] = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd", "rf")
_FACTOR_FILES: tuple[str, ...] = ("ff5_monthly.csv", "umd_monthly.csv")


@dataclass(frozen=True)
class BuildStats:
    """Row and predictor counts from a panel build (printed by ``--dry-run``)."""

    raw_rows: int
    kept_rows: int
    dropped_no_metadata_predictors: int
    dropped_no_metadata_rows: int
    dropped_no_in_sample_predictor_portfolios: int
    dropped_no_in_sample_rows: int
    dropped_pre_sample_rows: int
    dropped_null_ret_rows: int
    rows_per_portfolio: dict[str, int]
    predictors_per_portfolio: dict[str, int]
    rows_per_window: dict[str, int]


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _read_ls_rows(data_dir: Path, vintage: int, portfolio: str) -> pl.DataFrame:
    """Read ``<data_dir>/<vintage>/port_<portfolio>.csv`` and keep ``port == "LS"``.

    ``port`` and ``date`` are read as Utf8 (canonical files store ``"LS"`` and text
    dates); returns stay percent per month.
    """
    path = data_dir / str(vintage) / f"port_{portfolio}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"canonical portfolio file not found: {path}")
    frame = pl.read_csv(path, schema_overrides={"port": pl.Utf8, "date": pl.Utf8})
    _require_columns(frame, ("signalname", "port", "date", "ret"), str(path))
    return frame.filter(pl.col("port") == "LS")


def _read_factors(pinned_dir: Path) -> pl.DataFrame:
    """Load the pinned Ken French files (percent per month), keyed by ``yyyymm``.

    Months covered by one file but not the other keep nulls for the missing factors;
    everything is cast to Float64 for the panel join.
    """
    frames: dict[str, pl.DataFrame] = {}
    for name in _FACTOR_FILES:
        path = pinned_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"pinned factor file not found: {path}")
        frame = pl.read_csv(path, schema_overrides={"yyyymm": pl.Int32})
        _require_columns(frame, ("yyyymm",), str(path))
        frames[name] = frame
    factors = frames[_FACTOR_FILES[0]].join(
        frames[_FACTOR_FILES[1]], on="yyyymm", how="full", coalesce=True
    )
    _require_columns(factors, _FACTOR_COLUMNS, str(pinned_dir))
    return factors.select(["yyyymm", *_FACTOR_COLUMNS]).with_columns(
        pl.col(column).cast(pl.Float64) for column in _FACTOR_COLUMNS
    )


def _in_sample_means(frame: pl.DataFrame) -> pl.DataFrame:
    """In-sample mean of ``ret`` per (signalname, portfolio), summed in month order.

    The independent reference builder (tests/fixtures/build_fixture_panel.py)
    accumulates in ascending month order with plain Python floats. Polars sums with
    SIMD lanes and can differ in the last ulp, which would break the exact fixture
    cross-check, so this reduction uses the same sequential arithmetic.
    """
    in_sample = frame.filter(pl.col("window") == "in_sample").select(
        ["signalname", "portfolio", "yyyymm", "ret"]
    )
    records: list[tuple[str, str, float]] = []
    for key, group in in_sample.partition_by(["signalname", "portfolio"], as_dict=True).items():
        signalname, portfolio = key
        returns = group.sort("yyyymm")["ret"].to_list()
        records.append((signalname, portfolio, sum(returns) / len(returns)))
    means = pl.DataFrame(
        records,
        schema={"signalname": pl.Utf8, "portfolio": pl.Utf8, "is_mean": pl.Float64},
        orient="row",
    )
    return frame.join(means, on=["signalname", "portfolio"], how="left")


def _counts(frame: pl.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    rows_per_portfolio = dict(
        frame.group_by("portfolio").agg(pl.len().alias("n")).sort("portfolio").iter_rows()
    )
    predictors_per_portfolio = dict(
        frame.group_by("portfolio")
        .agg(pl.col("signalname").n_unique().alias("n"))
        .sort("portfolio")
        .iter_rows()
    )
    rows_per_window = dict(
        frame.group_by("window").agg(pl.len().alias("n")).sort("window").iter_rows()
    )
    return rows_per_portfolio, predictors_per_portfolio, rows_per_window


def _prepare(
    data_dir: Path,
    metadata: pl.DataFrame,
    portfolios: Sequence[str],
    *,
    vintage: int = 202510,
) -> tuple[pl.DataFrame, BuildStats]:
    """Join and filter the raw long-short returns; return the rows that enter the panel.

    Drops, in this order: rows whose predictor has no metadata row, predictor-portfolio
    pairs with no non-null in-sample return (their ``is_mean`` would be undefined),
    pre-sample months and rows with a null ``ret``. Nothing here depends on a return
    statistic other than counts, so the CLI can run this on full data before the
    specification freeze.
    """
    ordered_portfolios = list(dict.fromkeys(portfolios))
    if not ordered_portfolios:
        raise ValueError("at least one portfolio is required")
    _require_columns(metadata, _METADATA_COLUMNS, "metadata")
    if metadata["signalname"].n_unique() != metadata.height:
        raise ValueError("metadata contains duplicate signalname values")
    meta = metadata.select(_METADATA_COLUMNS)

    frames = [
        _read_ls_rows(data_dir, vintage, portfolio).with_columns(
            pl.lit(portfolio, dtype=pl.Utf8).alias("portfolio")
        )
        for portfolio in ordered_portfolios
    ]
    raw = pl.concat(frames, how="vertical").with_columns(
        (
            pl.col("date").str.slice(0, 4).cast(pl.Int32) * 100
            + pl.col("date").str.slice(5, 2).cast(pl.Int32)
        ).alias("yyyymm")
    )

    metadata_signals = meta.select("signalname")
    dropped_no_metadata_predictors = (
        raw.select("signalname").unique().join(metadata_signals, on="signalname", how="anti").height
    )
    dropped_no_metadata_rows = raw.join(metadata_signals, on="signalname", how="anti").height

    joined = raw.join(meta, on="signalname", how="inner")
    joined = joined.with_columns(
        classify_frame(joined, "default").alias("window"),
        classify_frame(joined, "alt").alias("window_alt"),
    )
    joined = joined.with_columns(
        pl.when(pl.col("window") == "in_sample")
        .then(pl.col("ret"))
        .otherwise(None)
        .count()
        .over(["signalname", "portfolio"])
        .alias("_in_sample_count")
    )
    no_in_sample = joined.filter(pl.col("_in_sample_count") == 0)
    dropped_no_in_sample_predictor_portfolios = (
        no_in_sample.select(["signalname", "portfolio"]).unique().height
    )
    dropped_no_in_sample_rows = no_in_sample.height
    kept = joined.filter(pl.col("_in_sample_count") > 0).drop("_in_sample_count")

    dropped_pre_sample_rows = kept.filter(pl.col("window") == "pre_sample").height
    kept = kept.filter(pl.col("window") != "pre_sample")
    dropped_null_ret_rows = kept.filter(pl.col("ret").is_null()).height
    kept = kept.filter(pl.col("ret").is_not_null())

    rows_per_portfolio, predictors_per_portfolio, rows_per_window = _counts(kept)
    stats = BuildStats(
        raw_rows=raw.height,
        kept_rows=kept.height,
        dropped_no_metadata_predictors=dropped_no_metadata_predictors,
        dropped_no_metadata_rows=dropped_no_metadata_rows,
        dropped_no_in_sample_predictor_portfolios=dropped_no_in_sample_predictor_portfolios,
        dropped_no_in_sample_rows=dropped_no_in_sample_rows,
        dropped_pre_sample_rows=dropped_pre_sample_rows,
        dropped_null_ret_rows=dropped_null_ret_rows,
        rows_per_portfolio=rows_per_portfolio,
        predictors_per_portfolio=predictors_per_portfolio,
        rows_per_window=rows_per_window,
    )
    return kept, stats


def build_panel(
    data_dir: Path,
    metadata: pl.DataFrame,
    portfolios: Sequence[str],
    *,
    pinned_dir: Path,
    vintage: int = 202510,
) -> pl.DataFrame:
    """Build the analysis panel from canonical raw returns, metadata and pinned factors.

    *data_dir* holds the canonical OSAP CSVs under ``<data_dir>/<vintage>/
    port_<name>.csv``; *metadata* is the predictor table per
    :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`; *portfolios* selects long-short
    series (``op``, ``deciles_ew``, ``deciles_vw``; rows with ``port == "LS"``);
    *pinned_dir* holds ``ff5_monthly.csv`` and ``umd_monthly.csv``.

    Output: a frame exactly per :data:`anomaly_decay.schemas.PANEL_SCHEMA`, keyed by
    (``signalname``, ``portfolio``, ``yyyymm``) and sorted by that key. Only months
    classified ``in_sample``, ``post_sample`` or ``post_publication`` are kept
    (pre-sample months are dropped), so every row is in exactly one window. Sources are
    joined on ``yyyymm`` (never the raw OSAP ``date``); ``month`` is the first day of
    the month. Returns are **percent per month** (no conversion). ``is_mean`` is the
    predictor×portfolio in-sample mean of ``ret``; ``ret_norm`` is ``ret / is_mean``,
    null when ``is_mean <= 0``. ``event_time_sample_end`` counts months from January
    ``SampleEndYear+1``; ``event_time_pub`` counts months from the first default
    post-publication month. Factor columns are null where Ken French coverage is
    absent. Window flags follow architecture §4.6 under both conventions (``default``
    and ``alt``).
    """
    kept, _ = _prepare(data_dir, metadata, portfolios, vintage=vintage)
    kept = kept.with_columns(
        pl.date(pl.col("yyyymm") // 100, pl.col("yyyymm") % 100, 1).alias("month"),
        (
            (pl.col("yyyymm") // 100 - (pl.col("sample_end_year").cast(pl.Int32) + 1)) * 12
            + (pl.col("yyyymm") % 100 - 1)
        )
        .cast(pl.Int32)
        .alias("event_time_sample_end"),
        (
            (pl.col("yyyymm") // 100 - (pl.col("pub_year").cast(pl.Int32) + 1)) * 12
            + (pl.col("yyyymm") % 100 - 1)
        )
        .cast(pl.Int32)
        .alias("event_time_pub"),
        (pl.col("window") == "post_sample").cast(pl.Int8).alias("post_sample"),
        (pl.col("window") == "post_publication").cast(pl.Int8).alias("post_publication"),
        (pl.col("window_alt") == "post_sample").cast(pl.Int8).alias("post_sample_alt"),
        (pl.col("window_alt") == "post_publication").cast(pl.Int8).alias("post_publication_alt"),
    )
    kept = _in_sample_means(kept)
    kept = kept.with_columns(
        pl.when(pl.col("is_mean") > 0)
        .then(pl.col("ret") / pl.col("is_mean"))
        .otherwise(None)
        .alias("ret_norm")
    )
    kept = kept.join(_read_factors(pinned_dir), on="yyyymm", how="left")

    panel = kept.select(list(PANEL_SCHEMA)).sort(list(PANEL_KEY))
    panel = panel.with_columns(pl.col(column).cast(dtype) for column, dtype in PANEL_SCHEMA.items())
    validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)
    return panel


def _format_counts(label: str, counts: dict[str, int]) -> str:
    body = ", ".join(f"{key}={value}" for key, value in counts.items())
    return f"{label}: {body}"


def _print_stats(stats: BuildStats) -> None:
    print("dry-run: counts only, nothing written")
    print(f"raw LS rows: {stats.raw_rows}")
    print(f"kept rows: {stats.kept_rows}")
    print(_format_counts("rows per portfolio", stats.rows_per_portfolio))
    print(_format_counts("predictors per portfolio", stats.predictors_per_portfolio))
    print(_format_counts("rows per window", stats.rows_per_window))
    print(f"dropped rows without metadata: {stats.dropped_no_metadata_rows}")
    print(
        "dropped predictor-portfolios without in-sample returns: "
        f"{stats.dropped_no_in_sample_predictor_portfolios} "
        f"({stats.dropped_no_in_sample_rows} rows)"
    )
    print(f"dropped pre-sample rows: {stats.dropped_pre_sample_rows}")
    print(f"dropped rows with null ret: {stats.dropped_null_ret_rows}")
    print(f"predictors without metadata: {stats.dropped_no_metadata_predictors}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.panel.build --data-dir DIR --metadata PATH --out PATH``.

    ``--dry-run`` prints row counts only (no return statistic), for use on full data
    before the specification freeze.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.panel.build",
        description="Build the panel from canonical raw data, metadata and factors.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="raw data directory")
    parser.add_argument(
        "--pinned-dir",
        type=Path,
        default=Path("data/pinned"),
        help="pinned Ken French directory (default: data/pinned)",
    )
    parser.add_argument("--vintage", type=int, default=202510, help="OSAP release vintage")
    parser.add_argument("--metadata", type=Path, required=True, help="predictors.parquet path")
    parser.add_argument(
        "--portfolios",
        default="op,deciles_ew,deciles_vw",
        help="comma-separated portfolio names (default: op,deciles_ew,deciles_vw)",
    )
    parser.add_argument("--out", type=Path, required=True, help="panel.parquet path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print row counts only and write nothing (allowed before the freeze)",
    )
    args = parser.parse_args(argv)

    metadata = pl.read_parquet(args.metadata)
    portfolios = tuple(name for name in args.portfolios.split(",") if name)
    if args.dry_run:
        _, stats = _prepare(args.data_dir, metadata, portfolios, vintage=args.vintage)
        _print_stats(stats)
        return 0

    frame = build_panel(
        args.data_dir,
        metadata,
        portfolios,
        pinned_dir=args.pinned_dir,
        vintage=args.vintage,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    print(f"wrote {args.out}: {frame.height} rows, {frame['signalname'].n_unique()} predictors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
