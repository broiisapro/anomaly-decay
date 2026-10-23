"""Descriptive results feeding the figures and the report (architecture §5.3).

All three tables cover the **full-extension universe**: the ``op``
portfolio only, every predictor in the panel, every month of the vintage, default
window convention. They answer "what does the raw data look like by window and around
the sample-end and publication events", not "what do the regressions say" — the
estimation side owns the latter.

- :func:`predictor_summary` — per (signalname, window) month count, mean return and
  standard error of the mean.
- :func:`event_time` — cross-predictor averages of the long-short return around the two
  events, using the panel's ``event_time_sample_end`` / ``event_time_pub`` columns as
  given (computed by :mod:`anomaly_decay.panel.build`).
- :func:`coverage` — count-level facts about the vintage, for the report's data section.

``__main__`` writes ``predictor_summary.csv``, ``event_time.csv`` and ``coverage.csv``
through :func:`anomaly_decay.io.write_results`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from anomaly_decay.io import write_results
from anomaly_decay.schemas import PREDICTORS_SCHEMA, validate_frame

_ANCHORS: tuple[tuple[str, str], ...] = (
    ("sample_end", "event_time_sample_end"),
    ("publication", "event_time_pub"),
)
_COVERAGE_PUBLICATION_CUTOFF = 2013
_COVERAGE_PORTFOLIOS: tuple[str, ...] = ("deciles_ew", "deciles_vw")


def predictor_summary(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (signalname, window) statistics on the full-extension universe.

    *panel* is a full panel per :data:`anomaly_decay.schemas.PANEL_SCHEMA`; only the
    ``op`` portfolio is used. Output per ``RESULTS_SCHEMAS["predictor_summary.csv"]``:
    ``n_months``, ``mean_ret`` (percent per month) and ``se_mean = sd / sqrt(n)``
    (sample sd, null when ``n < 2``).
    """
    frame = panel.filter(pl.col("portfolio") == "op").select("signalname", "window", "ret")
    stats = frame.group_by("signalname", "window").agg(
        pl.len().cast(pl.Int64).alias("n_months"),
        pl.col("ret").mean().alias("mean_ret"),
        (pl.col("ret").std() / pl.len().sqrt()).alias("se_mean"),
    )
    return stats.select(
        "signalname",
        "window",
        pl.col("n_months").cast(pl.Int64),
        pl.col("mean_ret").cast(pl.Float64),
        pl.col("se_mean").cast(pl.Float64),
    ).sort(["signalname", "window"])


def event_time(panel: pl.DataFrame, lo: int = -60, hi: int = 120) -> pl.DataFrame:
    """Cross-predictor mean return by event month around both events.

    *panel* is a full panel per :data:`anomaly_decay.schemas.PANEL_SCHEMA`; only the
    ``op`` portfolio is used. Each row lands at its ``event_time_sample_end`` (anchor
    ``sample_end``) and its ``event_time_pub`` (anchor ``publication``) as given by the
    panel builder. For every anchor and event month in ``[lo, hi]`` the output per
    ``RESULTS_SCHEMAS["event_time.csv"]`` holds the cross-predictor mean ``mean_ret``
    (percent per month), ``se`` (cross-predictor sd / sqrt(n_predictors)) and
    ``n_predictors``. A predictor with a null event time is skipped.
    """
    op = panel.filter(pl.col("portfolio") == "op")
    anchors: list[pl.DataFrame] = []
    for anchor, column in _ANCHORS:
        if column not in op.columns:
            continue
        stats = (
            op.select(
                pl.lit(anchor).alias("anchor"),
                pl.col(column).alias("event_month"),
                pl.col("ret"),
            )
            .filter(
                pl.col("event_month").is_not_null()
                & (pl.col("event_month") >= lo)
                & (pl.col("event_month") <= hi)
            )
            .group_by("anchor", "event_month")
            .agg(
                pl.col("ret").mean().alias("mean_ret"),
                (pl.col("ret").std() / pl.len().sqrt()).alias("se"),
                pl.len().cast(pl.Int64).alias("n_predictors"),
            )
        )
        anchors.append(stats)
    if not anchors:
        return pl.DataFrame(
            schema={
                "anchor": pl.Utf8,
                "event_month": pl.Int64,
                "mean_ret": pl.Float64,
                "se": pl.Float64,
                "n_predictors": pl.Int64,
            }
        )
    return (
        pl.concat(anchors)
        .select(
            "anchor",
            pl.col("event_month").cast(pl.Int64),
            pl.col("mean_ret").cast(pl.Float64),
            pl.col("se").cast(pl.Float64),
            pl.col("n_predictors").cast(pl.Int64),
        )
        .sort(["anchor", "event_month"])
    )


def coverage(panel: pl.DataFrame, metadata: pl.DataFrame, vintage: int) -> pl.DataFrame:
    """Count-level coverage facts for the report's data section.

    *panel* is the unfiltered panel per :data:`anomaly_decay.schemas.PANEL_SCHEMA` (the
    ``op`` counts come from it directly); *metadata* per
    :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`; *vintage* is the OSAP release year
    (e.g. ``202510``). Output rows per ``RESULTS_SCHEMAS["coverage.csv"]``: ``vintage``,
    ``first_month``, ``last_month``, ``n_predictors_op``, ``n_predictors_deciles_ew``,
    ``n_predictors_deciles_vw``, ``n_clear``, ``n_published_by_2013``,
    ``n_predictor_months_op`` — every value as text.
    """
    op = panel.filter(pl.col("portfolio") == "op")
    op_signal_count = op["signalname"].n_unique()
    items: list[tuple[str, int]] = [
        ("vintage", vintage),
        ("first_month", panel.select(pl.min("yyyymm")).item()),
        ("last_month", panel.select(pl.max("yyyymm")).item()),
        ("n_predictors_op", op_signal_count),
    ]
    for portfolio in _COVERAGE_PORTFOLIOS:
        count = panel.filter(pl.col("portfolio") == portfolio)["signalname"].n_unique()
        items.append((f"n_predictors_{portfolio}", count))
    items.extend(
        [
            ("n_clear", metadata.filter(pl.col("clear")).height),
            (
                "n_published_by_2013",
                metadata.filter(pl.col("pub_year") <= _COVERAGE_PUBLICATION_CUTOFF).height,
            ),
            ("n_predictor_months_op", op.height),
        ]
    )
    return pl.DataFrame(
        [(item, str(value)) for item, value in items],
        schema={"item": pl.Utf8, "value": pl.Utf8},
        orient="row",
    ).sort("item")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build the descriptive results from a panel and predictor metadata.

    Writes ``predictor_summary.csv``, ``event_time.csv`` and ``coverage.csv`` under
    *results-dir* through :func:`anomaly_decay.io.write_results`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.descriptive",
        description="Write predictor summary, event time and coverage results.",
    )
    parser.add_argument("--panel", type=Path, required=True, help="panel.parquet path")
    parser.add_argument("--metadata", type=Path, required=True, help="predictors.parquet path")
    parser.add_argument("--vintage", type=int, required=True, help="OSAP release year, e.g. 202510")
    parser.add_argument("--results-dir", type=Path, required=True, help="results output directory")
    args = parser.parse_args(argv)

    panel = pl.read_parquet(args.panel)
    metadata = pl.read_parquet(args.metadata)
    validate_frame(metadata, PREDICTORS_SCHEMA, key=["signalname"])
    write_results(predictor_summary(panel), "predictor_summary.csv", args.results_dir)
    write_results(event_time(panel), "event_time.csv", args.results_dir)
    write_results(coverage(panel, metadata, args.vintage), "coverage.csv", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
