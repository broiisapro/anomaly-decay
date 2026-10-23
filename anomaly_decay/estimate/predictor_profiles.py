"""Per-predictor bibliographic metadata and event-year return profiles (decision O45).

Two descriptive data products that feed the Phase 3 companion page. Neither is a
specification: they summarise the committed panel directly, after the specification
freeze, and the report labels them as descriptive.

- :func:`predictor_info` — one row per predictor with the bibliographic fields the site
  shows: authors, journal, publication year, sample years, clear/likely flag and the
  economic category.
- :func:`predictor_event_time` — per predictor and anchor (``sample_end`` /
  ``publication``) the mean ``op`` long-short return and month count by **event year**,
  for event months −60…119 (years −5…9). Yearly aggregation keeps the site on derived
  statistics rather than republishing monthly OSAP series.

Event months are the panel's ``event_time_sample_end`` and ``event_time_pub`` columns:
0 is the first post-sample (or first post-publication) month, so year −1 is the last
in-sample year for ``sample_end`` and the publication year for ``publication``. The
event year uses floor division, so month −1 and −12 both belong to year −1 and month
−13 to year −2 (Python ``//`` semantics), which a test pins down.

``__main__`` writes ``predictor_info.csv`` and ``predictor_event_time.csv`` through
:func:`anomaly_decay.io.write_results`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from anomaly_decay.io import write_results
from anomaly_decay.schemas import PREDICTORS_SCHEMA, RESULTS_SCHEMAS, validate_frame

_ANCHORS: tuple[tuple[str, str], ...] = (
    ("sample_end", "event_time_sample_end"),
    ("publication", "event_time_pub"),
)

EVENT_MONTH_LO = -60
EVENT_MONTH_HI = 119
"""Event-month window kept in the profiles: −60…119 inclusive (event years −5…9)."""

EVENT_YEAR_LO = EVENT_MONTH_LO // 12
EVENT_YEAR_HI = EVENT_MONTH_HI // 12
"""Event-year bounds implied by :data:`EVENT_MONTH_LO`/:data:`EVENT_MONTH_HI`."""

_EVENT_TIME_COLUMNS: tuple[str, ...] = tuple(RESULTS_SCHEMAS["predictor_event_time.csv"])


def predictor_info(metadata: pl.DataFrame) -> pl.DataFrame:
    """Per-predictor bibliographic table for the site.

    *metadata* is the predictor table per :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`
    (one row per predictor, sorted by ``signalname``). Output per
    ``RESULTS_SCHEMAS["predictor_info.csv"]``: ``signalname``, ``authors``, ``journal``,
    ``pub_year``, ``sample_start_year``, ``sample_end_year``, ``clear`` and
    ``cat_economic``; the three year columns are cast to Int64 and rows are sorted by
    ``signalname``.
    """
    validate_frame(metadata, PREDICTORS_SCHEMA)
    return metadata.select(
        pl.col("signalname"),
        pl.col("authors"),
        pl.col("journal"),
        pl.col("pub_year").cast(pl.Int64),
        pl.col("sample_start_year").cast(pl.Int64),
        pl.col("sample_end_year").cast(pl.Int64),
        pl.col("clear"),
        pl.col("cat_economic"),
    ).sort("signalname")


def predictor_event_time(panel: pl.DataFrame) -> pl.DataFrame:
    """Per predictor, anchor and event year: mean long-short return and month count.

    *panel* is a panel per :data:`anomaly_decay.schemas.PANEL_SCHEMA` (only the columns
    ``signalname``, ``portfolio``, ``ret``, ``event_time_sample_end`` and
    ``event_time_pub`` are used; extra columns are ignored). Only the ``op`` portfolio
    is used, and only event months in :data:`EVENT_MONTH_LO`…:data:`EVENT_MONTH_HI`.

    Output per ``RESULTS_SCHEMAS["predictor_event_time.csv"]``: per
    (``signalname``, ``anchor``, ``event_year``) the mean ``ret`` in percent per month
    and ``n_months``, the number of months with a non-null return in that event year.
    The event year is ``floor(event_month / 12)``. A predictor with no usable months for
    an anchor simply has no rows for it.
    """
    op = panel.filter(pl.col("portfolio") == "op")
    parts: list[pl.DataFrame] = []
    for anchor, column in _ANCHORS:
        if column not in op.columns:
            continue
        parts.append(
            op.filter(
                pl.col(column).is_not_null()
                & (pl.col(column) >= EVENT_MONTH_LO)
                & (pl.col(column) <= EVENT_MONTH_HI)
                & pl.col("ret").is_not_null()
            )
            .select(
                pl.col("signalname"),
                pl.lit(anchor).alias("anchor"),
                pl.col(column).floordiv(12).cast(pl.Int64).alias("event_year"),
                pl.col("ret").cast(pl.Float64),
            )
            .group_by("signalname", "anchor", "event_year")
            .agg(
                pl.col("ret").mean().cast(pl.Float64).alias("mean_ret"),
                pl.len().cast(pl.Int64).alias("n_months"),
            )
        )
    if not parts:
        return pl.DataFrame(schema=RESULTS_SCHEMAS["predictor_event_time.csv"])
    return pl.concat(parts).select(_EVENT_TIME_COLUMNS).sort(["signalname", "anchor", "event_year"])


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the two per-predictor profiles from a panel and predictor metadata."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.predictor_profiles",
        description="Write predictor information and event-year results.",
    )
    parser.add_argument("--panel", type=Path, required=True, help="panel.parquet path")
    parser.add_argument("--metadata", type=Path, required=True, help="predictors.parquet path")
    parser.add_argument("--results-dir", type=Path, required=True, help="results output directory")
    args = parser.parse_args(argv)

    panel = pl.read_parquet(args.panel)
    metadata = pl.read_parquet(args.metadata)
    write_results(predictor_info(metadata), "predictor_info.csv", args.results_dir)
    write_results(predictor_event_time(panel), "predictor_event_time.csv", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
