"""Merge returns, metadata and factors into the panel.

Interface stub: implemented in Epic 01 workstream A.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

__all__ = ["build_panel", "main"]


def build_panel(data_dir: Path, metadata: pl.DataFrame, portfolios: Sequence[str]) -> pl.DataFrame:
    """Build ``panel.parquet`` from canonical raw returns, metadata and pinned factors.

    Inputs: *data_dir* holds canonical OSAP CSVs (``<vintage>/port_<name>.csv``) and
    pinned Ken French monthly CSVs; *metadata* is the predictor table per
    :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`; *portfolios* selects long-short
    series (``op``, ``deciles_ew``, ``deciles_vw``; rows with ``port == "LS"``).

    Output: a frame exactly per :data:`anomaly_decay.schemas.PANEL_SCHEMA`, keyed by
    (``signalname``, ``portfolio``, ``yyyymm``) and sorted by that key. Only months
    classified ``in_sample``, ``post_sample`` or ``post_publication`` are kept
    (pre-sample months are dropped), so every row is in exactly one window. Join all
    sources on ``yyyymm`` (never the raw OSAP ``date``); ``month`` is the first day of
    the month. Returns are **percent per month** (no conversion). ``is_mean`` is the
    predictor×portfolio in-sample mean of ``ret``; ``ret_norm`` is ``ret / is_mean``,
    null when ``is_mean <= 0``. Factor columns are null where Ken French coverage is
    absent. Window flags follow architecture §4.6 under both conventions (``default``
    and ``alt``).
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.panel.build --data-dir DIR --metadata PATH --out PATH``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.panel.build",
        description="Build the panel from canonical raw data, metadata and factors.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="raw data directory")
    parser.add_argument("--metadata", type=Path, required=True, help="predictors.parquet path")
    parser.add_argument("--out", type=Path, required=True, help="panel.parquet path")
    parser.add_argument(
        "--portfolios", default="op", help="comma-separated portfolio names (default: op)"
    )
    args = parser.parse_args(argv)
    metadata = pl.read_parquet(args.metadata)
    frame = build_panel(args.data_dir, metadata, tuple(args.portfolios.split(",")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
