"""Predictor metadata table (OSAP signal doc + cited supplement).

Interface stub: implemented in Epic 01 workstream A.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

__all__ = ["build_metadata", "main"]


def build_metadata(signal_doc_path: Path, supplement_path: Path | None = None) -> pl.DataFrame:
    """Build the predictor metadata table for the pinned vintage.

    Inputs: *signal_doc_path* is the canonical OSAP signal-doc CSV
    (``data/raw/<vintage>/signal_doc.csv``); *supplement_path* optionally fills missing
    sample-period or publication fields from ``config/predictor_dates_supplement.csv``,
    one cited source per row.

    Output: one row per predictor, keyed by ``signalname``, exactly per
    :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA` (columns ``signalname,
    sample_start_year, sample_end_year, pub_year, journal, authors, cat_economic,
    cat_data, clear, date_source``). The universe is ``Cat.Signal == "Predictor"``;
    ``clear`` is ``Predictability in OP == "1_clear"``. Years are plain calendar years;
    no returns are involved, so no units apply.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.data.metadata --data-dir DIR --out PATH``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.data.metadata",
        description="Build the predictor metadata table from the OSAP signal doc.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="raw data directory")
    parser.add_argument("--out", type=Path, required=True, help="output predictors.parquet path")
    args = parser.parse_args(argv)
    frame = build_metadata(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
