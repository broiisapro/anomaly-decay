"""Event-time figure: average long-short return around sample end and publication.

Interface stub: implemented in Epic 01 workstream C.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

__all__ = ["main", "plot_event_time"]


def plot_event_time(event_time: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot average return in event time with confidence bands.

    Input: *event_time* per ``RESULTS_SCHEMAS["event_time.csv"]`` — anchor
    ``sample_end`` or ``publication``, event months -60..+120, ``mean_ret`` and ``se``
    in **percent per month**, one row per (anchor, event_month). Event month 0 is the
    first month after the anchor (window rules per architecture §4.6).

    Output: files written under *out_dir* (``event_time.png`` and ``event_time.svg``),
    returned as paths.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.event_time --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.event_time",
        description="Plot average returns in event time around sample end and publication.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = pl.read_csv(args.results_dir / "event_time.csv")
    plot_event_time(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
