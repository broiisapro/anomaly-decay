"""Distribution figure: per-predictor in-sample vs post-publication mean returns.

Interface stub: implemented in Epic 01 workstream C.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

__all__ = ["main", "plot_distribution"]


def plot_distribution(predictor_summary: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot the distribution of per-predictor mean returns by window.

    Input: *predictor_summary* per ``RESULTS_SCHEMAS["predictor_summary.csv"]`` —
    one row per (``signalname``, ``window``), ``mean_ret`` and ``se_mean`` in
    **percent per month**; window is one of ``in_sample``, ``post_sample``,
    ``post_publication`` (default convention, architecture §4.6).

    Output: files written under *out_dir* (``distribution.png`` and
    ``distribution.svg``), returned as paths.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.distribution --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.distribution",
        description="Plot per-predictor mean returns in-sample vs post-publication.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = pl.read_csv(args.results_dir / "predictor_summary.csv")
    plot_distribution(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
