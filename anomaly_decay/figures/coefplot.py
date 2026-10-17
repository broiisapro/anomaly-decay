"""Coefficient plot of the robustness rows.

Interface stub: implemented in Epic 01 workstream C.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

__all__ = ["main", "plot_coefficients"]


def plot_coefficients(robustness: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot post-sample and post-publication coefficients across robustness specs.

    Input: *robustness* per ``RESULTS_SCHEMAS["robustness.csv"]`` — ``coef``/``se`` in
    **percent per month**, terms ``post_sample`` and ``post_publication``, one row per
    (``spec``, ``term``); window rules per architecture §4.6.

    Output: files written under *out_dir* (``coefplot.png`` and ``coefplot.svg``),
    returned as paths.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.coefplot --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.coefplot",
        description="Plot robustness coefficients with confidence intervals.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = pl.read_csv(args.results_dir / "robustness.csv")
    plot_coefficients(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
