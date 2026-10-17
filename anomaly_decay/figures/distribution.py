"""Distribution figure: per-predictor in-sample vs post-publication mean returns."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from anomaly_decay.figures.style import PALETTE, apply_style, save_figure
from anomaly_decay.io import read_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

__all__ = ["build_distribution_figure", "main", "plot_distribution"]

IN_SAMPLE = "in_sample"
POST_PUBLICATION = "post_publication"
_HISTOGRAM_BINS = 15


def _display_window(window: str) -> str:
    return window.replace("_", "-").capitalize()


def _paired_means(predictor_summary: pl.DataFrame) -> pl.DataFrame:
    means = (
        predictor_summary.filter(pl.col("window").is_in([IN_SAMPLE, POST_PUBLICATION]))
        .pivot(on="window", index="signalname", values="mean_ret")
        .drop_nulls([IN_SAMPLE, POST_PUBLICATION])
        .sort("signalname")
    )
    if means.is_empty():
        raise ValueError(
            "predictor_summary has no predictor present in both in-sample and post-publication"
        )
    return means


def build_distribution_figure(predictor_summary: pl.DataFrame) -> Figure:
    """Build the scatter and paired-histogram figure from a ``predictor_summary`` frame."""
    validate_frame(predictor_summary, RESULTS_SCHEMAS["predictor_summary.csv"])
    paired = _paired_means(predictor_summary)
    in_sample = paired[IN_SAMPLE].to_list()
    post_publication = paired[POST_PUBLICATION].to_list()
    in_label = _display_window(IN_SAMPLE)
    post_label = _display_window(POST_PUBLICATION)

    apply_style()
    fig, (scatter_ax, histogram_ax) = plt.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained")

    low = min(min(in_sample), min(post_publication))
    high = max(max(in_sample), max(post_publication))
    padding = 0.05 * max(high - low, 1e-9)
    axis_low, axis_high = low - padding, high + padding
    scatter_ax.plot(
        [axis_low, axis_high],
        [axis_low, axis_high],
        color="0.5",
        linewidth=1.0,
        linestyle="--",
        label="45° line",
    )
    scatter_ax.scatter(
        in_sample,
        post_publication,
        s=22,
        color=PALETTE[0],
        alpha=0.8,
        edgecolors="none",
        label="Predictors",
    )
    scatter_ax.axhline(0.0, color="0.4", linewidth=1.0, label="Zero line")
    scatter_ax.set_xlim(axis_low, axis_high)
    scatter_ax.set_ylim(axis_low, axis_high)
    scatter_ax.set_xlabel(f"{in_label} mean return (% per month)")
    scatter_ax.set_ylabel(f"{post_label} mean return (% per month)")
    scatter_ax.set_title("Per-predictor means")
    scatter_ax.legend(loc="upper left", frameon=False)

    histogram_ax.hist(
        [in_sample, post_publication],
        bins=_HISTOGRAM_BINS,
        color=[PALETTE[0], PALETTE[1]],
        alpha=0.8,
        label=[in_label, post_label],
    )
    histogram_ax.axvline(0.0, color="0.4", linewidth=1.0)
    histogram_ax.set_xlabel("Mean return (% per month)")
    histogram_ax.set_ylabel("Number of predictors")
    histogram_ax.set_title("Distribution across predictors")
    histogram_ax.legend(loc="best", frameon=False)
    return fig


def plot_distribution(predictor_summary: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot the distribution of per-predictor mean returns by window.

    Input: *predictor_summary* per ``RESULTS_SCHEMAS["predictor_summary.csv"]`` —
    one row per (``signalname``, ``window``), ``mean_ret`` and ``se_mean`` in
    **percent per month**; window is one of ``in_sample``, ``post_sample``,
    ``post_publication`` (default convention, architecture §4.6). The figure compares
    the in-sample with the post-publication means for every predictor that has both.

    Output: files written under *out_dir* (``distribution.png`` and
    ``distribution.svg``), returned as paths.
    """
    fig = build_distribution_figure(predictor_summary)
    try:
        return save_figure(fig, out_dir, "distribution")
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.distribution --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.distribution",
        description="Plot per-predictor mean returns in-sample vs post-publication.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = read_results("predictor_summary.csv", args.results_dir)
    plot_distribution(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
