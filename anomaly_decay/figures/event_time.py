"""Event-time figure: average long-short return around sample end and publication."""

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

__all__ = ["build_event_time_figure", "main", "plot_event_time"]

ANCHORS: tuple[str, ...] = ("sample_end", "publication")
"""Anchor values in ``event_time.csv``; both get their own panel."""

_ANCHOR_TITLES = {
    "sample_end": "Around the sample end",
    "publication": "Around publication",
}
_CI_MULTIPLIER = 1.96
_MOVING_AVERAGE_MONTHS = 12
_Y_LABEL = "Long-short return (% per month)"


def build_event_time_figure(event_time: pl.DataFrame) -> Figure:
    """Build the two-panel event-time figure from an ``event_time.csv`` frame."""
    validate_frame(event_time, RESULTS_SCHEMAS["event_time.csv"])
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True, layout="constrained")
    for ax, anchor in zip(axes, ANCHORS, strict=True):
        anchor_frame = event_time.filter(pl.col("anchor") == anchor).sort("event_month")
        if anchor_frame.is_empty():
            raise ValueError(f"event_time frame has no rows for anchor {anchor!r}")
        months = anchor_frame["event_month"].to_numpy()
        mean = anchor_frame["mean_ret"].to_numpy()
        se = anchor_frame["se"].to_numpy()
        moving_average = (
            anchor_frame["mean_ret"]
            .rolling_mean(
                window_size=_MOVING_AVERAGE_MONTHS,
                center=True,
                min_samples=_MOVING_AVERAGE_MONTHS,
            )
            .to_numpy()
        )
        ax.fill_between(
            months,
            mean - _CI_MULTIPLIER * se,
            mean + _CI_MULTIPLIER * se,
            color=PALETTE[0],
            alpha=0.20,
            linewidth=0,
            label="95% confidence band",
        )
        ax.plot(months, mean, color=PALETTE[0], linewidth=1.2, label="Mean return")
        ax.plot(
            months,
            moving_average,
            color=PALETTE[1],
            linewidth=1.6,
            label=f"{_MOVING_AVERAGE_MONTHS}-month centred moving average",
        )
        ax.axvline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        ax.axhline(0.0, color="0.4", linewidth=1.0)
        ax.set_title(_ANCHOR_TITLES[anchor])
        ax.set_xlabel("Event month")
        ax.legend(loc="best", frameon=False)
    axes[0].set_ylabel(_Y_LABEL)
    return fig


def plot_event_time(event_time: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot average return in event time with confidence bands.

    Input: *event_time* per ``RESULTS_SCHEMAS["event_time.csv"]`` — anchor
    ``sample_end`` or ``publication``, event months -60..+120, ``mean_ret`` and ``se``
    in **percent per month**, one row per (anchor, event_month). Event month 0 is the
    first month after the anchor (window rules per architecture §4.6).

    Output: files written under *out_dir* (``event_time.png`` and ``event_time.svg``),
    returned as paths.
    """
    fig = build_event_time_figure(event_time)
    try:
        return save_figure(fig, out_dir, "event_time")
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.event_time --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.event_time",
        description="Plot average returns in event time around sample end and publication.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = read_results("event_time.csv", args.results_dir)
    plot_event_time(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
