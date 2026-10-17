"""Coefficient plot of the robustness rows."""

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

__all__ = ["build_coefficient_figure", "main", "plot_coefficients"]

TERMS: tuple[str, ...] = ("post_sample", "post_publication")
_REFERENCE_SPEC = "main"
_CI_MULTIPLIER = 1.96
_ROW_HEIGHT = 0.62
_TERM_OFFSET = 0.16


def _display_term(term: str) -> str:
    return term.replace("_", "-").capitalize()


def _ordered_specs(robustness: pl.DataFrame) -> list[tuple[str, str]]:
    """Return (spec, label) pairs in first-seen order, with the reference spec first."""
    order: list[str] = []
    labels: dict[str, str] = {}
    for row in robustness.iter_rows(named=True):
        if row["spec"] not in labels:
            order.append(row["spec"])
            labels[row["spec"]] = row["label"]
    reference = [spec for spec in order if spec == _REFERENCE_SPEC]
    return [(spec, labels[spec]) for spec in reference + [s for s in order if s not in reference]]


def build_coefficient_figure(robustness: pl.DataFrame) -> Figure:
    """Build the coefficient forest plot from a ``robustness.csv`` frame."""
    validate_frame(robustness, RESULTS_SCHEMAS["robustness.csv"])
    specs = _ordered_specs(robustness)
    if not specs:
        raise ValueError("robustness frame has no rows")

    apply_style()
    height = max(3.2, _ROW_HEIGHT * len(specs) + 1.3)
    fig, ax = plt.subplots(figsize=(9.0, height), layout="constrained")
    positions = list(range(len(specs) - 1, -1, -1))

    for term_index, term in enumerate(TERMS):
        rows = {
            row["spec"]: row
            for row in robustness.filter(pl.col("term") == term).iter_rows(named=True)
        }
        offset = _TERM_OFFSET if term_index == 0 else -_TERM_OFFSET
        y: list[float] = []
        coefficients: list[float] = []
        errors: list[float] = []
        for position, (spec, _) in zip(positions, specs, strict=True):
            if spec not in rows:
                continue
            y.append(position + offset)
            coefficients.append(rows[spec]["coef"])
            errors.append(_CI_MULTIPLIER * rows[spec]["se"])
        ax.errorbar(
            coefficients,
            y,
            xerr=errors,
            fmt="o",
            color=PALETTE[term_index],
            markersize=4.5,
            capsize=3,
            linewidth=1.2,
            label=_display_term(term),
        )

    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_yticks(positions, [label for _, label in specs])
    ax.set_xlabel("Coefficient (% per month), with 95% confidence interval")
    ax.set_title("Robustness specifications")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="best", frameon=False)
    return fig


def plot_coefficients(robustness: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Plot post-sample and post-publication coefficients across robustness specs.

    Input: *robustness* per ``RESULTS_SCHEMAS["robustness.csv"]`` — ``coef``/``se`` in
    **percent per month**, terms ``post_sample`` and ``post_publication``, one row per
    (``spec``, ``term``); window rules per architecture §4.6. One row per specification
    label, with the reference (``main``) specification first; labels are read from the
    results frame.

    Output: files written under *out_dir* (``coefplot.png`` and ``coefplot.svg``),
    returned as paths.
    """
    fig = build_coefficient_figure(robustness)
    try:
        return save_figure(fig, out_dir, "coefplot")
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.figures.coefplot --results-dir DIR``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.figures.coefplot",
        description="Plot robustness coefficients with confidence intervals.",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="results directory")
    args = parser.parse_args(argv)
    frame = read_results("robustness.csv", args.results_dir)
    plot_coefficients(frame, args.results_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
