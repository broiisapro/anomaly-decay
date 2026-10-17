"""Tests for the results figures: structure and byte-level determinism."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest
from anomaly_decay.figures import coefplot, distribution, event_time
from anomaly_decay.io import read_results
from matplotlib.container import ErrorbarContainer

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "results_fixture"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_RENDER_ALL_FIGURES = """
import sys
from pathlib import Path

from anomaly_decay.figures import coefplot, distribution, event_time
from anomaly_decay.io import read_results

results_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
event_time.plot_event_time(read_results("event_time.csv", results_dir), out_dir)
distribution.plot_distribution(read_results("predictor_summary.csv", results_dir), out_dir)
coefplot.plot_coefficients(read_results("robustness.csv", results_dir), out_dir)
"""


@pytest.fixture(scope="module")
def event_time_frame() -> pl.DataFrame:
    return read_results("event_time.csv", RESULTS_FIXTURE)


@pytest.fixture(scope="module")
def predictor_summary_frame() -> pl.DataFrame:
    return read_results("predictor_summary.csv", RESULTS_FIXTURE)


@pytest.fixture(scope="module")
def robustness_frame() -> pl.DataFrame:
    return read_results("robustness.csv", RESULTS_FIXTURE)


def _render_all_in_subprocess(results_dir: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-c", _RENDER_ALL_FIGURES, str(results_dir), str(out_dir)],
        check=True,
        cwd=REPO_ROOT,
    )


def _assert_png_and_svg(paths: list[Path], stem: str) -> None:
    assert [path.name for path in paths] == [f"{stem}.png", f"{stem}.svg"]
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert paths[0].read_bytes().startswith(PNG_MAGIC)
    svg = paths[1].read_text(encoding="utf-8")
    assert svg.startswith("<?xml")
    assert "<svg" in svg


def test_plot_event_time_writes_png_and_svg(tmp_path: Path, event_time_frame: pl.DataFrame) -> None:
    _assert_png_and_svg(
        event_time.plot_event_time(event_time_frame, tmp_path / "figures"), "event_time"
    )


def test_event_time_figure_has_two_panels_with_full_event_window(
    event_time_frame: pl.DataFrame,
) -> None:
    fig = event_time.build_event_time_figure(event_time_frame)
    try:
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == [
            "Around the sample end",
            "Around publication",
        ]
        assert fig.axes[0].get_ylabel() == "Long-short return (% per month)"
        for ax in fig.axes:
            months = ax.lines[0].get_xdata()
            assert min(months) == -60
            assert max(months) == 120
            legend = ax.get_legend()
            assert legend is not None
            labels = [text.get_text() for text in legend.get_texts()]
            assert any("95% confidence band" in label for label in labels)
            assert any("12-month centred moving average" in label for label in labels)
            vertical = [line for line in ax.lines if line.get_xdata()[0] == 0.0]
            horizontal = [line for line in ax.lines if line.get_ydata()[0] == 0.0]
            assert ax.collections
            assert vertical
            assert horizontal
    finally:
        plt.close(fig)


def test_event_time_render_is_byte_identical_within_process(
    tmp_path: Path, event_time_frame: pl.DataFrame
) -> None:
    first = event_time.plot_event_time(event_time_frame, tmp_path / "first")
    second = event_time.plot_event_time(event_time_frame, tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_every_figure_is_byte_identical_across_processes(
    tmp_path: Path,
    event_time_frame: pl.DataFrame,
    predictor_summary_frame: pl.DataFrame,
    robustness_frame: pl.DataFrame,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    here = tmp_path / "here"
    _render_all_in_subprocess(RESULTS_FIXTURE, first)
    _render_all_in_subprocess(RESULTS_FIXTURE, second)
    event_time.plot_event_time(event_time_frame, here)
    distribution.plot_distribution(predictor_summary_frame, here)
    coefplot.plot_coefficients(robustness_frame, here)
    for stem in ("event_time", "distribution", "coefplot"):
        for suffix in (".png", ".svg"):
            filename = f"{stem}{suffix}"
            assert (first / filename).read_bytes() == (second / filename).read_bytes(), filename
            assert (first / filename).read_bytes() == (here / filename).read_bytes(), filename


def test_event_time_cli_writes_into_the_results_figures_directory(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    shutil.copytree(RESULTS_FIXTURE, results_dir)
    assert event_time.main(["--results-dir", str(results_dir)]) == 0
    for name in ("event_time.png", "event_time.svg"):
        path = results_dir / "figures" / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_plot_distribution_writes_png_and_svg(
    tmp_path: Path, predictor_summary_frame: pl.DataFrame
) -> None:
    _assert_png_and_svg(
        distribution.plot_distribution(predictor_summary_frame, tmp_path / "figures"),
        "distribution",
    )


def test_distribution_figure_pairs_scatter_with_shared_histogram(
    predictor_summary_frame: pl.DataFrame,
) -> None:
    fig = distribution.build_distribution_figure(predictor_summary_frame)
    try:
        assert len(fig.axes) == 2
        scatter_ax, histogram_ax = fig.axes
        assert len(scatter_ax.collections) == 1
        offsets = scatter_ax.collections[0].get_offsets()
        assert len(offsets) == predictor_summary_frame["signalname"].n_unique()

        paired = (
            predictor_summary_frame.filter(
                pl.col("window").is_in(["in_sample", "post_publication"])
            )
            .pivot(on="window", index="signalname", values="mean_ret")
            .drop_nulls(["in_sample", "post_publication"])
            .sort("signalname")
        )
        expected = list(
            zip(
                paired["in_sample"].to_list(),
                paired["post_publication"].to_list(),
                strict=True,
            )
        )
        assert [tuple(point) for point in offsets.tolist()] == expected

        legend = scatter_ax.get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert "45° line" in labels
        assert "Zero line" in labels
        diagonal = next(line for line in scatter_ax.lines if line.get_label() == "45° line")
        assert list(diagonal.get_xdata()) == list(diagonal.get_ydata())
        assert scatter_ax.get_xlabel() == "In-sample mean return (% per month)"
        assert scatter_ax.get_ylabel() == "Post-publication mean return (% per month)"

        assert histogram_ax.patches
        histogram_legend = histogram_ax.get_legend()
        assert histogram_legend is not None
        histogram_labels = [text.get_text() for text in histogram_legend.get_texts()]
        assert histogram_labels == ["In-sample", "Post-publication"]
    finally:
        plt.close(fig)


def test_plot_coefficients_writes_png_and_svg(
    tmp_path: Path, robustness_frame: pl.DataFrame
) -> None:
    _assert_png_and_svg(
        coefplot.plot_coefficients(robustness_frame, tmp_path / "figures"), "coefplot"
    )


def test_coefficient_figure_uses_result_labels_with_reference_first(
    robustness_frame: pl.DataFrame,
) -> None:
    fig = coefplot.build_coefficient_figure(robustness_frame)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        tick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        labels = robustness_frame["label"].unique(maintain_order=True).to_list()
        reference_label = robustness_frame.filter(pl.col("spec") == "main")["label"].unique().item()
        assert tick_labels[0] == reference_label
        assert sorted(tick_labels) == sorted(labels)

        legend = ax.get_legend()
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert legend_labels == ["Post-sample", "Post-publication"]
        containers = [
            container for container in ax.containers if isinstance(container, ErrorbarContainer)
        ]
        assert len(containers) == 2
        for container in containers:
            assert len(container.lines[0].get_xdata()) == len(labels)
        assert any(all(x == 0.0 for x in line.get_xdata()) for line in ax.lines)
    finally:
        plt.close(fig)


def test_distribution_cli_writes_into_the_results_figures_directory(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    shutil.copytree(RESULTS_FIXTURE, results_dir)
    assert distribution.main(["--results-dir", str(results_dir)]) == 0
    for name in ("distribution.png", "distribution.svg"):
        assert (results_dir / "figures" / name).is_file()


def test_coefplot_cli_writes_into_the_results_figures_directory(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    shutil.copytree(RESULTS_FIXTURE, results_dir)
    assert coefplot.main(["--results-dir", str(results_dir)]) == 0
    for name in ("coefplot.png", "coefplot.svg"):
        assert (results_dir / "figures" / name).is_file()
