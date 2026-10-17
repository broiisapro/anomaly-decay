"""Tests for the results figures: structure and byte-level determinism."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest
from anomaly_decay.figures.event_time import build_event_time_figure, main, plot_event_time
from anomaly_decay.schemas import RESULTS_SCHEMAS

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "results_fixture"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_RENDER_EVENT_TIME = """
import sys
from pathlib import Path

import polars as pl
from anomaly_decay.figures.event_time import plot_event_time
from anomaly_decay.schemas import RESULTS_SCHEMAS

results_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
frame = pl.read_csv(results_dir / "event_time.csv", schema=RESULTS_SCHEMAS["event_time.csv"])
plot_event_time(frame, out_dir)
"""


@pytest.fixture(scope="module")
def event_time_frame() -> pl.DataFrame:
    return pl.read_csv(
        RESULTS_FIXTURE / "event_time.csv",
        schema=RESULTS_SCHEMAS["event_time.csv"],
    )


def _render_event_time_subprocess(results_dir: Path, out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, "-c", _RENDER_EVENT_TIME, str(results_dir), str(out_dir)],
        check=True,
        cwd=REPO_ROOT,
    )


def test_plot_event_time_writes_png_and_svg(tmp_path: Path, event_time_frame: pl.DataFrame) -> None:
    paths = plot_event_time(event_time_frame, tmp_path / "figures")
    assert [path.name for path in paths] == ["event_time.png", "event_time.svg"]
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert paths[0].read_bytes().startswith(PNG_MAGIC)
    svg = paths[1].read_text(encoding="utf-8")
    assert svg.startswith("<?xml")
    assert "<svg" in svg


def test_event_time_figure_has_two_panels_with_full_event_window(
    event_time_frame: pl.DataFrame,
) -> None:
    fig = build_event_time_figure(event_time_frame)
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
    first = plot_event_time(event_time_frame, tmp_path / "first")
    second = plot_event_time(event_time_frame, tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_event_time_render_is_byte_identical_across_processes(
    tmp_path: Path, event_time_frame: pl.DataFrame
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _render_event_time_subprocess(RESULTS_FIXTURE, first)
    _render_event_time_subprocess(RESULTS_FIXTURE, second)
    here = plot_event_time(event_time_frame, tmp_path / "here")
    for name, local in (
        ("event_time.png", here[0]),
        ("event_time.svg", here[1]),
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).read_bytes() == local.read_bytes()


def test_cli_writes_into_the_results_figures_directory(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    shutil.copytree(RESULTS_FIXTURE, results_dir)
    assert main(["--results-dir", str(results_dir)]) == 0
    for name in ("event_time.png", "event_time.svg"):
        path = results_dir / "figures" / name
        assert path.is_file()
        assert path.stat().st_size > 0
