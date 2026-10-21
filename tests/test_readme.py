"""Tests for the generated README results block (issue #93).

The README must never carry a typed number: :mod:`anomaly_decay.readme` renders the block
from the committed ``results/`` files with the report's formatting helpers, and ``--check``
fails when the block is stale or when a number appears outside it. The tests exercise the
check against the committed README, the write path on a temporary copy, and a mutation per
inputs file so a silently ignored result cannot pass.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay import readme, reporting
from anomaly_decay.io import read_results, write_results

README_PATH = reporting.REPO_ROOT / "README.md"
RESULTS_DIR = reporting.REPO_ROOT / "results"
BLOCK_FILES = ("main.csv", "decay.csv", "coverage.csv", "bootstrap.csv")

OPENING_PARAGRAPH = (
    "Do stock-return anomalies survive being published? McLean and Pontiff (2016) found that "
    "long-short returns fall after a predictor's original sample ends and fall further after "
    "publication. This repository replicates that result with the open Chen–Zimmermann data, "
    "extends it through the latest data vintage, and documents every specification and "
    "deviation. `make reproduce` rebuilds every number in the report from scratch."
)


def committed_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def block_of(text: str) -> str:
    start = text.index(readme.START_MARKER)
    end = text.index(readme.END_MARKER) + len(readme.END_MARKER)
    return text[start:end]


@pytest.fixture()
def results_copy(tmp_path: Path) -> Path:
    """A writable copy of the four results files the block reads."""
    target = tmp_path / "results"
    target.mkdir()
    for name in BLOCK_FILES:
        shutil.copyfile(RESULTS_DIR / name, target / name)
    return target


def mutate(results_dir: Path, name: str, frame: pl.DataFrame) -> None:
    write_results(frame, name, results_dir)


def test_committed_block_matches_the_rendered_results() -> None:
    text = committed_text()
    assert text.count(readme.START_MARKER) == 1
    assert text.count(readme.END_MARKER) == 1
    assert block_of(text) == readme.render_block()


def test_check_passes_on_the_committed_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert readme.main(["--check"]) == 0
    assert "is up to date" in capsys.readouterr().out


def test_check_rejects_a_stale_block(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme_path = tmp_path / "README.md"
    stale = committed_text().replace("49.5%", "50.0%", 1)
    readme_path.write_text(stale, encoding="utf-8")

    assert readme.main(["--check", "--readme", str(readme_path)]) == 1
    assert "out of date" in capsys.readouterr().err


def test_write_restores_a_stale_block_and_leaves_prose_alone(tmp_path: Path) -> None:
    readme_path = tmp_path / "README.md"
    stale_block = f"{readme.START_MARKER}\nstale\n{readme.END_MARKER}"
    readme_path.write_text(readme.replace_block(committed_text(), stale_block), encoding="utf-8")

    assert readme.main(["--readme", str(readme_path)]) == 0
    assert readme_path.read_text(encoding="utf-8") == committed_text()


def test_check_rejects_a_typed_number_outside_the_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        committed_text() + "\nThe sample has 198 predictors.\n", encoding="utf-8"
    )

    assert readme.main(["--check", "--readme", str(readme_path)]) == 1
    assert "numbers outside the results block" in capsys.readouterr().err


def test_find_stray_numbers_accepts_years_and_code() -> None:
    text = (
        "McLean and Pontiff (2016) use Python `3.12` and `make phase2`.\n\n"
        "```bash\nuv run python -m anomaly_decay.readme\n```\n"
    )
    assert readme.find_stray_numbers(text) == []


def test_find_stray_numbers_flags_typed_results() -> None:
    assert readme.find_stray_numbers("Returns fall by 49.5% in 2016.") == ["49", "5"]


def test_find_stray_numbers_ignores_the_generated_block() -> None:
    text = f"# x\n\n{readme.render_block()}\n"
    assert readme.find_stray_numbers(text) == []


def test_check_requires_one_pair_of_markers() -> None:
    problems = readme.check_readme("# no markers", "block")
    assert len(problems) == 1
    assert "exactly one" in problems[0]


def test_render_block_reflects_a_changed_decay(tmp_path: Path, results_copy: Path) -> None:
    decay = read_results("decay.csv", results_copy).with_columns(
        pl.when(
            (pl.col("spec") == "main")
            & (pl.col("method") == "pooled_ratio")
            & (pl.col("term") == "post_publication")
        )
        .then(pl.lit(-77.7))
        .otherwise(pl.col("decay_pct"))
        .alias("decay_pct")
    )
    mutate(results_copy, "decay.csv", decay)

    block = readme.render_block(results_copy)
    assert "77.7%" in block
    assert "49.5%" not in block


def test_render_block_reflects_a_changed_bootstrap_interval(
    tmp_path: Path, results_copy: Path
) -> None:
    bootstrap = read_results("bootstrap.csv", results_copy).with_columns(
        pl.when(pl.col("statistic") == "decay_pct_post_publication")
        .then(pl.lit(-90.0))
        .otherwise(pl.col("ci_low"))
        .alias("ci_low")
    )
    mutate(results_copy, "bootstrap.csv", bootstrap)

    block = readme.render_block(results_copy)
    assert "[-90.0%" in block


def test_render_block_reflects_changed_coverage(tmp_path: Path, results_copy: Path) -> None:
    coverage = read_results("coverage.csv", results_copy).with_columns(
        pl.when(pl.col("item") == "n_predictors_op")
        .then(pl.lit("999"))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    mutate(results_copy, "coverage.csv", coverage)

    block = readme.render_block(results_copy)
    assert "999 predictors in total" in block


def test_render_block_reports_missing_results(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        readme.render_block(tmp_path)


def test_readme_opening_paragraph_is_the_release_text() -> None:
    paragraphs = committed_text().split("\n\n")
    assert paragraphs[0] == "# anomaly-decay"
    assert paragraphs[1].split() == OPENING_PARAGRAPH.split()


def test_readme_has_the_release_sections() -> None:
    text = committed_text()
    assert "![Event-time long-short returns around publication]" in text
    assert "results/figures/event_time.png" in text
    for command in (
        "uv sync --frozen",
        "make reproduce-fixture",
        "make reproduce",
        "make phase2",
    ):
        assert command in text
    assert "https://broiisapro.github.io/anomaly-decay/" in text
    assert "available after release" in text
    assert "docs/architecture.md" in text
    assert "MIT licence" in text
    assert "does not redistribute raw data" in text
    assert "## Limitations" in text
