"""Opt-in full-data reproduction of every committed results CSV.

The module is skipped with a clear reason unless both canonical OSAP caches exist:
``data/raw/202510/`` (primary vintage) and ``data/raw/202410/`` (the pre-registered
comparison release). Populate them with ``make fetch`` and
``uv run python -m anomaly_decay.data.fetch --osap --vintage 202410``.

The test rebuilds the metadata and the panel from the cache, runs the estimate, the
descriptive, the per-predictor profile and the three Phase 2 CLIs into a temporary
results directory, and compares every committed ``results/*.csv`` with its fresh copy.
On macOS the comparison is byte for byte; elsewhere floats are compared at 1e-10
relative/absolute and text exactly, because the linear-algebra libraries differ across
operating systems (decision O34).

Subprocesses run with the Makefile's thread environment (OpenMP/BLAS pools pinned to one
thread). ``POLARS_MAX_THREADS`` is removed from the inherited environment: the committed
files were produced with Polars' default thread pool, and its reductions (for example
the pooled in-sample mean behind ``decay_pct``) change summation order with the thread
count, which moves the last digit of some decay percentages.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.io import read_results
from polars.testing import assert_frame_equal

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
RAW_DIR = REPO_ROOT / "data" / "raw"
PINNED_DIR = REPO_ROOT / "data" / "pinned"
SPECS = REPO_ROOT / "config" / "specifications.yaml"
PREREG = REPO_ROOT / "config" / "phase2_preregistration.yaml"
SUPPLEMENT = REPO_ROOT / "config" / "predictor_dates_supplement.csv"
PRIMARY_VINTAGE = 202510
COMPARISON_VINTAGE = 202410
PORTFOLIOS = "op,deciles_ew,deciles_vw"

COMMITTED_FILES = tuple(sorted(path.name for path in RESULTS_DIR.glob("*.csv")))
"""Every committed results table, in stable order."""

COMPARISON_MODE = "bytes" if sys.platform == "darwin" else "numeric"
"""Per-platform comparison: byte-identical on macOS, 1e-10 numeric elsewhere."""

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (RAW_DIR / str(PRIMARY_VINTAGE)).is_dir()
        or not (RAW_DIR / str(COMPARISON_VINTAGE)).is_dir(),
        reason=(
            "full reproduction needs the OSAP 202510 and 202410 caches under data/raw/ "
            "(run make fetch and fetch --osap --vintage 202410)"
        ),
    ),
]


def _subprocess_environment() -> dict[str, str]:
    """The Makefile's thread pins, with Polars left at its default thread pool."""
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    environment.pop("POLARS_MAX_THREADS", None)
    return environment


def _run(module: str, *arguments: object) -> None:
    """Run one CLI module in a subprocess, failing with its output on a non-zero exit."""
    command = [sys.executable, "-m", module, *(str(argument) for argument in arguments)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{module} exited with status {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


@pytest.fixture(scope="module")
def rebuilt_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Rebuild metadata, panel and all results from the cache; return the results dir."""
    root = tmp_path_factory.mktemp("full-reproduction")
    metadata = root / "predictors.parquet"
    panel = root / "panel.parquet"
    results = root / "results"
    results.mkdir()
    signal_doc = RAW_DIR / str(PRIMARY_VINTAGE) / "signal_doc.csv"

    _run(
        "anomaly_decay.data.metadata",
        "--signal-doc",
        signal_doc,
        "--supplement",
        SUPPLEMENT,
        "--out",
        metadata,
    )
    _run(
        "anomaly_decay.panel.build",
        "--data-dir",
        RAW_DIR,
        "--pinned-dir",
        PINNED_DIR,
        "--vintage",
        PRIMARY_VINTAGE,
        "--metadata",
        metadata,
        "--portfolios",
        PORTFOLIOS,
        "--out",
        panel,
    )
    _run(
        "anomaly_decay.estimate.main",
        "--panel",
        panel,
        "--specs",
        SPECS,
        "--results-dir",
        results,
        "--print-headline",
    )
    _run(
        "anomaly_decay.estimate.descriptive",
        "--panel",
        panel,
        "--metadata",
        metadata,
        "--vintage",
        PRIMARY_VINTAGE,
        "--results-dir",
        results,
    )
    _run(
        "anomaly_decay.estimate.predictor_profiles",
        "--panel",
        panel,
        "--metadata",
        metadata,
        "--results-dir",
        results,
    )
    _run(
        "anomaly_decay.estimate.bootstrap",
        "--panel",
        panel,
        "--prereg",
        PREREG,
        "--specs",
        SPECS,
        "--results-dir",
        results,
    )
    _run(
        "anomaly_decay.estimate.multiple_testing",
        "--panel",
        panel,
        "--signal-doc",
        signal_doc,
        "--prereg",
        PREREG,
        "--specs",
        SPECS,
        "--results-dir",
        results,
    )
    _run(
        "anomaly_decay.estimate.vintages",
        "--vintages",
        f"{COMPARISON_VINTAGE},{PRIMARY_VINTAGE}",
        "--data-dir",
        RAW_DIR,
        "--pinned-dir",
        PINNED_DIR,
        "--specs",
        SPECS,
        "--results-dir",
        results,
    )
    return results


def _first_difference(committed: Path, rebuilt: Path) -> str:
    """A short description of the first differing line of two text files."""
    committed_lines = committed.read_text().splitlines()
    rebuilt_lines = rebuilt.read_text().splitlines()
    for number, (expected, actual) in enumerate(
        zip(committed_lines, rebuilt_lines, strict=False), start=1
    ):
        if expected != actual:
            return (
                f"first difference at line {number}:\n"
                f"  committed: {expected}\n"
                f"  rebuilt:   {actual}"
            )
    return (
        "the files differ outside their line content (line endings or encoding); "
        f"committed {len(committed_lines)} lines, rebuilt {len(rebuilt_lines)} lines"
    )


def _assert_frames_match(name: str, expected: pl.DataFrame, actual: pl.DataFrame) -> None:
    """Compare one results table with the cross-platform tolerance (O34)."""
    try:
        assert_frame_equal(
            expected,
            actual,
            check_row_order=True,
            check_column_order=True,
            check_dtypes=True,
            check_exact=False,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{name}: rebuilt file differs from the committed copy "
            f"(numeric comparison at 1e-10)\n{exc}"
        ) from exc


def test_rebuild_covers_exactly_the_committed_files(rebuilt_results: Path) -> None:
    """The pipeline's CSV output set matches `results/*.csv`, with nothing extra."""
    rebuilt = tuple(sorted(path.name for path in rebuilt_results.glob("*.csv")))
    assert rebuilt == COMMITTED_FILES


@pytest.mark.parametrize("name", COMMITTED_FILES)
def test_committed_results_file_reproduces(name: str, rebuilt_results: Path) -> None:
    """Every committed results file is reproduced from the raw cache."""
    committed = RESULTS_DIR / name
    rebuilt = rebuilt_results / name
    assert rebuilt.is_file(), f"{name}: the pipeline did not write this file"
    if COMPARISON_MODE == "bytes":
        if committed.read_bytes() != rebuilt.read_bytes():
            raise AssertionError(
                f"{name}: rebuilt file is not byte-identical to the committed copy\n"
                f"{_first_difference(committed, rebuilt)}"
            )
    else:
        expected = read_results(name, RESULTS_DIR)
        actual = read_results(name, rebuilt_results)
        _assert_frames_match(name, expected, actual)


def test_numeric_comparison_accepts_last_digit_noise_and_rejects_real_differences() -> None:
    """The non-macOS path tolerates 1e-12 float noise but not a 1e-6 change."""
    expected = pl.DataFrame({"spec": ["main"], "value": [0.123456789]})
    within_tolerance = expected.with_columns((pl.col("value") + 1e-12).alias("value"))
    _assert_frames_match("synthetic", expected, within_tolerance)
    outside_tolerance = expected.with_columns((pl.col("value") + 1e-6).alias("value"))
    with pytest.raises(AssertionError):
        _assert_frames_match("synthetic", expected, outside_tolerance)
