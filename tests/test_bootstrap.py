"""Tests for the pre-registered moving-block bootstrap by month.

The named classic-bug test makes i.i.d. resampling impossible to pass off as a block
bootstrap: it checks that every drawn block is a contiguous run of calendar months
with no wrap and, as a control, that an i.i.d. draw over the same range fails the same
property. The remaining tests pin the estimator to ``run_spec``/``decay_table``, the
per-replicate mechanics, seed determinism, the CLI contract and a coverage sanity
check on a synthetic panel with a known effect.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from anomaly_decay.config import Spec, load_specifications
from anomaly_decay.estimate.bootstrap import (
    STATISTICS,
    BootstrapError,
    bootstrap_intervals,
    draw_month_positions,
    estimation_sample,
)
from anomaly_decay.estimate.bootstrap import (
    main as bootstrap_main,
)
from anomaly_decay.estimate.main import decay_table, run_spec
from anomaly_decay.io import read_results, write_results
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_PANEL = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"

PREREG_TEXT = """\
version: 1

bootstrap:
  id: bootstrap_main
  base_spec: main
  method: moving_block_by_month
  block_length_months: 12
  reps: 40
  seed: 7
  statistics:
    - coef_post_sample
    - coef_post_publication
    - decay_pct_post_sample
    - decay_pct_post_publication
  interval: percentile
  level: 0.95
  exploratory: false

multiple_testing:
  hurdle_t: 3.0
  weak_rule: tstat_below_hurdle
  missing_tstat: exclude_and_count
  tstat_sources:
    original_paper: "x"
    in_sample_replicated: "y"
  specs:
    - id: mt_weak_original
      base_spec: main
      tstat_source: original_paper
      extra_terms: [post_sample_x_weak, post_publication_x_weak]
      exploratory: false

vintages:
  primary: 202510
  compare_preference: [202410]
  predictor_set: intersection
  specs: [main]
  metrics: [coef]
  exploratory: false
"""


def _panel() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE_PANEL)


def _spec(spec_id: str) -> Spec:
    specs = load_specifications(SPECS_PATH)
    return next(spec for spec in specs if spec.id == spec_id)


def _write_prereg(tmp_path: Path) -> Path:
    path = tmp_path / "phase2_preregistration.yaml"
    path.write_text(PREREG_TEXT, encoding="utf-8")
    return path


def _run(sample: pl.DataFrame, **overrides: object) -> pl.DataFrame:
    arguments: dict[str, object] = {
        "spec_id": "bootstrap_main",
        "statistics": STATISTICS,
        "reps": 20,
        "seed": 2027,
        "block_length": 12,
        "level": 0.95,
    }
    arguments.update(overrides)
    return bootstrap_intervals(sample, **arguments)  # type: ignore[arg-type]


def test_resampled_months_come_in_contiguous_blocks() -> None:
    n_months, block_length = 97, 7
    positions = draw_month_positions(np.random.default_rng(65), n_months, block_length)
    assert positions.shape == (n_months,)
    assert positions.min() >= 0
    assert positions.max() <= n_months - 1

    blocks = [positions[start : start + block_length] for start in range(0, n_months, block_length)]
    assert len(blocks) == -(-n_months // block_length)
    for block in blocks:
        assert np.all(np.diff(block) == 1), f"block is not contiguous: {block}"

    # Control: an i.i.d. draw over the same months is not built from contiguous runs.
    iid = np.random.default_rng(65).integers(0, n_months, size=n_months)
    assert np.count_nonzero(np.diff(iid) == 1) < n_months // 4


def test_short_sample_is_one_block_without_wrap() -> None:
    positions = draw_month_positions(np.random.default_rng(1), n_months=5, block_length=12)
    assert positions.tolist() == [0, 1, 2, 3, 4]


def test_every_predictors_observations_move_with_their_month() -> None:
    from anomaly_decay.estimate.bootstrap import _month_row_index

    # months 1-4 with rows (post-sort) 0,1 / 2,3 / 4 / 5,6; month 3 drawn twice
    starts = np.array([0, 2, 4, 5], dtype=np.int64)
    lengths = np.array([2, 2, 1, 2], dtype=np.int64)
    positions = np.array([2, 0, 2], dtype=np.int64)
    assert _month_row_index(positions, starts, lengths).tolist() == [4, 0, 1, 4]


@pytest.mark.parametrize("spec_id", ["main", "pubconv_alt", "clear_only"])
def test_estimation_sample_matches_run_spec_counts(spec_id: str) -> None:
    panel = _panel()
    spec = _spec(spec_id)
    sample = estimation_sample(panel, spec)
    result = run_spec(panel, spec)
    assert sample.height == result["n_obs"].max()
    assert sample["signalname"].n_unique() == result["n_predictors"].max()
    assert sample["ret"].null_count() == 0


def test_estimation_sample_rejects_unsupported_specs() -> None:
    panel = _panel()
    with pytest.raises(BootstrapError, match="dep_var 'ret' only"):
        estimation_sample(panel, _spec("normalized"))
    with pytest.raises(BootstrapError, match="no extra terms"):
        estimation_sample(panel, _spec("ext_pub_after_2000"))


def test_point_estimates_match_run_spec_and_decay_table() -> None:
    panel = _panel()
    spec = _spec("main")
    coefs = run_spec(panel, spec)
    decay = decay_table(panel, spec, coefs)
    frame = _run(estimation_sample(panel, spec), reps=5)
    estimates = {row["statistic"]: row["estimate"] for row in frame.iter_rows(named=True)}
    for term in ("post_sample", "post_publication"):
        expected_coef = coefs.filter(pl.col("term") == term)["coef"].item()
        expected_decay = decay.filter(
            (pl.col("method") == "pooled_ratio") & (pl.col("term") == term)
        )["decay_pct"].item()
        assert estimates[f"coef_{term}"] == pytest.approx(expected_coef, abs=1e-10)
        assert estimates[f"decay_pct_{term}"] == pytest.approx(expected_decay, abs=1e-10)


def test_bootstrap_frame_validates_and_pins_the_configured_values() -> None:
    frame = _run(estimation_sample(_panel(), _spec("main")), reps=7, seed=11, block_length=6)
    validate_frame(frame, RESULTS_SCHEMAS["bootstrap.csv"])
    assert frame.select("spec", "statistic").n_unique() == frame.height
    assert frame["spec"].to_list() == ["bootstrap_main"] * 4
    assert set(frame["statistic"].to_list()) == set(STATISTICS)
    assert frame["reps"].to_list() == [7] * 4
    assert frame["seed"].to_list() == [11] * 4
    assert frame["block_length"].to_list() == [6] * 4
    assert (frame["boot_se"] > 0).all()
    assert (frame["ci_low"] < frame["ci_high"]).all()


def test_same_seed_writes_byte_identical_bootstrap_csv(tmp_path: Path) -> None:
    sample = estimation_sample(_panel(), _spec("main"))
    first = write_results(_run(sample, reps=30), "bootstrap.csv", tmp_path / "first")
    second = write_results(_run(sample, reps=30), "bootstrap.csv", tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()

    statistic = pl.col("statistic") == "coef_post_sample"
    same_mean = _run(sample, reps=30).filter(statistic)["boot_mean"].item()
    other_mean = _run(sample, reps=30, seed=2028).filter(statistic)["boot_mean"].item()
    assert same_mean != other_mean


def _synthetic_panel() -> pl.DataFrame:
    """Fixture panel with a known effect: in-sample mean 0.5, post-sample coef -1.0."""
    panel = _panel()
    post_sample = panel["post_sample"].to_numpy().astype(np.float64)
    post_publication = panel["post_publication"].to_numpy().astype(np.float64)
    noise = np.random.default_rng(2027).normal(0.0, 0.5, size=panel.height)
    ret = 0.5 - 1.0 * post_sample - 0.5 * post_publication + noise
    return panel.with_columns(pl.Series("ret", ret, dtype=pl.Float64))


def test_interval_covers_the_known_effect_on_a_synthetic_panel() -> None:
    sample = estimation_sample(_synthetic_panel(), _spec("main"))
    frame = _run(sample, reps=200, seed=42)
    rows = {row["statistic"]: row for row in frame.iter_rows(named=True)}
    coef_row = rows["coef_post_sample"]
    assert coef_row["estimate"] == pytest.approx(-1.0, abs=0.1)
    assert coef_row["ci_low"] <= -1.0 <= coef_row["ci_high"]
    assert coef_row["ci_high"] < 0.0
    assert coef_row["boot_se"] > 0.0
    decay_row = rows["decay_pct_post_sample"]
    assert decay_row["estimate"] == pytest.approx(-200.0, abs=20.0)
    assert decay_row["ci_low"] <= -200.0 <= decay_row["ci_high"]


def test_invalid_arguments_raise() -> None:
    sample = estimation_sample(_panel(), _spec("main"))
    with pytest.raises(BootstrapError, match="statistics must be a non-empty selection"):
        _run(sample, statistics=["sharpe"])
    with pytest.raises(BootstrapError, match="reps must be a positive integer"):
        _run(sample, reps=0)
    with pytest.raises(BootstrapError, match="block_length must be a positive integer"):
        _run(sample, block_length=0)
    with pytest.raises(BootstrapError, match="level must be strictly between 0 and 1"):
        _run(sample, level=1.5)
    with pytest.raises(BootstrapError, match="n_months must be a positive integer"):
        draw_month_positions(np.random.default_rng(1), 0, 12)
    with pytest.raises(BootstrapError, match="block_length must be a positive integer"):
        draw_month_positions(np.random.default_rng(1), 10, 0)


def test_cli_writes_bootstrap_csv_with_the_preregistered_settings(tmp_path: Path) -> None:
    prereg = _write_prereg(tmp_path)
    results = tmp_path / "results"
    code = bootstrap_main(
        [
            "--panel",
            str(FIXTURE_PANEL),
            "--prereg",
            str(prereg),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(results),
        ]
    )
    assert code == 0
    frame = read_results("bootstrap.csv", results)
    validate_frame(frame, RESULTS_SCHEMAS["bootstrap.csv"])
    assert frame["spec"].to_list() == ["bootstrap_main"] * 4
    assert frame["reps"].to_list() == [40] * 4
    assert frame["seed"].to_list() == [7] * 4
    assert frame["block_length"].to_list() == [12] * 4


def test_cli_prints_no_warning_for_the_preregistered_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prereg = _write_prereg(tmp_path)
    code = bootstrap_main(
        [
            "--panel",
            str(FIXTURE_PANEL),
            "--prereg",
            str(prereg),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(tmp_path / "results"),
            "--seed",
            "7",
            "--reps",
            "40",
        ]
    )
    assert code == 0
    assert capsys.readouterr().err == ""


def test_cli_warns_when_not_the_preregistered_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prereg = _write_prereg(tmp_path)
    code = bootstrap_main(
        [
            "--panel",
            str(FIXTURE_PANEL),
            "--prereg",
            str(prereg),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(tmp_path / "results"),
            "--seed",
            "8",
            "--reps",
            "5",
        ]
    )
    assert code == 0
    assert "only seed 7 with 40 reps is the pre-registered run" in capsys.readouterr().err


def test_cli_reports_an_unknown_base_spec(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = yaml.safe_load(PREREG_TEXT)
    data["bootstrap"]["base_spec"] = "nope"
    prereg = tmp_path / "phase2_preregistration.yaml"
    prereg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    code = bootstrap_main(
        [
            "--panel",
            str(FIXTURE_PANEL),
            "--prereg",
            str(prereg),
            "--specs",
            str(SPECS_PATH),
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    assert code == 1
    assert "specification 'nope': not in" in capsys.readouterr().err


def test_runtime_on_the_fixture_panel_with_200_reps_is_under_20_seconds() -> None:
    sample = estimation_sample(_panel(), _spec("main"))
    started = time.perf_counter()
    frame = _run(sample, reps=200, seed=2027)
    elapsed = time.perf_counter() - started
    assert frame.height == len(STATISTICS)
    assert elapsed < 20.0, f"200 replicates took {elapsed:.1f}s"


def test_statistics_selection_limits_the_output_rows() -> None:
    frame = _run(
        estimation_sample(_panel(), _spec("main")),
        statistics=("coef_post_sample", "decay_pct_post_publication"),
        reps=3,
    )
    assert frame["statistic"].to_list() == ["coef_post_sample", "decay_pct_post_publication"]
