"""Seeded moving-block bootstrap by month for the main decay estimates.

The pre-registered scheme (`config/phase2_preregistration.yaml`, frozen with the
specifications by the tag ``specs-frozen``) resamples the ``main`` specification's
estimation sample as follows:

- the sampling units are the distinct calendar months of the estimation sample, in
  calendar order (the panel month key ``yyyymm``);
- a replicate draws ``ceil(T / L)`` *start indices* uniformly with replacement from
  ``0 .. T - L`` inclusive (``T`` months, block length ``L``, **no wrap**: a block is
  the contiguous run of ``L`` months starting at its draw and the last possible start
  is ``T - L``); the blocks are concatenated and truncated to ``T`` month positions.
  A month drawn twice appears twice;
- every predictor's observations in a resampled month occurrence move together (the
  whole cross-section of that month is copied);
- each replicate re-estimates the ``main`` specification on the resampled rows with
  within-predictor (entity) demeaned OLS in numpy and forms the pooled-ratio decay
  against *that replicate's* in-sample mean;
- intervals are percentile intervals over the replicates; the generator is
  ``numpy.random.default_rng(seed)`` (no global seeding, fixed draw order: one
  ``integers`` call per replicate).

This module implements that scheme and nothing else. The classic failure this guards
against — resampling individual predictor-months i.i.d. instead of months in blocks —
is excluded by :func:`draw_month_positions` and
``tests/test_bootstrap.py::test_resampled_months_come_in_contiguous_blocks``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from anomaly_decay.config import (
    Spec,
    load_phase2_preregistration,
    load_specifications,
)
from anomaly_decay.io import write_results
from anomaly_decay.schemas import RESULTS_SCHEMAS

__all__ = [
    "BootstrapError",
    "bootstrap_intervals",
    "draw_month_positions",
    "estimation_sample",
    "main",
]

POST_SAMPLE = "post_sample"
POST_PUBLICATION = "post_publication"
STATISTICS: tuple[str, ...] = (
    "coef_post_sample",
    "coef_post_publication",
    "decay_pct_post_sample",
    "decay_pct_post_publication",
)

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "signalname",
    "yyyymm",
    "ret",
    "post_sample",
    "post_publication",
    "window",
)


class BootstrapError(RuntimeError):
    """Raised when the bootstrap cannot run on the supplied panel, spec or arguments."""


def estimation_sample(panel: pl.DataFrame, spec: Spec) -> pl.DataFrame:
    """The rows of *panel* that enter the bootstrap regression for *spec*.

    Mirrors the sample construction of ``run_spec`` for the bootstrap's supported
    case — ``dep_var == "ret"`` and no extra terms: the spec's portfolio, universe
    and month filters, rows with a non-null dependent variable, and the
    convention-appropriate window dummies exposed under the canonical names
    ``post_sample`` / ``post_publication``. Output columns:
    ``signalname, yyyymm, ret, post_sample, post_publication, window``; a test
    asserts the height and predictor count equal ``run_spec``'s ``n_obs`` and
    ``n_predictors``.
    """
    if spec.dep_var != "ret":
        raise BootstrapError(
            f"specification '{spec.id}': the bootstrap supports dep_var 'ret' only, "
            f"got {spec.dep_var!r}"
        )
    if spec.extra_terms:
        raise BootstrapError(
            f"specification '{spec.id}': the bootstrap supports no extra terms, "
            f"got {list(spec.extra_terms)}"
        )
    sample = panel.filter(pl.col("portfolio") == spec.portfolio)
    if spec.pub_year_max is not None:
        sample = sample.filter(pl.col("pub_year") <= spec.pub_year_max)
    if spec.sample_start_year_min is not None:
        sample = sample.filter(pl.col("sample_start_year") >= spec.sample_start_year_min)
    if spec.clear_only:
        sample = sample.filter(pl.col("clear"))
    if spec.start_yyyymm is not None:
        sample = sample.filter(pl.col("yyyymm") >= spec.start_yyyymm)
    if spec.end_yyyymm is not None:
        sample = sample.filter(pl.col("yyyymm") <= spec.end_yyyymm)
    sample = sample.filter(pl.col("ret").is_not_null())
    if spec.convention == "alt":
        sample = sample.with_columns(
            pl.col("post_sample_alt").cast(pl.Float64).alias(POST_SAMPLE),
            pl.col("post_publication_alt").cast(pl.Float64).alias(POST_PUBLICATION),
        )
    else:
        sample = sample.with_columns(
            pl.col(POST_SAMPLE).cast(pl.Float64),
            pl.col(POST_PUBLICATION).cast(pl.Float64),
        )
    if sample.height == 0:
        raise BootstrapError(f"specification '{spec.id}': estimation sample is empty")
    return sample.select(_REQUIRED_COLUMNS)


def draw_month_positions(
    rng: np.random.Generator, n_months: int, block_length: int
) -> NDArray[np.int64]:
    """One replicate's month positions: contiguous blocks, no wrap, truncated to fit.

    ``ceil(n_months / block_length)`` start indices are drawn uniformly with
    replacement from ``0 .. n_months - block_length`` inclusive; each block is the
    contiguous run of ``block_length`` positions from its start; the blocks are
    concatenated and cut to ``n_months`` positions. A position drawn twice appears
    twice. When ``n_months <= block_length`` the replicate is the whole sample once,
    in order.
    """
    if not isinstance(n_months, int) or n_months < 1:
        raise BootstrapError(f"n_months must be a positive integer, got {n_months!r}")
    if not isinstance(block_length, int) or block_length < 1:
        raise BootstrapError(f"block_length must be a positive integer, got {block_length!r}")
    if n_months <= block_length:
        return np.arange(n_months, dtype=np.int64)
    n_blocks = -(-n_months // block_length)
    starts = np.asarray(rng.integers(0, n_months - block_length + 1, size=n_blocks), dtype=np.int64)
    positions = (starts[:, None] + np.arange(block_length, dtype=np.int64)).ravel()
    return positions[:n_months]


def _month_row_index(
    positions: NDArray[np.int64],
    month_starts: NDArray[np.int64],
    month_lengths: NDArray[np.int64],
) -> NDArray[np.int64]:
    """The row indices of the month positions, in draw order.

    With the sample sorted by month, the rows of month ``m`` are the contiguous range
    ``month_starts[m] .. month_starts[m] + month_lengths[m] - 1``; the result is those
    ranges concatenated in the order the months were drawn, so every predictor's
    observations of a drawn month occurrence travel together.
    """
    counts = month_lengths[positions]
    total = int(counts.sum())
    bases = np.repeat(month_starts[positions], counts)
    cumulative = np.cumsum(counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(cumulative - counts, counts)
    return bases + within


def _within_coefficients(
    y: NDArray[np.float64],
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    entities: NDArray[np.int64],
) -> tuple[float, float]:
    """Within-entity demeaned OLS of *y* on *x1*, *x2*; the two slope coefficients.

    Absorbing the predictor fixed effects is exactly subtracting each entity's mean
    from the dependent variable and the two regressors (the same transformation
    ``PanelOLS(entity_effects=True)`` applies), after which the normal equations are
    solved directly. Raises :class:`BootstrapError` when a term has no variation.
    """

    def demean(values: NDArray[np.float64]) -> NDArray[np.float64]:
        counts = np.bincount(entities)
        totals = np.bincount(entities, weights=values)
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        return values - means[entities]

    yd = demean(y)
    design = np.column_stack((demean(x1), demean(x2)))
    xtx = design.T @ design
    xty = design.T @ yd
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError as exc:
        raise BootstrapError(
            "the regression is singular (a term has no variation in the sample)"
        ) from exc
    return float(beta[0]), float(beta[1])


def bootstrap_intervals(
    sample: pl.DataFrame,
    *,
    spec_id: str,
    statistics: Sequence[str],
    reps: int,
    seed: int,
    block_length: int,
    level: float,
) -> pl.DataFrame:
    """Moving-block bootstrap distribution of the decay statistics; schema rows.

    Inputs: *sample* as returned by :func:`estimation_sample`; *spec_id* is written to
    the ``spec`` column (the registered analysis id, e.g. ``bootstrap_main``);
    *statistics* selects output rows among :data:`STATISTICS`; *reps*, *seed*,
    *block_length* and *level* come from the pre-registration. Output: one row per
    statistic per ``RESULTS_SCHEMAS["bootstrap.csv"]`` with the point estimate from
    the original sample, the bootstrap mean and standard error, and the percentile
    interval. ``estimate`` for the coefficient statistics equals ``run_spec``'s
    coefficient to 1e-10 (asserted in the tests); for the decay statistics it is
    ``100 * coef / pooled in-sample mean`` exactly as in ``decay_table``.

    A replicate with no in-sample rows or a zero replicate in-sample mean gets a NaN
    decay value; NaN values are excluded from that statistic's mean and interval.
    The replicate draw order is fixed (one ``integers`` call per replicate), so the
    same seed and panel give byte-identical CSV output.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in sample.columns]
    if missing:
        raise BootstrapError(f"sample is missing columns: {missing}")
    unknown = [statistic for statistic in statistics if statistic not in STATISTICS]
    if not statistics or unknown:
        raise BootstrapError(
            f"statistics must be a non-empty selection of {list(STATISTICS)}, "
            f"got {list(statistics)}"
        )
    if reps < 1:
        raise BootstrapError(f"reps must be a positive integer, got {reps}")
    if block_length < 1:
        raise BootstrapError(f"block_length must be a positive integer, got {block_length}")
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise BootstrapError(f"level must be strictly between 0 and 1, got {level}")
    if sample.height < 2:
        raise BootstrapError("sample is too small to estimate")

    ordered = sample.sort(["yyyymm", "signalname"])
    months = ordered["yyyymm"].to_numpy()
    unique_months, month_lengths = np.unique(months, return_counts=True)
    n_months = int(unique_months.size)
    month_starts = np.concatenate(([0], np.cumsum(month_lengths)[:-1])).astype(np.int64)
    entities = np.asarray(
        np.unique(ordered["signalname"].to_numpy(), return_inverse=True)[1], dtype=np.int64
    )
    y = ordered["ret"].to_numpy().astype(np.float64)
    x1 = ordered[POST_SAMPLE].to_numpy().astype(np.float64)
    x2 = ordered[POST_PUBLICATION].to_numpy().astype(np.float64)
    in_sample = (ordered["window"] == "in_sample").to_numpy()

    coef1, coef2 = _within_coefficients(y, x1, x2, entities)
    in_sample_rows = ordered.filter(pl.col("window") == "in_sample")
    if in_sample_rows.height == 0:
        raise BootstrapError("no in-sample rows for the pooled in-sample mean")
    in_sample_mean = float(in_sample_rows.select(pl.col("ret").mean()).item())
    if in_sample_mean == 0.0:
        raise BootstrapError("pooled in-sample mean is zero")

    rng = np.random.default_rng(seed)
    coef_replicates = np.empty((reps, 2), dtype=np.float64)
    decay_replicates = np.empty((reps, 2), dtype=np.float64)
    for index in range(reps):
        positions = draw_month_positions(rng, n_months, block_length)
        rows = _month_row_index(positions, month_starts, month_lengths)
        replicate_y = y[rows]
        try:
            replicate_coefs = _within_coefficients(replicate_y, x1[rows], x2[rows], entities[rows])
        except BootstrapError as exc:
            raise BootstrapError(f"replicate {index}: {exc}") from exc
        coef_replicates[index] = replicate_coefs
        replicate_in_sample = in_sample[rows]
        decay_replicates[index] = (math.nan, math.nan)
        if replicate_in_sample.any():
            replicate_mean = float(replicate_y[replicate_in_sample].mean())
            if replicate_mean != 0.0:
                decay_replicates[index] = (
                    100.0 * replicate_coefs[0] / replicate_mean,
                    100.0 * replicate_coefs[1] / replicate_mean,
                )

    point = {
        "coef_post_sample": coef1,
        "coef_post_publication": coef2,
        "decay_pct_post_sample": 100.0 * coef1 / in_sample_mean,
        "decay_pct_post_publication": 100.0 * coef2 / in_sample_mean,
    }
    replicates = {
        "coef_post_sample": coef_replicates[:, 0],
        "coef_post_publication": coef_replicates[:, 1],
        "decay_pct_post_sample": decay_replicates[:, 0],
        "decay_pct_post_publication": decay_replicates[:, 1],
    }
    tail = 100.0 * (1.0 - level) / 2.0
    records = []
    for statistic in statistics:
        values = replicates[statistic]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise BootstrapError(f"statistic {statistic!r}: no finite bootstrap replicates")
        ci_low, ci_high = np.percentile(finite, [tail, 100.0 - tail])
        records.append(
            {
                "spec": spec_id,
                "statistic": statistic,
                "estimate": point[statistic],
                "boot_mean": float(finite.mean()),
                "boot_se": float(finite.std(ddof=1)) if finite.size > 1 else math.nan,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "reps": reps,
                "block_length": block_length,
                "seed": seed,
            }
        )
    return pl.DataFrame(records, schema=RESULTS_SCHEMAS["bootstrap.csv"])


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.estimate.bootstrap [--panel PATH] [--prereg PATH]
    [--specs PATH] [--results-dir DIR] [--seed N] [--reps N]``.

    Defaults follow the pre-registration: seed 2027, 2000 replicates, block length 12,
    percentile 95% intervals, base specification ``main``. Overriding ``--seed`` or
    ``--reps`` is allowed but prints a warning that only the pre-registered
    combination is the confirmatory run. Writes ``results/bootstrap.csv`` through
    :func:`anomaly_decay.io.write_results`; returns 1 with a message on
    :class:`BootstrapError`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.bootstrap",
        description="Moving-block bootstrap by month for the main decay estimates.",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/panel.parquet"),
        help="panel.parquet path",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("config/phase2_preregistration.yaml"),
        help="phase2_preregistration.yaml path",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=Path("config/specifications.yaml"),
        help="specifications.yaml path",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="results output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="bootstrap seed (default: pre-registered)"
    )
    parser.add_argument(
        "--reps", type=int, default=None, help="bootstrap replicates (default: pre-registered)"
    )
    args = parser.parse_args(argv)

    prereg = load_phase2_preregistration(args.prereg)
    configure = prereg.bootstrap
    seed = configure.seed if args.seed is None else args.seed
    reps = configure.reps if args.reps is None else args.reps
    if seed != configure.seed or reps != configure.reps:
        print(
            f"warning: only seed {configure.seed} with {configure.reps} reps is the "
            "pre-registered run",
            file=sys.stderr,
        )

    try:
        panel = pl.read_parquet(args.panel)
        specs = load_specifications(args.specs)
        by_id = {spec.id: spec for spec in specs}
        base_spec = by_id.get(configure.base_spec)
        if base_spec is None:
            raise BootstrapError(f"specification '{configure.base_spec}': not in {args.specs}")
        sample = estimation_sample(panel, base_spec)
        frame = bootstrap_intervals(
            sample,
            spec_id=configure.id,
            statistics=configure.statistics,
            reps=reps,
            seed=seed,
            block_length=configure.block_length_months,
            level=configure.level,
        )
    except BootstrapError as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 1
    write_results(frame, "bootstrap.csv", args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
