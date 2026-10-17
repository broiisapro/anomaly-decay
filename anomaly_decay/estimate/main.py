"""Main specification: pooled panel regression of long-short returns.

Interface stub: implemented in Epic 01 workstream B.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from anomaly_decay.config import Spec, load_specifications

__all__ = ["decay_table", "main", "run_spec"]


def run_spec(panel: pl.DataFrame, spec: Spec) -> pl.DataFrame:
    """Estimate one specification.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA`; *spec* is one
    entry of ``config/specifications.yaml``. The dependent variable is
    ``spec.dep_var`` (``ret``, ``ret_norm`` or ``ret_factor_adj``; returns and their
    normalized or factor-adjusted counterparts are **percent per month**), regressed
    on ``post_sample`` and ``post_publication`` (plus any ``spec.extra_terms``) with
    predictor fixed effects and ``spec.cov`` standard errors; the sample is filtered
    by ``spec.universe`` and ``spec.months``.

    Output: rows ``spec, term, coef, se, t, n_predictors, n_obs`` (one row per term)
    per ``RESULTS_SCHEMAS["main.csv"]``. Coefficients are in percent per month.
    """
    raise NotImplementedError


def decay_table(panel: pl.DataFrame, spec: Spec, coefs: pl.DataFrame) -> pl.DataFrame:
    """Percentage decay relative to the in-sample mean, two ways.

    Inputs: *panel* (schema above), *spec*, and *coefs* as returned by
    :func:`run_spec`. Output rows per ``RESULTS_SCHEMAS["decay.csv"]`` for methods
    ``pooled_ratio`` (coefficient ÷ pooled in-sample mean of the estimation sample)
    and ``normalized`` (regression on ``ret_norm``; predictors with in-sample mean
    ``<= 0`` are excluded and counted via ``n_predictors``). ``decay_pct`` and
    ``se_pct`` are percentages (e.g. 26.0 means 26% lower); ``in_sample_mean`` is in
    percent per month. Window rules follow architecture §4.6.
    """
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.estimate.main --panel PATH --specs PATH --results-dir DIR
    [--print-headline]``.

    Writes ``main.csv``, ``decay.csv`` and ``specs_run.csv`` under *results-dir*;
    ``--print-headline`` prints the post-sample and post-publication decay (%) with
    SEs and N predictors.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.main",
        description="Run the main specification and write decay results.",
    )
    parser.add_argument("--panel", type=Path, required=True, help="panel.parquet path")
    parser.add_argument("--specs", type=Path, required=True, help="specifications.yaml path")
    parser.add_argument("--results-dir", type=Path, required=True, help="results output directory")
    parser.add_argument(
        "--print-headline", action="store_true", help="print headline decay numbers"
    )
    args = parser.parse_args(argv)
    panel = pl.read_parquet(args.panel)
    specs = load_specifications(args.specs)
    run_spec(panel, specs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
