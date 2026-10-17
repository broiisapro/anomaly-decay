"""Main specification: pooled panel regression of long-short returns.

The runner filters the Polars panel to one specification's estimation sample,
translates it to the pandas ``(signalname, month)`` MultiIndex layout that
linearmodels requires, absorbs predictor fixed effects and returns coefficients
with clustered standard errors. The Polars-to-pandas conversion happens only
here and in the sibling modules of ``anomaly_decay.estimate``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from linearmodels.panel.model import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from linearmodels.panel.utility import AbsorbingEffectError

from anomaly_decay.config import Spec, load_specifications
from anomaly_decay.estimate.factors import factor_adjusted_returns
from anomaly_decay.io import write_results
from anomaly_decay.schemas import PANEL_KEY, RESULTS_SCHEMAS

__all__ = ["EstimationError", "decay_table", "main", "run_spec"]

POST_SAMPLE = "post_sample"
POST_PUBLICATION = "post_publication"
POOLED_RATIO = "pooled_ratio"
NORMALIZED = "normalized"
_PUB_AFTER_2000 = "post_publication_x_pub_after_2000"


class EstimationError(RuntimeError):
    """Raised when a specification cannot be estimated on the supplied panel.

    The message records what went wrong (empty sample, a term with no
    variation, a singular regression, ...); the CLI writes it to
    ``specs_run.csv`` and continues with the remaining specifications.
    """


def _dummy_columns(spec: Spec) -> tuple[str, str]:
    """Panel columns holding the post-sample and post-publication dummies."""
    if spec.convention == "alt":
        return "post_sample_alt", "post_publication_alt"
    return "post_sample", "post_publication"


def _term_expressions(spec: Spec) -> dict[str, pl.Expr]:
    """Regression terms in output order, as expressions over the panel.

    The two window dummies keep the names ``post_sample`` and
    ``post_publication`` whatever convention supplies their values. Extra terms
    are derived from the convention-appropriate post-publication dummy.
    """
    post_sample, post_publication = _dummy_columns(spec)
    terms: dict[str, pl.Expr] = {
        POST_SAMPLE: pl.col(post_sample).cast(pl.Float64),
        POST_PUBLICATION: pl.col(post_publication).cast(pl.Float64),
    }
    for name in spec.extra_terms:
        if name == _PUB_AFTER_2000:
            terms[name] = pl.col(post_publication).cast(pl.Float64) * (
                pl.col("pub_year") > 2000
            ).cast(pl.Float64)
        else:
            raise EstimationError(f"specification '{spec.id}': unknown extra term {name!r}")
    return terms


def _filter_sample(panel: pl.DataFrame, spec: Spec) -> pl.DataFrame:
    """Apply the specification's portfolio, universe and month filters."""
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
    return sample


def _estimation_sample(panel: pl.DataFrame, spec: Spec, *, with_terms: bool) -> pl.DataFrame:
    """Filtered rows that enter the regression, complete on the dependent variable.

    Adds the regression-term columns when *with_terms* is true. The
    factor-adjusted dependent variable is joined from
    :func:`anomaly_decay.estimate.factors.factor_adjusted_returns` on
    :data:`~anomaly_decay.schemas.PANEL_KEY`. Rows whose dependent variable is
    null (``ret_norm`` for predictors with a non-positive in-sample mean,
    ``ret_factor_adj`` for portfolios without enough usable in-sample months)
    are dropped, so the returned frame is exactly the estimation sample.
    """
    sample = _filter_sample(panel, spec)
    if spec.dep_var == "ret_factor_adj":
        adjusted = factor_adjusted_returns(panel)
        sample = sample.join(adjusted, on=list(PANEL_KEY), how="left")
    elif spec.dep_var not in ("ret", "ret_norm"):
        raise EstimationError(f"specification '{spec.id}': unsupported dep_var {spec.dep_var!r}")
    sample = sample.filter(pl.col(spec.dep_var).is_not_null())
    if with_terms:
        for name, expression in _term_expressions(spec).items():
            sample = sample.with_columns(expression.alias(name))
    return sample


def _fit(model: PanelOLS, spec: Spec) -> PanelEffectsResults:
    """Fit *model* with the covariance estimator named by *spec*.

    Production keeps linearmodels' defaults: ``debiased=True``, ``auto_df=True``
    and ``count_effects=True``, so with entity effects and time clustering the
    covariance is scaled by ``n / (n - d - k)`` (d = number of predictors,
    k = number of regressors); the effects are counted because predictor effects
    are not nested in the time clusters.
    """
    if spec.cov == "clustered_time":
        return model.fit(cov_type="clustered", cluster_time=True)
    if spec.cov == "clustered_twoway":
        return model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    if spec.cov == "driscoll_kraay":
        kernel = str(spec.cov_params.get("kernel", "newey-west"))
        bandwidth = spec.cov_params.get("bandwidth")
        if bandwidth is None:
            return model.fit(cov_type="kernel", kernel=kernel)
        if not isinstance(bandwidth, int):
            raise EstimationError(f"specification '{spec.id}': bandwidth must be an integer")
        return model.fit(cov_type="kernel", kernel=kernel, bandwidth=bandwidth)
    raise EstimationError(f"specification '{spec.id}': unsupported cov {spec.cov!r}")


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

    The regression is linearmodels ``PanelOLS(dep, exog, entity_effects=True)``
    with **no constant column** — the predictor effects absorb it, and adding a
    constant alongside them would be collinear. For ``clustered_time`` the
    covariance is the one-way month-clustered Rogers sandwich that
    ``tests/test_estimate.py::test_clustered_se_matches_manual_computation``
    reproduces by hand:

        Cov = n / (n - d - k) * (X'X)^-1 [sum_g s_g s_g'] (X'X)^-1

    where :math:`s_g = \\sum_{i \\in g} x_i e_i` is the sum of score vectors
    over the observations of calendar month *g*, ``x_i`` and ``e_i`` are the
    within-predictor demeaned regressors and residuals, ``k`` is the number of
    terms and ``d`` is the number of predictors (linearmodels counts the
    estimated effects in the degrees of freedom because predictor effects are
    not nested in the time clusters). The two-way and Driscoll-Kraay covariances
    are linearmodels estimators with default options.

    Raises :class:`EstimationError` when the sample is empty, a term has no
    variation, or the regression is singular/absorbed.
    """
    sample = _estimation_sample(panel, spec, with_terms=True)
    if sample.height == 0:
        raise EstimationError(f"specification '{spec.id}': estimation sample is empty")
    term_names = list(_term_expressions(spec))
    for term in term_names:
        if sample[term].n_unique() < 2:
            raise EstimationError(
                f"specification '{spec.id}': term '{term}' has no variation "
                "in the estimation sample"
            )

    pdf = sample.select(["signalname", "month", spec.dep_var, *term_names]).to_pandas()
    pdf = pdf.set_index(["signalname", "month"])
    try:
        model = PanelOLS(pdf[spec.dep_var], pdf[term_names], entity_effects=True)
        results = _fit(model, spec)
    except (AbsorbingEffectError, ValueError, np.linalg.LinAlgError) as exc:
        raise EstimationError(f"specification '{spec.id}': {exc}") from exc

    params = results.params
    std_errors = results.std_errors
    tstats = results.tstats
    n_predictors = sample["signalname"].n_unique()
    n_obs = sample.height
    rows = [
        {
            "spec": spec.id,
            "term": term,
            "coef": float(params[term]),
            "se": float(std_errors[term]),
            "t": float(tstats[term]),
            "n_predictors": n_predictors,
            "n_obs": n_obs,
        }
        for term in term_names
    ]
    return pl.DataFrame(rows, schema=RESULTS_SCHEMAS["main.csv"])


def decay_table(panel: pl.DataFrame, spec: Spec, coefs: pl.DataFrame) -> pl.DataFrame:
    """Percentage decay relative to the in-sample mean, two ways.

    Inputs: *panel* (schema above), *spec*, and *coefs* as returned by
    :func:`run_spec`. Output rows per ``RESULTS_SCHEMAS["decay.csv"]`` for methods
    ``pooled_ratio`` (coefficient ÷ pooled in-sample mean of the estimation sample)
    and ``normalized`` (regression on ``ret_norm``; predictors with in-sample mean
    ``<= 0`` are excluded and counted via ``n_predictors``). ``decay_pct`` and
    ``se_pct`` are percentages (e.g. 26.0 means 26% lower); ``in_sample_mean`` is in
    percent per month. Window rules follow architecture §4.6.

    Which method is produced follows ``spec.dep_var``: ``ret`` gives
    ``pooled_ratio`` (``in_sample_mean`` is the mean ``ret`` over the in-sample
    rows of the spec's own estimation sample, and the standard-error percentage
    treats that mean as fixed: ``se_pct = 100 * se / in_sample_mean``, so it
    carries only the coefficient's sampling error), ``ret_norm`` gives
    ``normalized`` with ``decay_pct = 100 * coef`` and ``in_sample_mean = 1``
    (the dependent variable is already scaled by each predictor's in-sample
    mean). ``n_predictors`` and ``n_obs`` are the coefficients' regression
    counts. A negative ``decay_pct`` means returns are lower. A zero pooled
    in-sample mean is undefined and raises :class:`EstimationError`; a negative
    one flips the sign of the ratio.
    """
    if spec.dep_var == "ret_norm":
        method = NORMALIZED
        in_sample_mean = 1.0
        rows = [
            {
                "spec": spec.id,
                "method": method,
                "term": row["term"],
                "decay_pct": 100.0 * row["coef"],
                "se_pct": 100.0 * row["se"],
                "in_sample_mean": in_sample_mean,
                "n_predictors": row["n_predictors"],
                "n_obs": row["n_obs"],
            }
            for row in coefs.iter_rows(named=True)
        ]
    elif spec.dep_var == "ret":
        method = POOLED_RATIO
        sample = _estimation_sample(panel, spec, with_terms=False)
        in_sample = sample.filter(pl.col("window") == "in_sample")
        if in_sample.height == 0:
            raise EstimationError(
                f"specification '{spec.id}': no in-sample rows for the pooled mean"
            )
        in_sample_mean = float(in_sample.select(pl.col("ret").mean()).item())
        if in_sample_mean == 0.0:
            raise EstimationError(f"specification '{spec.id}': pooled in-sample mean is zero")
        rows = [
            {
                "spec": spec.id,
                "method": method,
                "term": row["term"],
                "decay_pct": 100.0 * row["coef"] / in_sample_mean,
                "se_pct": 100.0 * row["se"] / in_sample_mean,
                "in_sample_mean": in_sample_mean,
                "n_predictors": row["n_predictors"],
                "n_obs": row["n_obs"],
            }
            for row in coefs.iter_rows(named=True)
        ]
    else:
        raise EstimationError(
            f"specification '{spec.id}': decay requires dep_var ret or ret_norm, "
            f"got {spec.dep_var!r}"
        )
    return pl.DataFrame(rows, schema=RESULTS_SCHEMAS["decay.csv"])


def _spec_status(spec: Spec, status: str) -> dict[str, object]:
    return {
        "spec": spec.id,
        "label": spec.label,
        "status": status,
        "exploratory": spec.exploratory,
    }


def _print_headline(decay: pl.DataFrame) -> int:
    """Print the two pooled-ratio headline lines; 0 on success, 1 if unavailable."""
    pooled = decay.filter(pl.col("method") == POOLED_RATIO)
    rows = {row["term"]: row for row in pooled.iter_rows(named=True)}
    lines: list[str] = []
    for term, label in ((POST_SAMPLE, "post-sample"), (POST_PUBLICATION, "post-publication")):
        row = rows.get(term)
        if row is None:
            print(
                f"headline unavailable: no pooled_ratio result for {term}",
                file=sys.stderr,
            )
            return 1
        lines.append(
            f"{label}: {row['decay_pct']:.1f}% (SE {row['se_pct']:.1f}%), "
            f"N predictors = {row['n_predictors']}, N obs = {row['n_obs']}"
        )
    print("\n".join(lines))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.estimate.main --panel PATH --specs PATH --results-dir DIR
    [--print-headline]``.

    Writes ``main.csv``, ``decay.csv`` and ``specs_run.csv`` under *results-dir*;
    ``--print-headline`` prints the post-sample and post-publication decay (%) with
    SEs and N predictors.

    Every specification in *specs* is run (unless ``--only ID`` restricts the run
    to one); a specification that raises :class:`EstimationError` is recorded as
    ``failed: <reason>`` in ``specs_run.csv`` and does not stop the others.
    ``decay.csv`` carries the pooled-ratio rows of ``main`` and the normalized
    rows of ``normalized``, whichever of the two ran.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.main",
        description="Run the specification panel regressions and write decay results.",
    )
    parser.add_argument("--panel", type=Path, required=True, help="panel.parquet path")
    parser.add_argument("--specs", type=Path, required=True, help="specifications.yaml path")
    parser.add_argument("--results-dir", type=Path, required=True, help="results output directory")
    parser.add_argument(
        "--print-headline", action="store_true", help="print headline decay numbers"
    )
    parser.add_argument("--only", metavar="ID", default=None, help="run one specification id only")
    args = parser.parse_args(argv)

    panel = pl.read_parquet(args.panel)
    specs = load_specifications(args.specs)
    by_id = {spec.id: spec for spec in specs}
    if args.only is not None and args.only not in by_id:
        parser.error(f"unknown specification id: {args.only}")

    estimates: dict[str, pl.DataFrame] = {}
    decay_parts: list[pl.DataFrame] = []
    statuses: list[dict[str, object]] = []
    for spec in specs:
        if args.only is not None and spec.id != args.only:
            statuses.append(_spec_status(spec, "skipped"))
            continue
        try:
            coefs = run_spec(panel, spec)
            if spec.id in ("main", "normalized"):
                decay_parts.append(decay_table(panel, spec, coefs))
        except EstimationError as exc:
            statuses.append(_spec_status(spec, f"failed: {exc}"))
        else:
            estimates[spec.id] = coefs
            statuses.append(_spec_status(spec, "ok"))

    if "main" in estimates:
        write_results(estimates["main"], "main.csv", args.results_dir)

    decay = pl.concat(decay_parts) if decay_parts else None
    if decay is not None:
        write_results(decay, "decay.csv", args.results_dir)

    write_results(
        pl.DataFrame(statuses, schema=RESULTS_SCHEMAS["specs_run.csv"]),
        "specs_run.csv",
        args.results_dir,
    )

    if args.print_headline:
        if decay is None:
            print("headline unavailable: the main specification did not run", file=sys.stderr)
            return 1
        return _print_headline(decay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
