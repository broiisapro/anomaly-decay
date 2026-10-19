"""Multiple-testing lens: t-statistic sources and weak-predictor flags.

Phase 2 pre-registers a multiple-testing analysis that asks whether anomaly
decay is larger for predictors with weaker original evidence. Two t-statistic
sources are defined by ``config/phase2_preregistration.yaml``:

``original_paper``
    The ``T-Stat`` column of the OSAP signal doc — the original paper's reported
    t-statistic, used as reported (the sign follows the paper's direction; no
    absolute value is applied).
``in_sample_replicated``
    Mean / (sd / sqrt(n)) of the ``op`` long-short return over the predictor's
    in-sample months (``window == "in_sample"``), with the sample standard
    deviation (``ddof=1``).

A predictor is *weak* for a source when its t-statistic is **strictly below**
the pre-registered Harvey–Liu–Zhu hurdle (``hurdle_t``, 3.0), and the weak flag
is null exactly when the t-statistic is null. Predictors whose t-statistic is
missing are excluded from the estimation and counted (``missing_tstat:
exclude_and_count``), never imputed.

The signal doc is read as text (``infer_schema_length=0``) and parsed
explicitly. Parse rules, applied to the raw cell after stripping whitespace:

- empty or whitespace-only → null (counted as missing);
- anything ``float()`` accepts (decimal digits, a leading ``+`` or ``-``, an
  exponent such as ``5e0``) → its value; the sign is kept as reported;
- anything else (letters, thousands separators, parentheses, a missing
  ``T-Stat`` column, a predictor absent from the doc) → null (counted);
- non-finite values (``nan``, ``inf``) → null.

Only rows with ``Cat.Signal == "Predictor"`` are considered when the column is
present, and signal names must be unique among them.

The pre-registered multiple-testing specifications extend the ``main`` sample
with predictor-level interactions: ``post_sample_x_weak`` /
``post_publication_x_weak`` multiply the window dummies by the source's weak
flag and ``post_sample_x_tstat`` / ``post_publication_x_tstat`` by its
t-statistic. Those flags and values are predictor-level, so they have no main
effect of their own — the predictor effects absorb it and no extra term is
added. Predictors with a missing t for the specification's source are excluded
from the estimation and counted (``n_excluded_missing_t``), never imputed.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from linearmodels.panel.model import PanelOLS
from linearmodels.panel.utility import AbsorbingEffectError
from polars.datatypes import DataTypeClass

from anomaly_decay.config import (
    MultipleTestingPrereg,
    MultipleTestingSpecPrereg,
    Spec,
    load_phase2_preregistration,
    load_specifications,
)
from anomaly_decay.estimate.main import EstimationError, _filter_sample, _fit
from anomaly_decay.io import write_results
from anomaly_decay.schemas import RESULTS_SCHEMAS

__all__ = [
    "DEFAULT_HURDLE_T",
    "TSTAT_SOURCE_COLUMNS",
    "TSTAT_SOURCES",
    "TstatError",
    "main",
    "run_multiple_testing",
    "run_multiple_testing_spec",
    "tstat_summary",
    "tstats",
]

DEFAULT_HURDLE_T = 3.0
"""Pre-registered hurdle (Harvey, Liu and Zhu 2016); weak = strictly below."""

TSTAT_SOURCES: tuple[str, ...] = ("original_paper", "in_sample_replicated")
"""Canonical order of the two t-statistic sources."""

TSTAT_SOURCE_COLUMNS: Mapping[str, tuple[str, str]] = {
    "original_paper": ("t_original", "weak_original"),
    "in_sample_replicated": ("t_replicated", "weak_replicated"),
}
"""Per source: the :func:`tstats` columns holding its t-statistic and weak flag."""

TSTATS_SCHEMA: dict[str, DataTypeClass] = {
    "signalname": pl.Utf8,
    "t_original": pl.Float64,
    "t_replicated": pl.Float64,
    "weak_original": pl.Boolean,
    "weak_replicated": pl.Boolean,
}

_INTERACTION_KINDS: Mapping[str, str] = {
    "post_sample_x_weak": "weak",
    "post_publication_x_weak": "weak",
    "post_sample_x_tstat": "tstat",
    "post_publication_x_tstat": "tstat",
}
"""Interaction terms a pre-registered specification may add, and their multiplier."""


class TstatError(ValueError):
    """Raised when the signal doc cannot produce a usable t-statistic table."""


def _parse_tstat(value: object) -> float | None:
    """Parse one ``T-Stat`` cell; see the module docstring for the rules."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _original_tstats(signal_doc_path: Path) -> pl.DataFrame:
    """Per-predictor ``T-Stat`` values from the signal doc, as ``signalname, t``."""
    path = Path(signal_doc_path)
    if not path.is_file():
        raise TstatError(f"signal doc not found: {path}")
    raw = pl.read_csv(path, infer_schema_length=0)
    if "Acronym" not in raw.columns:
        raise TstatError(f"signal doc {path}: missing column 'Acronym'")
    if "Cat.Signal" in raw.columns:
        raw = raw.filter(pl.col("Cat.Signal") == "Predictor")
    if "T-Stat" in raw.columns:
        values = [_parse_tstat(value) for value in raw["T-Stat"].to_list()]
    else:
        values = [None] * raw.height

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for name, value in zip(raw["Acronym"].to_list(), values, strict=True):
        signalname = str(name).strip() if name is not None else ""
        if not signalname:
            raise TstatError(f"signal doc {path}: a Predictor row has no Acronym")
        if signalname in seen:
            raise TstatError(f"signal doc {path}: predictor {signalname!r} appears twice")
        seen.add(signalname)
        rows.append({"signalname": signalname, "t_original": value})
    return pl.DataFrame(rows, schema={"signalname": pl.Utf8, "t_original": pl.Float64})


def _replicated_tstats(panel: pl.DataFrame) -> pl.DataFrame:
    """Per-predictor in-sample t of the ``op`` long-short return.

    ``mean / (sd / sqrt(n))`` over the predictor's ``window == "in_sample"``
    rows, sample sd (``ddof=1``). Null when fewer than two non-null returns are
    available or their sd is zero, so the statistic is always finite.
    """
    stats = (
        panel.filter((pl.col("portfolio") == "op") & (pl.col("window") == "in_sample"))
        .group_by("signalname")
        .agg(
            pl.col("ret").mean().alias("mean_ret"),
            pl.col("ret").std().alias("sd_ret"),
            pl.col("ret").count().alias("n_months"),
        )
    )
    return stats.select(
        "signalname",
        pl.when((pl.col("n_months") >= 2) & (pl.col("sd_ret") > 0.0))
        .then(pl.col("mean_ret") / (pl.col("sd_ret") / pl.col("n_months").sqrt()))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("t_replicated"),
    )


def tstats(
    panel: pl.DataFrame, signal_doc_path: Path, *, hurdle_t: float = DEFAULT_HURDLE_T
) -> pl.DataFrame:
    """Per-predictor t-statistics from both pre-registered sources, with weak flags.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA` and the
    canonical OSAP signal-doc CSV. Output: one row per signalname present in
    *panel*, sorted by signalname, exactly :data:`TSTATS_SCHEMA`: ``t_original``
    (parsed ``T-Stat``; null and counted when missing, non-numeric or absent),
    ``t_replicated`` (see :func:`_replicated_tstats`), and ``weak_original`` /
    ``weak_replicated`` — True when the t-statistic is **strictly below**
    *hurdle_t*, null exactly when the t-statistic is null. The sign of the
    reported t is kept as published (no absolute value).
    """
    original = _original_tstats(Path(signal_doc_path))
    replicated = _replicated_tstats(panel)
    predictors = pl.DataFrame({"signalname": panel["signalname"].unique().sort()})
    joined = (
        predictors.join(original, on="signalname", how="left")
        .join(replicated, on="signalname", how="left")
        .with_columns(
            (pl.col("t_original") < hurdle_t).alias("weak_original"),
            (pl.col("t_replicated") < hurdle_t).alias("weak_replicated"),
        )
        .select(list(TSTATS_SCHEMA))
    )
    return joined


def tstat_summary(stats: pl.DataFrame, *, hurdle_t: float = DEFAULT_HURDLE_T) -> pl.DataFrame:
    """One ``tstat_summary.csv`` row per source over the predictors in *stats*.

    Input: the output of :func:`tstats`. Output: rows per
    :data:`TSTAT_SUMMARY_SCHEMA`, sorted by ``tstat_source``. ``n_predictors`` is
    the number of predictors in *stats*; ``n_missing`` counts null t-statistics
    (empty, non-numeric or absent entries); ``n_below_hurdle`` counts weak
    predictors (strictly below *hurdle_t* — for the weak flags this equals the
    weak count, since nulls are neither weak nor counted); ``median_t`` is the
    median of the non-missing values (null when every value is missing);
    ``hurdle_t`` records the hurdle used.
    """
    missing_columns = [
        column for column, _ in TSTAT_SOURCE_COLUMNS.values() if column not in stats.columns
    ]
    if missing_columns:
        raise TstatError(f"tstat table is missing columns {missing_columns}")
    rows: list[dict[str, object]] = []
    for source in TSTAT_SOURCES:
        column, _ = TSTAT_SOURCE_COLUMNS[source]
        values = stats[column]
        rows.append(
            {
                "tstat_source": source,
                "n_predictors": stats.height,
                "n_missing": values.null_count(),
                "n_below_hurdle": int((values < hurdle_t).sum()),
                "median_t": values.median(),
                "hurdle_t": hurdle_t,
            }
        )
    frame = pl.DataFrame(rows, schema=RESULTS_SCHEMAS["tstat_summary.csv"])
    return frame.sort("tstat_source")


def _interaction_expressions(
    extra_terms: Sequence[str], t_column: str, weak_column: str
) -> dict[str, pl.Expr]:
    """Interaction columns for the spec's extra terms; unknown terms raise."""
    expressions: dict[str, pl.Expr] = {}
    for term in extra_terms:
        kind = _INTERACTION_KINDS.get(term)
        if kind is None:
            raise EstimationError(f"unknown extra term {term!r}")
        multiplier = t_column if kind == "tstat" else weak_column
        base_term = term.split("_x_", 1)[0]
        expressions[term] = pl.col(base_term) * pl.col(multiplier).cast(pl.Float64)
    return expressions


def run_multiple_testing_spec(
    panel: pl.DataFrame,
    stats: pl.DataFrame,
    prereg_spec: MultipleTestingSpecPrereg,
    base_spec: Spec,
) -> pl.DataFrame:
    """Estimate one pre-registered multiple-testing specification.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA`; *stats* as
    returned by :func:`tstats` (built with the pre-registered hurdle);
    *prereg_spec* one
    :class:`~anomaly_decay.config.MultipleTestingSpecPrereg`; *base_spec* the
    resolved Phase 1 specification named by ``prereg_spec.base_spec``.

    The estimation sample is the base specification's sample (through
    :func:`anomaly_decay.estimate.main._filter_sample`), joined with the t and
    weak columns of the spec's source. Predictors whose t is missing are
    excluded as whole predictors and counted in ``n_excluded_missing_t``. The
    regression is ``ret ~ post_sample + post_publication + interactions`` with
    predictor effects and the base specification's covariance (month-clustered
    for ``main``), fitted through
    :func:`anomaly_decay.estimate.main._fit` so the covariance estimator is the
    frozen one.

    Output: rows per ``RESULTS_SCHEMAS["multiple_testing.csv"]``; ``n_weak``
    counts the estimation-sample predictors whose flag is set (for the
    continuous specification it is the descriptive count below the hurdle, not a
    regressor). Raises
    :class:`~anomaly_decay.estimate.main.EstimationError` when the sample is
    empty, a term has no variation, or the source/terms are not pre-registered.
    """
    spec_id = prereg_spec.id
    if base_spec.dep_var != "ret":
        raise EstimationError(
            f"specification '{spec_id}': multiple-testing specifications require "
            f"dep_var 'ret', got {base_spec.dep_var!r}"
        )
    try:
        t_column, weak_column = TSTAT_SOURCE_COLUMNS[prereg_spec.tstat_source]
    except KeyError:
        raise EstimationError(
            f"specification '{spec_id}': unknown tstat source {prereg_spec.tstat_source!r}"
        ) from None

    sample = _filter_sample(panel, base_spec)
    sample = sample.join(
        stats.select("signalname", t_column, weak_column), on="signalname", how="left"
    )
    n_excluded = int(
        sample.filter(pl.col(t_column).is_null()).select(pl.col("signalname").n_unique()).item()
    )
    sample = sample.filter(pl.col(t_column).is_not_null() & pl.col(base_spec.dep_var).is_not_null())
    if sample.height == 0:
        raise EstimationError(f"specification '{spec_id}': estimation sample is empty")

    post_sample_column = "post_sample_alt" if base_spec.convention == "alt" else "post_sample"
    post_publication_column = (
        "post_publication_alt" if base_spec.convention == "alt" else "post_publication"
    )
    sample = sample.with_columns(
        pl.col(post_sample_column).cast(pl.Float64).alias("post_sample"),
        pl.col(post_publication_column).cast(pl.Float64).alias("post_publication"),
    )
    terms: dict[str, pl.Expr] = {
        "post_sample": pl.col("post_sample"),
        "post_publication": pl.col("post_publication"),
    }
    try:
        terms.update(_interaction_expressions(prereg_spec.extra_terms, t_column, weak_column))
    except EstimationError as exc:
        raise EstimationError(f"specification '{spec_id}': {exc}") from None
    for name, expression in terms.items():
        sample = sample.with_columns(expression.alias(name))

    term_names = list(terms)
    for term in term_names:
        if sample[term].n_unique() < 2:
            raise EstimationError(
                f"specification '{spec_id}': term '{term}' has no variation "
                "in the estimation sample"
            )

    pdf = sample.select(["signalname", "month", base_spec.dep_var, *term_names]).to_pandas()
    pdf = pdf.set_index(["signalname", "month"])
    try:
        model = PanelOLS(pdf[base_spec.dep_var], pdf[term_names], entity_effects=True)
        results = _fit(model, base_spec)
    except (AbsorbingEffectError, ValueError, np.linalg.LinAlgError) as exc:
        raise EstimationError(f"specification '{spec_id}': {exc}") from exc

    n_predictors = sample["signalname"].n_unique()
    n_obs = sample.height
    n_weak = int(
        sample.group_by("signalname")
        .agg(pl.col(weak_column).first())
        .select(pl.col(weak_column).sum())
        .item()
    )
    rows = [
        {
            "spec": spec_id,
            "tstat_source": prereg_spec.tstat_source,
            "term": term,
            "coef": float(results.params[term]),
            "se": float(results.std_errors[term]),
            "t": float(results.tstats[term]),
            "n_predictors": n_predictors,
            "n_obs": n_obs,
            "n_weak": n_weak,
            "n_excluded_missing_t": n_excluded,
        }
        for term in term_names
    ]
    return pl.DataFrame(rows, schema=RESULTS_SCHEMAS["multiple_testing.csv"])


def run_multiple_testing(
    panel: pl.DataFrame,
    stats: pl.DataFrame,
    prereg: MultipleTestingPrereg,
    specs: Sequence[Spec],
) -> pl.DataFrame:
    """Estimate every pre-registered multiple-testing specification, in file order.

    Inputs: *panel*, *stats* (from :func:`tstats`), the ``multiple_testing``
    section of ``config/phase2_preregistration.yaml`` and the resolved Phase 1
    specifications the ``base_spec`` names refer to. Output: concatenated rows
    of :func:`run_multiple_testing_spec` per
    ``RESULTS_SCHEMAS["multiple_testing.csv"]``. An
    :class:`~anomaly_decay.estimate.main.EstimationError` (including an unknown
    base specification) propagates to the caller; the CLI runs the
    specifications itself so it can record failures and continue.
    """
    by_id = {spec.id: spec for spec in specs}
    frames: list[pl.DataFrame] = []
    for entry in prereg.specs:
        base_spec = by_id.get(entry.base_spec)
        if base_spec is None:
            raise EstimationError(
                f"specification '{entry.id}': unknown base specification {entry.base_spec!r}"
            )
        frames.append(run_multiple_testing_spec(panel, stats, entry, base_spec))
    return pl.concat(frames)


def _print_summary(summary: pl.DataFrame) -> None:
    """Print one line per t-statistic source with its counts and median."""
    for row in summary.iter_rows(named=True):
        median = "n/a" if row["median_t"] is None else f"{row['median_t']:.4f}"
        print(
            f"tstat_summary: {row['tstat_source']}: n_predictors={row['n_predictors']}, "
            f"n_missing={row['n_missing']}, n_below_hurdle={row['n_below_hurdle']}, "
            f"median_t={median}, hurdle_t={row['hurdle_t']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.estimate.multiple_testing --panel PATH
    --signal-doc PATH --prereg PATH --specs PATH --results-dir DIR``.

    Writes ``multiple_testing.csv`` (one row per term of every pre-registered
    specification that ran) and ``tstat_summary.csv`` under *results-dir*, and
    prints the per-source counts and the per-specification weak and exclusion
    counts. A specification that raises
    :class:`~anomaly_decay.estimate.main.EstimationError` is reported on stderr
    and does not stop the others; the exit code is 1 when any specification
    failed, 0 otherwise. ``multiple_testing.csv`` is written only when at least
    one specification ran.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.multiple_testing",
        description="Run the pre-registered multiple-testing specifications and write results.",
    )
    parser.add_argument("--panel", type=Path, required=True, help="panel.parquet path")
    parser.add_argument("--signal-doc", type=Path, required=True, help="OSAP signal doc CSV")
    parser.add_argument(
        "--prereg", type=Path, required=True, help="config/phase2_preregistration.yaml path"
    )
    parser.add_argument("--specs", type=Path, required=True, help="specifications.yaml path")
    parser.add_argument("--results-dir", type=Path, required=True, help="results output directory")
    args = parser.parse_args(argv)

    panel = pl.read_parquet(args.panel)
    specs = load_specifications(args.specs)
    prereg = load_phase2_preregistration(args.prereg).multiple_testing
    stats = tstats(panel, args.signal_doc, hurdle_t=prereg.hurdle_t)
    summary = tstat_summary(stats, hurdle_t=prereg.hurdle_t)

    frames: list[pl.DataFrame] = []
    failures: list[str] = []
    for entry in prereg.specs:
        base_spec = next((spec for spec in specs if spec.id == entry.base_spec), None)
        if base_spec is None:
            failures.append(
                f"specification '{entry.id}': unknown base specification {entry.base_spec!r}"
            )
            continue
        try:
            frames.append(run_multiple_testing_spec(panel, stats, entry, base_spec))
        except EstimationError as exc:
            failures.append(str(exc))

    if frames:
        table = pl.concat(frames)
        write_results(table, "multiple_testing.csv", args.results_dir)
        print(f"wrote multiple_testing.csv ({len(frames)} specifications, {table.height} rows)")
    for frame in frames:
        row = frame.row(0, named=True)
        print(
            f"{row['spec']}: n_predictors={row['n_predictors']}, n_weak={row['n_weak']}, "
            f"n_excluded_missing_t={row['n_excluded_missing_t']}"
        )
    write_results(summary, "tstat_summary.csv", args.results_dir)
    _print_summary(summary)
    for failure in failures:
        print(f"failed: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
