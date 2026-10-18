"""Panel, predictor-metadata and results schemas (Polars) plus validators.

The schemas are the interface contract from docs/architecture.md §5.1–§5.3 and every
dtype, column name and column order is load-bearing for later epics. Results file
schemas are keyed by file name in :data:`RESULTS_SCHEMAS`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
from polars.datatypes import DataTypeClass

ALLOWED_WINDOWS: tuple[str, ...] = ("in_sample", "post_sample", "post_publication")

PREDICTORS_SCHEMA: dict[str, DataTypeClass] = {
    "signalname": pl.Utf8,
    "sample_start_year": pl.Int16,
    "sample_end_year": pl.Int16,
    "pub_year": pl.Int16,
    "journal": pl.Utf8,
    "authors": pl.Utf8,
    "cat_economic": pl.Utf8,
    "cat_data": pl.Utf8,
    "clear": pl.Boolean,
    "date_source": pl.Utf8,
}

PANEL_KEY: tuple[str, ...] = ("signalname", "portfolio", "yyyymm")

PANEL_SCHEMA: dict[str, DataTypeClass] = {
    "signalname": pl.Utf8,
    "portfolio": pl.Utf8,
    "yyyymm": pl.Int32,
    "month": pl.Date,
    "ret": pl.Float64,
    "window": pl.Utf8,
    "post_sample": pl.Int8,
    "post_publication": pl.Int8,
    "window_alt": pl.Utf8,
    "post_sample_alt": pl.Int8,
    "post_publication_alt": pl.Int8,
    "event_time_sample_end": pl.Int32,
    "event_time_pub": pl.Int32,
    "is_mean": pl.Float64,
    "ret_norm": pl.Float64,
    "sample_start_year": pl.Int16,
    "sample_end_year": pl.Int16,
    "pub_year": pl.Int16,
    "clear": pl.Boolean,
    "mkt_rf": pl.Float64,
    "smb": pl.Float64,
    "hml": pl.Float64,
    "rmw": pl.Float64,
    "cma": pl.Float64,
    "umd": pl.Float64,
    "rf": pl.Float64,
}

RESULTS_SCHEMAS: dict[str, dict[str, DataTypeClass]] = {
    "main.csv": {
        "spec": pl.Utf8,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
    },
    "decay.csv": {
        "spec": pl.Utf8,
        "method": pl.Utf8,
        "term": pl.Utf8,
        "decay_pct": pl.Float64,
        "se_pct": pl.Float64,
        "in_sample_mean": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
    },
    "robustness.csv": {
        "spec": pl.Utf8,
        "label": pl.Utf8,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
    },
    "extension.csv": {
        "spec": pl.Utf8,
        "label": pl.Utf8,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
    },
    "specs_run.csv": {
        "spec": pl.Utf8,
        "label": pl.Utf8,
        "status": pl.Utf8,
        "exploratory": pl.Boolean,
    },
    "predictor_summary.csv": {
        "signalname": pl.Utf8,
        "window": pl.Utf8,
        "n_months": pl.Int64,
        "mean_ret": pl.Float64,
        "se_mean": pl.Float64,
    },
    "event_time.csv": {
        "anchor": pl.Utf8,
        "event_month": pl.Int64,
        "mean_ret": pl.Float64,
        "se": pl.Float64,
        "n_predictors": pl.Int64,
    },
    "coverage.csv": {
        "item": pl.Utf8,
        "value": pl.Utf8,
    },
    "bootstrap.csv": {
        "spec": pl.Utf8,
        "statistic": pl.Utf8,
        "estimate": pl.Float64,
        "boot_mean": pl.Float64,
        "boot_se": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "reps": pl.Int64,
        "block_length": pl.Int64,
        "seed": pl.Int64,
    },
    "multiple_testing.csv": {
        "spec": pl.Utf8,
        "tstat_source": pl.Utf8,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
        "n_weak": pl.Int64,
        "n_excluded_missing_t": pl.Int64,
    },
    "tstat_summary.csv": {
        "tstat_source": pl.Utf8,
        "n_predictors": pl.Int64,
        "n_missing": pl.Int64,
        "n_below_hurdle": pl.Int64,
        "median_t": pl.Float64,
        "hurdle_t": pl.Float64,
    },
    "vintage_comparison.csv": {
        "spec": pl.Utf8,
        "vintage": pl.Int64,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "decay_pct": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
        "predictor_set": pl.Utf8,
    },
}


class SchemaError(ValueError):
    """Raised when a frame does not match the contract in docs/architecture.md §5."""


def validate_frame(
    df: pl.DataFrame,
    schema: Mapping[str, DataTypeClass],
    *,
    key: Sequence[str] | None = None,
) -> None:
    """Validate *df* against *schema*; raise :class:`SchemaError` on the first problem.

    Checks, in order: exact column names and order, exact dtypes, non-null key, key
    uniqueness, key sortedness (when *key* is given), and — for any ``window`` or
    ``window_alt`` column — that values are one of
    ``in_sample | post_sample | post_publication``.
    """
    expected_columns = list(schema)
    if df.columns != expected_columns:
        raise SchemaError(f"columns differ: expected {expected_columns}, got {df.columns}")
    for column, dtype in schema.items():
        actual = df.schema[column]
        if actual != dtype:
            raise SchemaError(f"column '{column}': expected dtype {dtype}, got {actual}")

    for column in ("window", "window_alt"):
        if column in schema:
            invalid = sorted(set(df[column].unique().to_list()) - set(ALLOWED_WINDOWS))
            if invalid:
                raise SchemaError(
                    f"column '{column}': values outside {list(ALLOWED_WINDOWS)}: {invalid}"
                )

    if key is not None:
        key_columns = list(key)
        missing = [column for column in key_columns if column not in df.columns]
        if missing:
            raise SchemaError(f"key columns not in frame: {missing}")
        nulls = df.select(
            pl.any_horizontal([pl.col(column).is_null() for column in key_columns]).sum()
        ).item()
        if nulls:
            raise SchemaError(f"key {key_columns} contains {nulls} null row(s)")
        if df.select(key_columns).n_unique() != df.height:
            raise SchemaError(f"key {key_columns} is not unique")
        sorted_keys = df.select(key_columns).sort(key_columns)
        if not df.select(key_columns).equals(sorted_keys):
            raise SchemaError(f"frame is not sorted by key {key_columns}")
