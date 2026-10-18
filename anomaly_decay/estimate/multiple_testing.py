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
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import polars as pl
from polars.datatypes import DataTypeClass

__all__ = [
    "DEFAULT_HURDLE_T",
    "TSTAT_SOURCE_COLUMNS",
    "TSTAT_SOURCES",
    "TSTAT_SUMMARY_SCHEMA",
    "TstatError",
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

TSTAT_SUMMARY_SCHEMA: dict[str, DataTypeClass] = {
    "tstat_source": pl.Utf8,
    "n_predictors": pl.Int64,
    "n_missing": pl.Int64,
    "n_below_hurdle": pl.Int64,
    "median_t": pl.Float64,
    "hurdle_t": pl.Float64,
}
"""Rows of ``tstat_summary.csv`` (also added to ``RESULTS_SCHEMAS`` in Phase 2)."""


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
    frame = pl.DataFrame(rows, schema=TSTAT_SUMMARY_SCHEMA)
    return frame.sort("tstat_source")
