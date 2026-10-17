"""Factor-adjusted long-short returns (architecture §4.9, decision O10).

For every (signalname, portfolio) the six FF5 + momentum factors are regressed on the
predictor's long-short return using **in-sample months only** — the in-sample window is
convention-independent, so beta estimation never sees post-sample or post-publication
returns (the classic look-ahead trap). A month enters the regression only when ``ret``
and all six factors are present, which in practice drops everything before 1963-07 (the
FF5 start). Portfolios with fewer than ``min_months`` usable in-sample months get no
betas, so every ``ret_factor_adj`` of theirs is null.

The adjusted return is ``ret - beta @ f``: the intercept is not subtracted, so every
adjusted row keeps the predictor's average return (the in-sample mean of the adjusted
series is the regression's alpha, not zero). ``rf`` is not subtracted either — OSAP
long-short portfolios are already zero-investment, so the long-short return is itself
the excess return. Units stay percent per month.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from anomaly_decay.schemas import PANEL_KEY

FACTOR_COLUMNS: tuple[str, ...] = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")
"""Ken French regressors, in the design-matrix column order."""

_BETA_COLUMNS: tuple[str, ...] = tuple(f"beta_{name}" for name in FACTOR_COLUMNS)
_GROUP_COLUMNS: tuple[str, ...] = PANEL_KEY[:2]
_REQUIRED_COLUMNS: tuple[str, ...] = (*PANEL_KEY, "ret", "window", *FACTOR_COLUMNS)


def factor_adjusted_returns(panel: pl.DataFrame, *, min_months: int = 36) -> pl.DataFrame:
    """Per-predictor factor-adjusted returns; returns ``PANEL_KEY + ret_factor_adj``.

    *panel* must follow :data:`anomaly_decay.schemas.PANEL_SCHEMA`. For each
    (signalname, portfolio) group the betas come from an OLS of ``ret`` on a constant
    and ``mkt_rf, smb, hml, rmw, cma, umd`` over that group's **in-sample** rows whose
    ``ret`` and six factors are all non-null; a group with fewer than *min_months* such
    rows gets no betas. ``ret_factor_adj = ret - beta @ f`` (intercept not subtracted)
    for every row with all six factors present, and null everywhere else. The frame is
    sorted by :data:`~anomaly_decay.schemas.PANEL_KEY`.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing columns required for factor adjustment: {missing}")

    betas = _betas(panel, min_months=min_months)
    adjusted = panel.join(betas, on=list(_GROUP_COLUMNS), how="left").with_columns(
        (pl.col("ret") - _factor_dot()).alias("ret_factor_adj")
    )
    return adjusted.select(*PANEL_KEY, "ret_factor_adj").sort(list(PANEL_KEY))


def _betas(panel: pl.DataFrame, *, min_months: int) -> pl.DataFrame:
    """One beta row per eligible (signalname, portfolio) group."""
    usable = panel.filter(
        (pl.col("window") == "in_sample")
        & pl.col("ret").is_not_null()
        & pl.all_horizontal(pl.col(FACTOR_COLUMNS).is_not_null())
    )
    counts = usable.group_by(*_GROUP_COLUMNS).agg(pl.len().alias("n_usable"))
    eligible = counts.filter(pl.col("n_usable") >= min_months).select(_GROUP_COLUMNS)

    records: list[dict[str, object]] = []
    if eligible.height:
        for (signalname, portfolio), group in usable.join(
            eligible, on=list(_GROUP_COLUMNS), how="inner"
        ).group_by(_GROUP_COLUMNS, maintain_order=True):
            design = np.column_stack(
                [group[factor].to_numpy() for factor in FACTOR_COLUMNS] + [np.ones(group.height)]
            )
            beta, *_ = np.linalg.lstsq(design, group["ret"].to_numpy(), rcond=None)
            records.append(
                {
                    "signalname": signalname,
                    "portfolio": portfolio,
                    **dict(zip(_BETA_COLUMNS, beta[:-1], strict=True)),
                }
            )
    return pl.DataFrame(
        records,
        schema={
            "signalname": pl.Utf8,
            "portfolio": pl.Utf8,
            **dict.fromkeys(_BETA_COLUMNS, pl.Float64),
        },
    )


def _factor_dot() -> pl.Expr:
    """``beta_mkt_rf * mkt_rf + ... + beta_umd * umd``; null if any factor is null."""
    dot = pl.lit(0.0)
    for factor, beta in zip(FACTOR_COLUMNS, _BETA_COLUMNS, strict=True):
        dot = dot + pl.col(beta) * pl.col(factor)
    return dot
