"""Tests for factor-adjusted returns (in-sample betas, no look-ahead)."""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
from anomaly_decay.estimate.factors import FACTOR_COLUMNS, factor_adjusted_returns

DEFAULT_BETAS: dict[str, float] = {
    "mkt_rf": 1.25,
    "smb": -0.5,
    "hml": 0.75,
    "rmw": 0.2,
    "cma": -0.3,
    "umd": 0.9,
}


def _months(first_yyyymm: int, count: int) -> list[int]:
    year, month = divmod(first_yyyymm, 100)
    out = []
    for _ in range(count):
        out.append(year * 100 + month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _factor_matrix(group: str, rows: int) -> np.ndarray:
    """A full-rank, reproducible factor matrix for one (signalname, portfolio) group."""
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    return np.random.default_rng(seed).normal(size=(rows, len(FACTOR_COLUMNS)))


def _panel(
    *,
    signalname: str = "Sig",
    portfolio: str = "op",
    n_in_sample: int = 60,
    n_post_sample: int = 6,
    n_post_publication: int = 12,
    betas: dict[str, float] | None = None,
    intercept: float = 0.0,
    pre_factor_row: bool = False,
) -> pl.DataFrame:
    """Single-group panel with ``ret = intercept + betas @ factors`` exactly.

    ``pre_factor_row`` prepends one in-sample month (198001) with null factors and a
    null return, as in the real panel before the FF5 start.
    """
    betas = betas or DEFAULT_BETAS
    months = _months(198002, n_in_sample + n_post_sample + n_post_publication)
    windows = (
        ["in_sample"] * n_in_sample
        + ["post_sample"] * n_post_sample
        + ["post_publication"] * n_post_publication
    )
    matrix = _factor_matrix(f"{signalname}/{portfolio}", len(months))
    rows: list[dict[str, object]] = []
    lead = [198001] if pre_factor_row else []
    for offset, yyyymm in enumerate(lead + months):
        index = offset - len(lead)
        if index < 0:
            rows.append(
                {
                    "signalname": signalname,
                    "portfolio": portfolio,
                    "yyyymm": yyyymm,
                    "ret": None,
                    "window": "in_sample",
                    **dict.fromkeys(FACTOR_COLUMNS, None),
                }
            )
            continue
        values = dict(zip(FACTOR_COLUMNS, matrix[index], strict=True))
        rows.append(
            {
                "signalname": signalname,
                "portfolio": portfolio,
                "yyyymm": yyyymm,
                "ret": intercept + sum(betas[factor] * value for factor, value in values.items()),
                "window": windows[index],
                **values,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "signalname": pl.Utf8,
            "portfolio": pl.Utf8,
            "yyyymm": pl.Int32,
            "ret": pl.Float64,
            "window": pl.Utf8,
            **dict.fromkeys(FACTOR_COLUMNS, pl.Float64),
        },
    )


def _join(panel: pl.DataFrame, adjusted: pl.DataFrame) -> pl.DataFrame:
    return panel.join(adjusted, on=["signalname", "portfolio", "yyyymm"], how="left")


def test_schema_and_row_count() -> None:
    panel = _panel()
    adjusted = factor_adjusted_returns(panel)
    assert adjusted.columns == ["signalname", "portfolio", "yyyymm", "ret_factor_adj"]
    assert adjusted.schema == {
        "signalname": pl.Utf8,
        "portfolio": pl.Utf8,
        "yyyymm": pl.Int32,
        "ret_factor_adj": pl.Float64,
    }
    assert adjusted.height == panel.height
    assert adjusted.equals(adjusted.sort(["signalname", "portfolio", "yyyymm"]))


def test_exact_beta_recovery_on_noiseless_predictor() -> None:
    betas = {"mkt_rf": 1.25, "smb": -0.5, "hml": 0.75, "rmw": 0.2, "cma": -0.3, "umd": 0.9}
    adjusted = factor_adjusted_returns(_panel(betas=betas, intercept=0.4))
    # ret - betas @ f = intercept exactly on every row that has all six factors
    assert adjusted["ret_factor_adj"].null_count() == 0
    assert np.allclose(adjusted["ret_factor_adj"].to_numpy(), 0.4, atol=1e-12)

    # out-of-sample returns that no longer follow the in-sample betas are not
    # force-fitted by the in-sample model: after a shift of -betas @ f the adjusted
    # return differs from the intercept by exactly -betas @ f
    strong = _panel(betas=betas, intercept=0.4)
    strong = strong.with_columns(
        (
            pl.col("ret")
            - pl.when(pl.col("window") != "in_sample")
            .then(sum(betas[f] * pl.col(f) for f in FACTOR_COLUMNS))
            .otherwise(0.0)
        ).alias("ret")
    )
    strong_adjusted = factor_adjusted_returns(strong)
    out = _join(strong, strong_adjusted).filter(pl.col("window") != "in_sample")
    expected_out = out.select(
        pl.lit(0.4) - sum(betas[f] * pl.col(f) for f in FACTOR_COLUMNS)
    ).to_series()
    assert np.allclose(out["ret_factor_adj"].to_numpy(), expected_out.to_numpy(), atol=1e-9)
    assert not np.allclose(out["ret_factor_adj"].to_numpy(), 0.4)


def test_betas_use_in_sample_rows_only() -> None:
    panel = _panel()
    baseline = factor_adjusted_returns(panel)

    corrupted_out_of_sample = panel.with_columns(
        pl.when(pl.col("window") != "in_sample")
        .then(pl.col("ret") + 5.0)
        .otherwise(pl.col("ret"))
        .alias("ret"),
        *[
            (
                pl.when(pl.col("window") != "in_sample")
                .then(pl.col(factor) * 3.0 + 1.0)
                .otherwise(pl.col(factor))
                .alias(factor)
            )
            for factor in FACTOR_COLUMNS
        ],
    )
    after = factor_adjusted_returns(corrupted_out_of_sample)

    in_sample_rows = pl.col("window") == "in_sample"
    baseline_in = _join(panel, baseline).filter(in_sample_rows)["ret_factor_adj"]
    after_in = _join(corrupted_out_of_sample, after).filter(in_sample_rows)["ret_factor_adj"]
    assert baseline_in.equals(after_in)

    # each out-of-sample row still applies the in-sample betas to its own factors;
    # with the noiseless panel the betas are the true ones, so subtract them by hand.
    joined = _join(corrupted_out_of_sample, after)
    expected_out = joined.select(
        pl.col("ret") - sum(DEFAULT_BETAS[factor] * pl.col(factor) for factor in FACTOR_COLUMNS)
    ).to_series()
    assert np.allclose(joined["ret_factor_adj"].to_numpy(), expected_out.to_numpy(), atol=1e-9)

    changed_in_sample = panel.with_columns(
        pl.when(in_sample_rows)
        .then(pl.col("mkt_rf") + 0.5)
        .otherwise(pl.col("mkt_rf"))
        .alias("mkt_rf")
    )
    moved = factor_adjusted_returns(changed_in_sample)
    assert not baseline["ret_factor_adj"].equals(moved["ret_factor_adj"])


def test_min_months_rule() -> None:
    at_threshold = factor_adjusted_returns(_panel(n_in_sample=36), min_months=36)
    assert at_threshold["ret_factor_adj"].null_count() == 0

    below = _panel(n_in_sample=35)
    assert factor_adjusted_returns(below, min_months=36)["ret_factor_adj"].null_count() == (
        below.height
    )
    assert factor_adjusted_returns(below, min_months=35)["ret_factor_adj"].null_count() == 0

    raised = _panel(n_in_sample=36)
    assert factor_adjusted_returns(raised, min_months=37)["ret_factor_adj"].null_count() == (
        raised.height
    )


def test_missing_factors_mean_null_and_do_not_count() -> None:
    with_pre = _panel(pre_factor_row=True)
    adjusted = factor_adjusted_returns(with_pre, min_months=36)
    assert adjusted.filter(pl.col("yyyymm") == 198001)["ret_factor_adj"].to_list() == [None]
    assert adjusted["ret_factor_adj"].null_count() == 1

    # the null-factor month does not count towards min_months: still 36 usable rows
    at_threshold = factor_adjusted_returns(_panel(pre_factor_row=True, n_in_sample=36))
    assert at_threshold["ret_factor_adj"].null_count() == 1

    one_short = factor_adjusted_returns(_panel(pre_factor_row=True, n_in_sample=35), min_months=36)
    assert one_short["ret_factor_adj"].null_count() == one_short.height


def test_non_in_sample_null_factors_are_null() -> None:
    panel = _panel().with_columns(
        pl.when(pl.col("window") == "post_publication")
        .then(None)
        .otherwise(pl.col("umd"))
        .alias("umd")
    )
    adjusted = _join(panel, factor_adjusted_returns(panel, min_months=36))
    nulls = adjusted.filter(pl.col("window") == "post_publication")["ret_factor_adj"]
    assert nulls.null_count() == nulls.len()
    assert adjusted.filter(pl.col("window") == "in_sample")["ret_factor_adj"].null_count() == 0


def test_multiple_predictors_and_portfolios_are_independent() -> None:
    specs = (
        ("A", "op", {"mkt_rf": 1.0, "smb": 0.1, "hml": 0.0, "rmw": 0.0, "cma": 0.0, "umd": 0.0}),
        (
            "A",
            "deciles_ew",
            {"mkt_rf": 0.5, "smb": -0.2, "hml": 0.3, "rmw": 0.1, "cma": 0.0, "umd": 0.0},
        ),
        ("B", "op", {"mkt_rf": -1.0, "smb": 0.0, "hml": 0.0, "rmw": 0.4, "cma": 0.2, "umd": 0.2}),
    )
    merged = pl.concat(
        [
            _panel(signalname=name, portfolio=portfolio, betas=betas, intercept=0.1)
            for name, portfolio, betas in specs
        ]
    ).sort(["signalname", "portfolio", "yyyymm"])
    adjusted = factor_adjusted_returns(merged, min_months=36)
    assert adjusted["ret_factor_adj"].null_count() == 0
    assert np.allclose(adjusted["ret_factor_adj"].to_numpy(), 0.1, atol=1e-12)
    assert set(adjusted["signalname"].unique().to_list()) == {"A", "B"}


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        factor_adjusted_returns(_panel().drop("umd"))
    with pytest.raises(ValueError, match="missing columns"):
        factor_adjusted_returns(_panel().drop("window"))
