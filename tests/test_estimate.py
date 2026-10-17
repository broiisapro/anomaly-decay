"""Tests for the specification runner (issue #17).

The central test reproduces linearmodels' month-clustered covariance by hand in
numpy; the others pin down the sample filters, the alternative window convention
and the failure mode for degenerate samples. ``make_panel`` and ``grid_records``
are shared with the robustness and extension test modules.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from anomaly_decay.config import Spec, load_specifications
from anomaly_decay.estimate.factors import FACTOR_COLUMNS, factor_adjusted_returns
from anomaly_decay.estimate.main import EstimationError, decay_table, run_spec
from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, RESULTS_SCHEMAS, validate_frame

REPO_ROOT = Path(__file__).parents[1]
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"


def load_spec(spec_id: str) -> Spec:
    """One resolved specification from the pre-registered file."""
    return next(spec for spec in load_specifications(SPECS_PATH) if spec.id == spec_id)


MAIN = load_spec("main")
NORMALIZED = load_spec("normalized")


def _window(post_sample: int, post_publication: int) -> str:
    if post_publication:
        return "post_publication"
    if post_sample:
        return "post_sample"
    return "in_sample"


def make_panel(records: list[dict[str, object]]) -> pl.DataFrame:
    """Build a schema-valid panel frame from compact per-row records.

    Required keys: ``signalname``, ``yyyymm`` (e.g. 196301) and ``ret``.
    Optional keys and defaults: ``portfolio`` (``op``), ``post_sample`` and
    ``post_publication`` (0), ``post_sample_alt``/``post_publication_alt``
    (mirroring the default convention), ``window``/``window_alt`` (inferred from
    the dummies), ``sample_start_year`` (1963), ``sample_end_year`` (1964),
    ``pub_year`` (1965) and ``clear`` (True). ``is_mean`` is the mean ``ret`` of
    the predictor's in-sample rows; ``ret_norm`` is ``ret / is_mean`` (null when
    the mean is non-positive). Factor columns are null.
    """
    in_sample: dict[str, list[float]] = {}
    for record in records:
        if not int(record.get("post_sample", 0)) and not int(record.get("post_publication", 0)):
            in_sample.setdefault(str(record["signalname"]), []).append(float(record["ret"]))

    rows: list[dict[str, object]] = []
    for record in records:
        signalname = str(record["signalname"])
        yyyymm = int(record["yyyymm"])
        ret = float(record["ret"])
        post_sample = int(record.get("post_sample", 0))
        post_publication = int(record.get("post_publication", 0))
        post_sample_alt = int(record.get("post_sample_alt", post_sample))
        post_publication_alt = int(record.get("post_publication_alt", post_publication))
        mean = statistics.fmean(in_sample[signalname]) if signalname in in_sample else None
        rows.append(
            {
                "signalname": signalname,
                "portfolio": str(record.get("portfolio", "op")),
                "yyyymm": yyyymm,
                "month": date(yyyymm // 100, yyyymm % 100, 1),
                "ret": ret,
                "window": str(record.get("window", _window(post_sample, post_publication))),
                "post_sample": post_sample,
                "post_publication": post_publication,
                "window_alt": str(
                    record.get("window_alt", _window(post_sample_alt, post_publication_alt))
                ),
                "post_sample_alt": post_sample_alt,
                "post_publication_alt": post_publication_alt,
                "event_time_sample_end": int(record.get("event_time_sample_end", 0)),
                "event_time_pub": int(record.get("event_time_pub", 0)),
                "is_mean": mean,
                "ret_norm": ret / mean if mean is not None and mean > 0 else None,
                "sample_start_year": int(record.get("sample_start_year", 1963)),
                "sample_end_year": int(record.get("sample_end_year", 1964)),
                "pub_year": int(record.get("pub_year", 1965)),
                "clear": bool(record.get("clear", True)),
                "mkt_rf": record.get("mkt_rf"),
                "smb": record.get("smb"),
                "hml": record.get("hml"),
                "rmw": record.get("rmw"),
                "cma": record.get("cma"),
                "umd": record.get("umd"),
                "rf": record.get("rf"),
            }
        )
    return pl.DataFrame(rows, schema=PANEL_SCHEMA)


def grid_records(
    n_signals: int,
    n_months: int = 48,
    *,
    start_index: int = 0,
    pub_year: int = 1965,
    sample_start_year: int = 1963,
    clear: bool = True,
    ret_offset: float = 0.0,
) -> list[dict[str, object]]:
    """Deterministic records: 1963-1964 in-sample, 1965 post-sample, 1966 post-publication.

    Returns are a fixed function of the predictor index and month, so every
    regression in the tests is exactly reproducible. *ret_offset* shifts a whole
    group of predictors to make leakage into the sample visible.
    """
    records: list[dict[str, object]] = []
    for index in range(start_index, start_index + n_signals):
        for month_index in range(n_months):
            year = 1963 + month_index // 12
            month = month_index % 12 + 1
            ret = (
                ret_offset
                + 0.5
                + 0.25 * index
                + 0.8 * (year == 1965)
                - 1.5 * (year == 1966)
                + ((index * 7 + month_index * 13) % 11 - 5) / 3.0
            )
            records.append(
                {
                    "signalname": f"S{index}",
                    "yyyymm": year * 100 + month,
                    "ret": ret,
                    "post_sample": int(year == 1965),
                    "post_publication": int(year == 1966),
                    "pub_year": pub_year,
                    "sample_start_year": sample_start_year,
                    "clear": clear,
                }
            )
    return records


def grid_panel(n_signals: int = 4, n_months: int = 48) -> pl.DataFrame:
    """``make_panel(grid_records(...))`` with the standard metadata."""
    return make_panel(grid_records(n_signals, n_months))


def decay_panel(n_signals: int = 4, n_months: int = 48) -> pl.DataFrame:
    """Known-effects panel: post-sample returns 30% below and post-publication 60% below.

    Predictor *index* has in-sample level ``1.0 + 0.4 * index``; 1965 is
    post-sample and 1966 post-publication with levels scaled by 0.7 and 0.4. The
    monthly wiggle is ``+/-0.1 * (index + 1)`` alternating, so it averages exactly
    zero within every 12-month window: the pooled in-sample mean is the average
    level (1.6) and the fixed-effects coefficients are -0.3 and -0.6 times it.
    """
    records: list[dict[str, object]] = []
    for index in range(n_signals):
        level = 1.0 + 0.4 * index
        for month_index in range(n_months):
            year = 1963 + month_index // 12
            month = month_index % 12 + 1
            if year <= 1964:
                factor = 1.0
            elif year == 1965:
                factor = 0.7
            else:
                factor = 0.4
            wiggle = 0.1 * (index + 1) * (1 if month_index % 2 == 0 else -1)
            records.append(
                {
                    "signalname": f"S{index}",
                    "yyyymm": year * 100 + month,
                    "ret": factor * level + wiggle,
                    "post_sample": int(year == 1965),
                    "post_publication": int(year == 1966),
                }
            )
    return make_panel(records)


def test_decay_table_recovers_known_effects() -> None:
    panel = decay_panel()
    coefs = run_spec(panel, MAIN)
    decay = decay_table(panel, MAIN, coefs)

    validate_frame(decay, RESULTS_SCHEMAS["decay.csv"])
    assert decay["method"].unique().to_list() == ["pooled_ratio"]
    pooled = {row["term"]: row for row in decay.iter_rows(named=True)}
    assert pooled["post_sample"]["in_sample_mean"] == pytest.approx(1.6, abs=1e-12)
    assert pooled["post_sample"]["decay_pct"] == pytest.approx(-30.0, abs=1e-9)
    assert pooled["post_publication"]["decay_pct"] == pytest.approx(-60.0, abs=1e-9)
    assert pooled["post_sample"]["n_predictors"] == 4
    assert pooled["post_sample"]["n_obs"] == 192
    coefficient_ses = coefs.sort("term")["se"].to_list()
    decay_ses = [pooled[term]["se_pct"] for term in sorted(pooled)]
    np.testing.assert_allclose(decay_ses, [100.0 * se / 1.6 for se in coefficient_ses])

    normalized = decay_table(panel, NORMALIZED, run_spec(panel, NORMALIZED))
    assert normalized["method"].unique().to_list() == ["normalized"]
    scaled = {row["term"]: row for row in normalized.iter_rows(named=True)}
    assert scaled["post_sample"]["in_sample_mean"] == 1.0
    assert scaled["post_sample"]["decay_pct"] == pytest.approx(-30.0, abs=1e-6)
    assert scaled["post_publication"]["decay_pct"] == pytest.approx(-60.0, abs=1e-6)


def test_decay_table_rejects_other_dependent_variables() -> None:
    panel = decay_panel()
    coefs = run_spec(panel, MAIN)
    with pytest.raises(EstimationError, match="decay requires"):
        decay_table(panel, replace(MAIN, dep_var="ret_factor_adj"), coefs)


FACTOR_ADJ = load_spec("factor_adj")


def _factor_value(name: str, month_index: int) -> float:
    if name == "mkt_rf":
        return math.sin(month_index / 3.0) + 0.1 * math.cos(month_index / 7.0)
    if name == "smb":
        return math.cos(month_index / 5.0)
    if name == "hml":
        return math.sin(month_index / 4.0) + 0.2
    if name == "rmw":
        return 0.5 * math.cos(month_index / 6.0)
    if name == "cma":
        return 0.7 * math.sin(month_index / 8.0)
    return math.cos(month_index / 9.0) + 0.3


def factor_records(
    n_signals: int = 3, n_months: int = 60, *, post_shift: float = 0.0
) -> list[dict[str, object]]:
    """Panel with an exact factor structure: ``ret = alpha + beta @ f`` on every row.

    In-sample months are 1963-1965 (36 per predictor, the factor module's
    minimum), post-sample is 1966 and post-publication 1967. With *post_shift*
    the stored factor values of the two post periods are shifted while ``ret``
    keeps using the unshifted factors, so the true betas stay the in-sample
    ones and post returns were generated out of the shifted sample.
    """
    records: list[dict[str, object]] = []
    for index in range(n_signals):
        alpha = 0.5 + 0.4 * index
        betas = {
            name: 0.2 + 0.1 * ((index + position) % 5)
            for position, name in enumerate(FACTOR_COLUMNS)
        }
        for month_index in range(n_months):
            year = 1963 + month_index // 12
            month = month_index % 12 + 1
            base = {name: _factor_value(name, month_index) for name in FACTOR_COLUMNS}
            stored = {
                name: value + post_shift if year >= 1966 else value for name, value in base.items()
            }
            records.append(
                {
                    "signalname": f"S{index}",
                    "yyyymm": year * 100 + month,
                    "ret": alpha + sum(betas[name] * base[name] for name in FACTOR_COLUMNS),
                    "post_sample": int(year == 1966),
                    "post_publication": int(year == 1967),
                    "sample_end_year": 1965,
                    "pub_year": 1966,
                    **stored,
                }
            )
    return records


def _hand_betas(panel: pl.DataFrame, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """OLS of ``ret`` on a constant and the six factors, per predictor, by hand."""
    betas: dict[str, np.ndarray] = {}
    for signal in rows["signalname"].unique().sort().to_list():
        group = rows.filter(pl.col("signalname") == signal)
        design = group.select(FACTOR_COLUMNS).to_numpy().astype(np.float64)
        design = np.column_stack([design, np.ones(group.height)])
        y = group["ret"].to_numpy().astype(np.float64)
        betas[signal] = np.linalg.lstsq(design, y, rcond=None)[0]
    return betas


def test_factor_alphas_use_in_sample_betas_only() -> None:
    """Classic-bug guard: factor betas never see post-sample or post-publication data.

    Post-period factor values are shifted on a panel whose returns were
    generated from the unshifted series. The in-sample adjusted returns must not
    move, the post-period adjusted returns must equal
    ``ret - beta_in_sample @ f_shifted`` with the betas recomputed by hand from
    in-sample rows only, and the ``factor_adj`` regression must consume exactly
    those adjusted returns. A full-sample-beta implementation is shown to give
    different in-sample values, so the test would catch it.
    """
    panel = make_panel(factor_records(post_shift=0.0))
    perturbed = make_panel(factor_records(post_shift=7.0))
    merged = panel.join(factor_adjusted_returns(panel), on=list(PANEL_KEY), how="left")
    merged_perturbed = perturbed.join(
        factor_adjusted_returns(perturbed), on=list(PANEL_KEY), how="left"
    )

    in_sample = panel.filter(pl.col("window") == "in_sample")
    key = ["signalname", "yyyymm"]
    left = merged.filter(pl.col("window") == "in_sample").select(key + ["ret_factor_adj"])
    right = merged_perturbed.filter(pl.col("window") == "in_sample").select(
        key + ["ret_factor_adj"]
    )
    assert left.sort(key).equals(right.sort(key))

    # only the factor loadings enter the adjustment; the constant (last column
    # of the hand design) is not subtracted
    hand = {
        signal: beta[: len(FACTOR_COLUMNS)]
        for signal, beta in _hand_betas(panel, in_sample).items()
    }
    post = merged_perturbed.filter(pl.col("window") != "in_sample").sort(key)
    expected = [
        row["ret"] - float(np.dot(hand[row["signalname"]], [row[name] for name in FACTOR_COLUMNS]))
        for row in post.iter_rows(named=True)
    ]
    np.testing.assert_allclose(post["ret_factor_adj"].to_numpy(), expected, atol=1e-10)

    full = {
        signal: beta[: len(FACTOR_COLUMNS)]
        for signal, beta in _hand_betas(perturbed, perturbed).items()
    }
    naive = [
        row["ret"] - float(np.dot(full[row["signalname"]], [row[name] for name in FACTOR_COLUMNS]))
        for row in in_sample.sort(key).iter_rows(named=True)
    ]
    naive_diff = np.abs(np.array(naive) - left.sort(key)["ret_factor_adj"].to_numpy()).max()
    assert naive_diff > 1e-3

    sample = merged_perturbed.filter(
        (pl.col("portfolio") == "op")
        & (pl.col("pub_year") <= 2013)
        & (pl.col("yyyymm") >= 196307)
        & (pl.col("yyyymm") <= 201312)
        & pl.col("ret_factor_adj").is_not_null()
    )
    x = sample.select(["post_sample", "post_publication"]).to_numpy().astype(np.float64)
    y = sample["ret_factor_adj"].to_numpy().astype(np.float64)
    entity = sample["signalname"].to_numpy()
    for signal in np.unique(entity):
        mask = entity == signal
        x[mask] -= x[mask].mean(axis=0)
        y[mask] -= y[mask].mean(axis=0)
    manual_beta = np.linalg.lstsq(x, y, rcond=None)[0]

    result = run_spec(perturbed, FACTOR_ADJ)
    np.testing.assert_allclose(result["coef"].to_numpy(), manual_beta, atol=1e-10)


def test_run_spec_output_matches_results_schema() -> None:
    panel = grid_panel(n_signals=4)
    result = run_spec(panel, MAIN)

    validate_frame(result, RESULTS_SCHEMAS["main.csv"])
    assert result["spec"].unique().to_list() == ["main"]
    assert result["term"].to_list() == ["post_sample", "post_publication"]
    assert result["n_predictors"].to_list() == [4, 4]
    assert result["n_obs"].to_list() == [4 * 48, 4 * 48]
    np.testing.assert_allclose(result["t"].to_numpy(), result["coef"] / result["se"], atol=1e-12)


def test_clustered_se_matches_manual_computation() -> None:
    """The month-clustered SE equals a hand-written numpy sandwich.

    With ``PanelOLS(dep, exog, entity_effects=True)`` fitted with
    ``cov_type="clustered", cluster_time=True`` and linearmodels' defaults
    (``debiased=True``, ``auto_df=True``, ``count_effects=True``), the
    covariance is

        Cov = n / (n - d - k) * (X'X)^-1 [sum_g s_g s_g'] (X'X)^-1

    where s_g = sum_{i in g} x_i e_i is the sum of score vectors over the
    observations of calendar month g, x and e are the within-predictor demeaned
    regressors and residuals, k is the number of terms (2) and d is the number
    of predictors. Predictor effects are not nested in the time clusters, so
    linearmodels counts them in the degrees-of-freedom scale
    ``n / (n - d - k)``.
    """
    panel = grid_panel(n_signals=4, n_months=48)
    result = run_spec(panel, MAIN)

    frame = panel.sort(["signalname", "yyyymm"])
    x = frame.select(["post_sample", "post_publication"]).to_numpy().astype(np.float64)
    y = frame["ret"].to_numpy().astype(np.float64)
    entity = frame["signalname"].to_numpy()
    cluster = frame["yyyymm"].to_numpy()

    for name in np.unique(entity):
        mask = entity == name
        x[mask] -= x[mask].mean(axis=0)
        y[mask] -= y[mask].mean(axis=0)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    eps = y - x @ beta

    scores = x * eps[:, None]
    meat = np.zeros((x.shape[1], x.shape[1]))
    for period in np.unique(cluster):
        total = scores[cluster == period].sum(axis=0)
        meat += np.outer(total, total)

    n = y.size
    k = x.shape[1]
    d = np.unique(entity).size
    scale = n / (n - d - k)
    xtx_inv = np.linalg.inv(x.T @ x)
    cov = scale * xtx_inv @ meat @ xtx_inv

    np.testing.assert_allclose(result["coef"].to_numpy(), beta, atol=1e-10)
    np.testing.assert_allclose(result["se"].to_numpy(), np.sqrt(np.diag(cov)), atol=1e-8)


def test_universe_and_month_filters_restrict_the_sample() -> None:
    eligible = grid_records(2, start_index=0)
    ineligible = (
        grid_records(1, start_index=2, pub_year=2014, ret_offset=1000.0)
        + grid_records(1, start_index=3, sample_start_year=1950, ret_offset=1000.0)
        + grid_records(1, start_index=4, clear=False, ret_offset=1000.0)
    )
    mixed = make_panel(eligible + ineligible)
    only_eligible = make_panel(eligible)

    assert run_spec(mixed, MAIN)["n_predictors"].to_list() == [4, 4]
    assert run_spec(mixed, replace(MAIN, sample_start_year_min=1963))["n_predictors"].to_list() == [
        3,
        3,
    ]
    assert run_spec(mixed, replace(MAIN, clear_only=True))["n_predictors"].to_list() == [3, 3]

    fully_filtered = replace(MAIN, sample_start_year_min=1963, clear_only=True)
    result = run_spec(mixed, fully_filtered)
    expected = run_spec(only_eligible, fully_filtered)
    assert result["n_predictors"].to_list() == [2, 2]
    assert result["n_obs"].to_list() == [2 * 48, 2 * 48]
    np.testing.assert_allclose(result["coef"].to_numpy(), expected["coef"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(result["se"].to_numpy(), expected["se"].to_numpy(), atol=1e-12)

    cut = run_spec(mixed, replace(fully_filtered, end_yyyymm=196606))
    assert cut["n_obs"].unique().to_list() == [2 * 42]
    assert cut["n_predictors"].unique().to_list() == [2]


def test_portfolio_filter_selects_the_specified_series() -> None:
    records = [dict(record, portfolio="op") for record in grid_records(2, start_index=0)]
    records += [dict(record, portfolio="deciles_ew") for record in grid_records(2, start_index=2)]
    panel = make_panel(records)

    op = run_spec(panel, MAIN)
    assert op["n_predictors"].to_list() == [2, 2]

    ew_spec = replace(MAIN, portfolio="deciles_ew")
    ew = run_spec(panel, ew_spec)
    assert ew["n_predictors"].to_list() == [2, 2]
    assert ew["n_obs"].to_list() == [2 * 48, 2 * 48]
    assert set(ew["spec"].unique().to_list()) == {"main"}


def _alternative_convention_panel() -> pl.DataFrame:
    """Three predictors whose two window conventions differ in the panel.

    Samples end in 1963. Two predictors are published in 1965 and one in 1964:
    post-publication starts in 1965 for the former under the default convention
    (January of publication year + 1) and in 1965 for the latter under the
    alternative convention (January of the publication year).
    """
    records: list[dict[str, object]] = []
    for index, pub_year in enumerate((1964, 1965, 1965)):
        for month_index in range(36):
            year = 1963 + month_index // 12
            month = month_index % 12 + 1
            post_sample = int(year == 1964)
            post_publication = int(year == 1965 and pub_year == 1964)
            post_sample_alt = int(year == 1964 and pub_year > 1964)
            post_publication_alt = int(
                (year == 1964 and pub_year == 1964) or (year == 1965 and pub_year >= 1964)
            )
            ret = (
                0.5
                + 0.3 * index
                + 0.7 * post_sample
                - 1.1 * post_publication
                + ((index * 5 + month_index * 7) % 9 - 4) / 4.0
            )
            records.append(
                {
                    "signalname": f"S{index}",
                    "yyyymm": year * 100 + month,
                    "ret": ret,
                    "post_sample": post_sample,
                    "post_publication": post_publication,
                    "post_sample_alt": post_sample_alt,
                    "post_publication_alt": post_publication_alt,
                    "pub_year": pub_year,
                }
            )
    return make_panel(records)


def test_alt_convention_uses_alt_dummies_and_reports_default_terms() -> None:
    panel = _alternative_convention_panel()
    default = run_spec(panel, MAIN)
    alt = run_spec(panel, replace(MAIN, convention="alt"))

    assert alt["term"].to_list() == ["post_sample", "post_publication"]
    assert alt["n_obs"].unique().to_list() == [3 * 36]

    swapped = panel.with_columns(
        pl.col("post_sample_alt").alias("post_sample"),
        pl.col("post_publication_alt").alias("post_publication"),
    )
    equivalent = run_spec(swapped, MAIN)
    np.testing.assert_allclose(alt["coef"].to_numpy(), equivalent["coef"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(alt["se"].to_numpy(), equivalent["se"].to_numpy(), atol=1e-12)

    assert not np.allclose(alt["coef"].to_numpy(), default["coef"].to_numpy())


def test_degenerate_samples_raise_estimation_error() -> None:
    panel = grid_panel(n_signals=3)

    with pytest.raises(EstimationError, match="no variation"):
        run_spec(panel, replace(MAIN, end_yyyymm=196412))

    with pytest.raises(EstimationError, match="empty"):
        run_spec(panel, replace(MAIN, start_yyyymm=197001))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("cov", "cov_params"),
    [
        ("clustered_time", {}),
        ("clustered_twoway", {}),
        ("driscoll_kraay", {"kernel": "bartlett", "bandwidth": 12}),
    ],
)
def test_covariance_estimators_run(cov: str, cov_params: dict[str, object]) -> None:
    panel = grid_panel(n_signals=4)
    spec = replace(MAIN, cov=cov, cov_params=cov_params)
    result = run_spec(panel, spec)

    validate_frame(result, RESULTS_SCHEMAS["main.csv"])
    assert result["coef"].is_finite().all()
    if cov != "clustered_twoway":
        assert result["se"].is_finite().all()
        assert result["se"].min() > 0.0
    # The two-way CGM covariance (entity + month, minus their intersection) is not
    # guaranteed positive semi-definite, so a deterministic synthetic design can
    # yield a NaN standard error; that is the estimator's output, not a mapping
    # error, and the real-data robustness run covers the finite case.
