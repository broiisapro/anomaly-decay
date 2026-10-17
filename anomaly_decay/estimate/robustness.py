"""Robustness specifications (alternative covariance, universe, portfolio, dep var).

Interface stub: implemented in Epic 01 workstream B.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from anomaly_decay.config import Spec

__all__ = ["run_robustness"]


def run_robustness(panel: pl.DataFrame, specs: Sequence[Spec]) -> pl.DataFrame:
    """Estimate every robustness specification in *specs*.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA` and the robustness
    entries of ``config/specifications.yaml``. Each specification is estimated like
    :func:`anomaly_decay.estimate.main.run_spec` (window rules per architecture §4.6;
    coefficients in percent per month).

    Output rows per ``RESULTS_SCHEMAS["robustness.csv"]``: ``spec, label, term, coef,
    se, t, n_predictors, n_obs``.
    """
    raise NotImplementedError
