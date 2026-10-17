"""Extension specifications (all predictors through the vintage's last month).

Interface stub: implemented in Epic 01 workstream B.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from anomaly_decay.config import Spec

__all__ = ["run_extension"]


def run_extension(panel: pl.DataFrame, specs: Sequence[Spec]) -> pl.DataFrame:
    """Estimate every extension specification in *specs*.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA` and the extension
    entries of ``config/specifications.yaml`` (e.g. ``full_extension`` through the
    pinned vintage's last month, ``ext_pub_after_2000`` with its extra interaction
    term). Window rules per architecture §4.6; coefficients in percent per month.

    Output rows per ``RESULTS_SCHEMAS["extension.csv"]``: ``spec, label, term, coef,
    se, t, n_predictors, n_obs``.
    """
    raise NotImplementedError
