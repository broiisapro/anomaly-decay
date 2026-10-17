"""Robustness specifications (alternative covariance, universe, portfolio, dep var)."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from anomaly_decay.config import Spec
from anomaly_decay.estimate.main import _labelled_rows, run_spec
from anomaly_decay.schemas import RESULTS_SCHEMAS

__all__ = ["ROBUSTNESS_GROUPS", "run_robustness"]

ROBUSTNESS_GROUPS: tuple[str, ...] = ("main", "robustness")
"""Groups collected in ``robustness.csv``; ``main`` is the reference row."""


def run_robustness(panel: pl.DataFrame, specs: Sequence[Spec]) -> pl.DataFrame:
    """Estimate every robustness specification in *specs*.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA` and the
    specification entries (entries of other groups are ignored). The output
    holds ``main`` for reference plus every ``group: robustness`` entry, each
    estimated like :func:`anomaly_decay.estimate.main.run_spec` (window rules
    per architecture §4.6; coefficients in percent per month) and labelled with
    the spec's ``label``. Rows follow ``RESULTS_SCHEMAS["robustness.csv"]``.

    Specifications are estimated once each, in the order given; an
    :class:`~anomaly_decay.estimate.main.EstimationError` from a specification
    propagates to the caller. The CLI runs the specs itself (so it can record
    failures in ``specs_run.csv``) and writes the same rows.
    """
    frames = [
        _labelled_rows(spec, run_spec(panel, spec))
        for spec in specs
        if spec.group in ROBUSTNESS_GROUPS
    ]
    if not frames:
        return pl.DataFrame(schema=RESULTS_SCHEMAS["robustness.csv"])
    return pl.concat(frames)
