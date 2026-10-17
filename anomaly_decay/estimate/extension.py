"""Extension specifications (all predictors through the vintage's last month)."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from anomaly_decay.config import Spec
from anomaly_decay.estimate.main import _labelled_rows, run_spec
from anomaly_decay.schemas import RESULTS_SCHEMAS

__all__ = ["EXTENSION_GROUP", "run_extension"]

EXTENSION_GROUP = "extension"
"""Group collected in ``extension.csv``."""


def run_extension(panel: pl.DataFrame, specs: Sequence[Spec]) -> pl.DataFrame:
    """Estimate every extension specification in *specs*.

    Inputs: *panel* per :data:`anomaly_decay.schemas.PANEL_SCHEMA` and the
    specification entries (entries of other groups are ignored). Every
    ``group: extension`` entry is estimated like
    :func:`anomaly_decay.estimate.main.run_spec` — e.g. ``full_extension``
    through the pinned vintage's last month, ``ext_pub_after_2000`` with its
    ``post_publication_x_pub_after_2000`` interaction reported as its own term.
    Output rows per ``RESULTS_SCHEMAS["extension.csv"]``, labelled with the
    spec's ``label``, in the order given.

    Specifications are estimated once each; an
    :class:`~anomaly_decay.estimate.main.EstimationError` from a specification
    propagates to the caller. The CLI runs the specs itself (so it can record
    failures in ``specs_run.csv``) and writes the same rows.
    """
    frames = [
        _labelled_rows(spec, run_spec(panel, spec))
        for spec in specs
        if spec.group == EXTENSION_GROUP
    ]
    if not frames:
        return pl.DataFrame(schema=RESULTS_SCHEMAS["extension.csv"])
    return pl.concat(frames)
