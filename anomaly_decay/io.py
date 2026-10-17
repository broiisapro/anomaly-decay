"""Deterministic writers for project data files.

Every cached or committed table goes through :func:`write_csv_canonical` so that the
same input always produces byte-identical output (fixed column order, sorted by the
declared key columns, LF line endings, UTF-8, no index).

Results files (``results/*.csv``, docs/architecture.md §5.3) additionally go through
:func:`write_results`, which validates against :data:`anomaly_decay.schemas.RESULTS_SCHEMAS`
— ordering columns per the schema and rows per :data:`RESULTS_SORT_KEYS` — and
:func:`read_results`, the only supported reader. Both keep the schema dtypes on both
sides of the round trip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from polars.datatypes import DataTypeClass

from anomaly_decay.schemas import RESULTS_SCHEMAS, SchemaError, validate_frame

RESULTS_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "main.csv": ("spec", "term"),
    "decay.csv": ("spec", "method", "term"),
    "robustness.csv": ("spec", "term"),
    "extension.csv": ("spec", "term"),
    "specs_run.csv": ("spec",),
    "predictor_summary.csv": ("signalname", "window"),
    "event_time.csv": ("anchor", "event_month"),
    "coverage.csv": ("item",),
}
"""Row sort order for every file in :data:`~anomaly_decay.schemas.RESULTS_SCHEMAS`."""

if set(RESULTS_SORT_KEYS) != set(RESULTS_SCHEMAS):  # pragma: no cover - import-time guard
    raise RuntimeError("RESULTS_SORT_KEYS and RESULTS_SCHEMAS cover different files")


def write_csv_canonical(df: pl.DataFrame, path: Path, sort_by: Sequence[str]) -> None:
    """Write *df* as a canonical CSV, creating parent directories as needed.

    Columns keep the frame's order; rows are sorted by *sort_by* (accepted by
    ``DataFrame.sort``); line endings are LF and the encoding is UTF-8. The same input
    frame always produces byte-identical output.
    """
    missing = [column for column in sort_by if column not in df.columns]
    if missing:
        raise ValueError(f"sort_by columns not in frame: {missing}")
    out = df.select(df.columns).sort(list(sort_by))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, line_terminator="\n")


def _schema_for(name: str) -> Mapping[str, DataTypeClass]:
    if name not in RESULTS_SCHEMAS:
        known = sorted(RESULTS_SCHEMAS)
        raise SchemaError(f"unknown results file {name!r}; expected one of {known}")
    return RESULTS_SCHEMAS[name]


def _frame_with_schema_dtypes(
    df: pl.DataFrame, schema: Mapping[str, DataTypeClass]
) -> pl.DataFrame:
    """Return *df* with columns in schema order and exact dtypes, or raise.

    Extra columns are dropped; missing columns and columns whose dtype differs from the
    schema raise :class:`~anomaly_decay.schemas.SchemaError`, so nothing is cast
    silently. Nulls are cast explicitly so that an all-null column does not fall back
    to Polars' Null dtype.
    """
    missing = [column for column in schema if column not in df.columns]
    if missing:
        raise SchemaError(f"missing columns for results schema: {missing}")
    for column, dtype in schema.items():
        actual = df.schema[column]
        if actual != dtype:
            raise SchemaError(f"column '{column}': expected dtype {dtype}, got {actual}")
    out = df.select([pl.col(column).cast(dtype) for column, dtype in schema.items()])
    validate_frame(out, schema)
    return out


def write_results(df: pl.DataFrame, name: str, results_dir: Path) -> Path:
    """Write *df* as the results file *name* under *results_dir*; return its path.

    *name* must be a key of :data:`~anomaly_decay.schemas.RESULTS_SCHEMAS`. The frame
    must carry every schema column with the schema's dtype (anything else raises
    :class:`~anomaly_decay.schemas.SchemaError` — nothing is cast silently). Columns are
    reordered per the schema, rows are sorted by :data:`RESULTS_SORT_KEYS` (a stable
    sort, so rows that differ only in unkeyed columns keep the caller's order), and the
    file is written by :func:`write_csv_canonical`.
    """
    schema = _schema_for(name)
    ordered = _frame_with_schema_dtypes(df, schema)
    path = Path(results_dir) / name
    write_csv_canonical(ordered, path, sort_by=RESULTS_SORT_KEYS[name])
    return path


def read_results(name: str, results_dir: Path) -> pl.DataFrame:
    """Read results file *name* from *results_dir* with the schema's exact dtypes.

    *name* must be a key of :data:`~anomaly_decay.schemas.RESULTS_SCHEMAS`; nothing is
    inferred by the CSV parser. The parsed frame is validated against the schema.
    """
    schema = _schema_for(name)
    path = Path(results_dir) / name
    if not path.is_file():
        raise FileNotFoundError(path)
    overrides: dict[str, DataTypeClass] = {}
    for column, dtype in schema.items():
        if dtype == pl.Utf8:
            overrides[column] = pl.Utf8
        elif dtype == pl.Boolean:
            overrides[column] = pl.Boolean
    frame = pl.read_csv(path, schema_overrides=overrides)
    out = _frame_with_schema_dtypes(frame, schema)
    return out
