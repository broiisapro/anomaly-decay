"""Deterministic writers for project data files.

Every cached or committed table goes through :func:`write_csv_canonical` so that the
same input always produces byte-identical output (fixed column order, sorted by the
declared key columns, LF line endings, UTF-8, no index).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl


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
