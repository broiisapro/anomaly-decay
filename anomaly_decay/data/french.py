"""Parse Ken French Data Library factor files (CSV inside zip), monthly block only.

The source files are CRLF, start with a few note lines, then a header line beginning
with a comma, then monthly rows ``YYYYMM,values...``, then a blank line and an
"Annual Factors" block, then a trailing copyright line. Only the monthly block is
returned; rows are labelled by ``yyyymm`` (``YYYYMM`` as Int32). Values remain in
**percent per month**. Missing-value codes ``-99.99`` and ``-999`` become null.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

import polars as pl

FF5_RENAME: dict[str, str] = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": "rf",
}

UMD_RENAME: dict[str, str] = {"Mom": "umd"}

MISSING_CODES = frozenset({-99.99, -999.0})


class FrenchParseError(ValueError):
    """Raised when a Ken French file does not have the expected structure."""


def parse_french_csv(text: str, rename: Mapping[str, str]) -> pl.DataFrame:
    """Parse the monthly block of a Ken French CSV.

    Returns columns ``yyyymm`` (Int32) followed by the renamed factor columns
    (Float64, percent per month). Blank lines, the annual block and the trailing
    copyright line end the monthly block and are ignored.
    """
    lines = text.splitlines()
    header_index: int | None = None
    for index, line in enumerate(lines):
        if line.startswith(","):
            header_index = index
            break
    if header_index is None:
        raise FrenchParseError("no header line starting with ',' found")

    raw_columns = [name.strip() for name in lines[header_index].split(",")[1:]]
    unknown = [name for name in raw_columns if name not in rename]
    if unknown:
        raise FrenchParseError(f"unexpected factor columns: {unknown}")
    columns = [rename[name] for name in raw_columns]

    months: list[int] = []
    values: dict[str, list[float | None]] = {column: [] for column in columns}
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        fields = [field.strip() for field in stripped.split(",")]
        label = fields[0]
        if len(fields) != len(columns) + 1 or len(label) != 6 or not label.isdigit():
            break
        months.append(int(label))
        for column, field in zip(columns, fields[1:], strict=True):
            value = float(field)
            values[column].append(None if value in MISSING_CODES else value)

    if not months:
        raise FrenchParseError("no monthly rows found")
    if len(set(months)) != len(months):
        raise FrenchParseError("duplicate months in monthly block")

    data: dict[str, pl.Series] = {"yyyymm": pl.Series("yyyymm", months, dtype=pl.Int32)}
    for column in columns:
        data[column] = pl.Series(column, values[column], dtype=pl.Float64)
    return pl.DataFrame(data)


def parse_french_zip(data: bytes, rename: Mapping[str, str]) -> pl.DataFrame:
    """Parse the first CSV member of a Ken French zip archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not names:
                raise FrenchParseError("zip archive contains no CSV member")
            text = archive.read(names[0]).decode("utf-8")
    except zipfile.BadZipFile as exc:
        raise FrenchParseError(f"not a zip archive: {exc}") from exc
    return parse_french_csv(text, rename)


def load_french_zip(path: Path, rename: Mapping[str, str]) -> pl.DataFrame:
    """Read and parse a Ken French zip file from disk."""
    return parse_french_zip(Path(path).read_bytes(), rename)
