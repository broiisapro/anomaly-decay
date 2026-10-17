"""Predictor metadata table (OSAP signal doc + cited supplement).

``build_metadata`` turns the canonical OSAP signal doc into the predictor table of
docs/architecture.md §5.1: one row per ``Cat.Signal == "Predictor"``, sorted by
``signalname``, exactly :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`. Missing
calendar years, journals, authors or categories are filled only from
``config/predictor_dates_supplement.csv``, one cited source per row; a field that is
already present is never overwritten. ``date_source`` records provenance:
``signaldoc`` or ``supplement:<citation key>`` (keys joined with ``+`` when several
were used for one predictor).

Validation is loud: duplicate signal names, a required field still missing after the
supplement, ``sample_start_year > sample_end_year`` or ``pub_year < sample_end_year``
raise :class:`MetadataError`; a predictor published in the same year its sample ends
is only a :class:`MetadataWarning`.

CLI::

    python -m anomaly_decay.data.metadata --signal-doc PATH [--supplement PATH] --out PATH
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from anomaly_decay.schemas import PREDICTORS_SCHEMA, SchemaError, validate_frame

__all__ = ["MetadataError", "MetadataWarning", "build_metadata", "main"]

SIGNAL_DOC_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Acronym",
    "Cat.Signal",
    "Predictability in OP",
    "SampleStartYear",
    "SampleEndYear",
    "Year",
    "Journal",
    "Authors",
    "Cat.Economic",
    "Cat.Data",
)
YEAR_FIELDS: tuple[str, ...] = ("sample_start_year", "sample_end_year", "pub_year")
TEXT_FIELDS: tuple[str, ...] = ("journal", "authors", "cat_economic", "cat_data")
SUPPLEMENT_FIELDS: frozenset[str] = frozenset((*YEAR_FIELDS, *TEXT_FIELDS))
SUPPLEMENT_COLUMNS: tuple[str, ...] = (
    "signalname",
    "field",
    "value",
    "citation_key",
    "citation",
)
_YEAR_RE = re.compile(r"^[0-9]+$")
_INT16_MIN = -32768
_INT16_MAX = 32767


class MetadataError(ValueError):
    """Raised when the signal doc or its supplement cannot produce a valid table."""


class MetadataWarning(UserWarning):
    """Emitted for suspicious-but-valid metadata (e.g. pub year equals sample end)."""


def _parse_year(value: object, where: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _YEAR_RE.match(text):
        raise MetadataError(f"{where}: not an integer year: {value!r}")
    year = int(text)
    if not _INT16_MIN <= year <= _INT16_MAX:
        raise MetadataError(f"{where}: year outside Int16 range: {value!r}")
    return year


def _parse_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _signal_doc_records(signal_doc_path: Path) -> list[dict[str, object]]:
    path = Path(signal_doc_path)
    if not path.is_file():
        raise MetadataError(f"signal doc not found: {path}")
    raw = pl.read_csv(path, infer_schema_length=0)
    missing = [column for column in SIGNAL_DOC_REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise MetadataError(f"signal doc {path}: missing columns {missing}")
    predictors = raw.filter(pl.col("Cat.Signal") == "Predictor")
    if predictors.height == 0:
        raise MetadataError(f"signal doc {path}: no rows with Cat.Signal == 'Predictor'")

    records: list[dict[str, object]] = []
    for row in predictors.iter_rows(named=True):
        signalname = _parse_text(row["Acronym"])
        if signalname is None:
            raise MetadataError("signal doc has a Predictor row without an Acronym")
        records.append(
            {
                "signalname": signalname,
                "sample_start_year": _parse_year(
                    row["SampleStartYear"], f"{signalname}: SampleStartYear"
                ),
                "sample_end_year": _parse_year(
                    row["SampleEndYear"], f"{signalname}: SampleEndYear"
                ),
                "pub_year": _parse_year(row["Year"], f"{signalname}: Year"),
                "journal": _parse_text(row["Journal"]),
                "authors": _parse_text(row["Authors"]),
                "cat_economic": _parse_text(row["Cat.Economic"]),
                "cat_data": _parse_text(row["Cat.Data"]),
                "clear": row["Predictability in OP"] == "1_clear",
                "date_source": "signaldoc",
            }
        )
    duplicated = sorted(
        {
            str(record["signalname"])
            for record in records
            if sum(1 for other in records if other["signalname"] == record["signalname"]) > 1
        }
    )
    if duplicated:
        raise MetadataError(f"signal doc lists predictors more than once: {duplicated}")
    return records


def _read_supplement(supplement_path: Path) -> list[dict[str, Any]]:
    path = Path(supplement_path)
    if not path.is_file():
        raise MetadataError(f"supplement not found: {path}")
    frame = pl.read_csv(path, infer_schema_length=0)
    if list(frame.columns) != list(SUPPLEMENT_COLUMNS):
        raise MetadataError(
            f"supplement {path}: expected header {list(SUPPLEMENT_COLUMNS)}, got {frame.columns}"
        )
    return frame.to_dicts()


def _apply_supplement(records: list[dict[str, object]], supplement_path: Path) -> None:
    by_signal = {str(record["signalname"]): record for record in records}
    seen: set[tuple[str, str]] = set()
    applied: dict[str, set[str]] = {}
    for row in _read_supplement(supplement_path):
        signalname = _parse_text(row["signalname"])
        field = _parse_text(row["field"])
        if signalname is None or signalname not in by_signal:
            raise MetadataError(f"supplement references unknown predictor {signalname!r}")
        if field is None or field not in SUPPLEMENT_FIELDS:
            raise MetadataError(
                f"supplement references unknown field {field!r} for {signalname!r}; "
                f"allowed: {sorted(SUPPLEMENT_FIELDS)}"
            )
        citation_key = _parse_text(row["citation_key"])
        citation = _parse_text(row["citation"])
        if citation_key is None or citation is None:
            raise MetadataError(
                f"supplement row for {signalname!r}.{field}: citation_key and citation are required"
            )
        key = (signalname, field)
        if key in seen:
            raise MetadataError(f"duplicate supplement row for {signalname!r}.{field}")
        seen.add(key)

        record = by_signal[signalname]
        if record[field] is not None:
            continue  # fill only null fields
        if field in YEAR_FIELDS:
            value = _parse_year(row["value"], f"supplement {signalname}.{field}")
            if value is None:
                raise MetadataError(f"supplement row for {signalname!r}.{field} has no value")
            record[field] = value
        else:
            text = _parse_text(row["value"])
            if text is None:
                raise MetadataError(f"supplement row for {signalname!r}.{field} has no value")
            record[field] = text
        applied.setdefault(signalname, set()).add(citation_key)

    for signalname, keys in applied.items():
        by_signal[signalname]["date_source"] = "supplement:" + "+".join(sorted(keys))


def _validate_records(records: list[dict[str, object]]) -> None:
    for record in records:
        signalname = record["signalname"]
        for field in YEAR_FIELDS:
            if record[field] is None:
                raise MetadataError(f"predictor {signalname!r}: missing {field}")
        start = record["sample_start_year"]
        end = record["sample_end_year"]
        pub = record["pub_year"]
        assert isinstance(start, int) and isinstance(end, int) and isinstance(pub, int)
        if start > end:
            raise MetadataError(
                f"predictor {signalname!r}: sample_start_year {start} > sample_end_year {end}"
            )
        if pub < end:
            raise MetadataError(f"predictor {signalname!r}: pub_year {pub} < sample_end_year {end}")
    same_year = sorted(
        str(record["signalname"])
        for record in records
        if record["pub_year"] == record["sample_end_year"]
    )
    if same_year:
        warnings.warn(
            f"{len(same_year)} predictor(s) published in the same year their sample "
            f"ends: {same_year}",
            MetadataWarning,
            stacklevel=2,
        )


def build_metadata(signal_doc_path: Path, supplement_path: Path | None = None) -> pl.DataFrame:
    """Build the predictor metadata table.

    Inputs: *signal_doc_path* is the canonical OSAP signal-doc CSV
    (``data/raw/<vintage>/signal_doc.csv``); *supplement_path* optionally fills missing
    sample-period or publication fields from ``config/predictor_dates_supplement.csv``,
    one cited source per row.

    Output: one row per ``Cat.Signal == "Predictor"``, sorted by ``signalname``,
    exactly :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`. Years are plain calendar
    years; no returns are involved, so no units apply.
    """
    records = _signal_doc_records(Path(signal_doc_path))
    if supplement_path is not None:
        _apply_supplement(records, Path(supplement_path))
    _validate_records(records)

    frame = (
        pl.DataFrame(records)
        .select(
            pl.col("signalname").cast(pl.Utf8),
            pl.col("sample_start_year").cast(pl.Int16),
            pl.col("sample_end_year").cast(pl.Int16),
            pl.col("pub_year").cast(pl.Int16),
            pl.col("journal").cast(pl.Utf8),
            pl.col("authors").cast(pl.Utf8),
            pl.col("cat_economic").cast(pl.Utf8),
            pl.col("cat_data").cast(pl.Utf8),
            pl.col("clear").cast(pl.Boolean),
            pl.col("date_source").cast(pl.Utf8),
        )
        .sort("signalname")
    )
    validate_frame(frame, PREDICTORS_SCHEMA, key=["signalname"])
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m anomaly_decay.data.metadata --signal-doc PATH [...] --out PATH``."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.data.metadata",
        description="Build the predictor metadata table from the OSAP signal doc.",
    )
    parser.add_argument("--signal-doc", type=Path, required=True, help="signal doc CSV path")
    parser.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="cited supplement CSV for missing fields",
    )
    parser.add_argument("--out", type=Path, required=True, help="output predictors.parquet path")
    args = parser.parse_args(argv)

    try:
        frame = build_metadata(args.signal_doc, args.supplement)
    except (MetadataError, SchemaError) as exc:
        print(f"metadata failed: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    clear = int(frame["clear"].sum())
    published_by_2013 = frame.filter(pl.col("pub_year") <= 2013).height
    print(f"predictors: {frame.height}")
    print(f"clear: {clear}")
    print(f"published by 2013: {published_by_2013}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
