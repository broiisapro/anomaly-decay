"""Static JSON data for the interactive companion page (Phase 3, decision O46).

The companion page is a static HTML page whose Observable JS cells fetch one JSON file
per predictor at run time. This module builds those files from ``results/`` only:

- ``predictors.json`` — the index the page's selector reads: signalname, authors,
  journal, publication year, sample years and the clear/likely flag.
- ``predictors/<signalname>.json`` — one file per predictor: the same bibliographic
  header plus the in-sample / post-sample / post-publication mean return, standard error
  and month count, and the event-year series for the ``sample_end`` and ``publication``
  anchors.

Every float is rounded to :data:`FLOAT_DIGITS` decimals and written by :func:`json.dumps`
with sorted keys, so a rebuild from the same results is byte-identical and the files do
not depend on the platform's float formatting. ``allow_nan=False`` keeps the output valid
JSON: a non-finite value becomes ``null``.

CLI::

    python -m anomaly_decay.site --results-dir results --out report/_site/data
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from anomaly_decay.io import read_results

WINDOWS: tuple[str, ...] = ("in_sample", "post_sample", "post_publication")
"""Window keys in the order the page shows them."""

ANCHORS: tuple[str, ...] = ("sample_end", "publication")
"""Event anchors, matching ``predictor_event_time.csv``."""

FLOAT_DIGITS = 6
"""Decimals kept for every float in the JSON files."""

_SAFE_SIGNALNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]*")
"""Signal names that can be used as a single file name without escaping."""


class SiteError(ValueError):
    """Raised when the committed results cannot form a consistent site build."""


@dataclass(frozen=True)
class SiteBuild:
    """Summary of one :func:`write_site_data` run."""

    out_dir: Path
    n_predictors: int
    n_bytes: int


def round_float(value: object | None) -> float | None:
    """Round *value* to :data:`FLOAT_DIGITS` decimals; null and non-finite become ``None``.

    Negative zero is normalised to ``0.0`` so the JSON text never depends on the sign of
    a value that rounds to zero.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SiteError(f"expected a numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        return None
    rounded = round(number, FLOAT_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _int_value(value: object | None) -> int | None:
    """An integer for JSON, or ``None`` when the results cell is null."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SiteError(f"expected an integer value, got {value!r}")
    return value


def _load(results_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the three results files and check that they describe the same universe."""
    info = read_results("predictor_info.csv", results_dir)
    summary = read_results("predictor_summary.csv", results_dir)
    event_time = read_results("predictor_event_time.csv", results_dir)

    names = info["signalname"].to_list()
    if len(set(names)) != len(names):
        raise SiteError("predictor_info.csv contains duplicate signal names")
    unsafe = sorted(name for name in names if not _SAFE_SIGNALNAME.fullmatch(name))
    if unsafe:
        raise SiteError(f"signal names cannot be used as file names: {unsafe}")
    known = set(names)

    if summary.select("signalname", "window").n_unique() != summary.height:
        raise SiteError("predictor_summary.csv has duplicate (signalname, window) rows")
    if event_time.select("signalname", "anchor", "event_year").n_unique() != event_time.height:
        raise SiteError(
            "predictor_event_time.csv has duplicate (signalname, anchor, event_year) rows"
        )

    for frame, name in (
        (summary, "predictor_summary.csv"),
        (event_time, "predictor_event_time.csv"),
    ):
        unknown = sorted(set(frame["signalname"].to_list()) - known)
        if unknown:
            raise SiteError(f"{name} contains predictors absent from predictor_info.csv: {unknown}")

    unknown_windows = sorted(set(summary["window"].to_list()) - set(WINDOWS))
    if unknown_windows:
        raise SiteError(f"predictor_summary.csv has unknown windows: {unknown_windows}")
    unknown_anchors = sorted(set(event_time["anchor"].to_list()) - set(ANCHORS))
    if unknown_anchors:
        raise SiteError(f"predictor_event_time.csv has unknown anchors: {unknown_anchors}")

    return info, summary, event_time


def _index_entry(row: Mapping[str, object]) -> dict[str, object]:
    """One row of ``predictors.json``, in the fields the selector and header need."""
    return {
        "signalname": row["signalname"],
        "authors": row["authors"],
        "journal": row["journal"],
        "pub_year": _int_value(row["pub_year"]),
        "sample_start_year": _int_value(row["sample_start_year"]),
        "sample_end_year": _int_value(row["sample_end_year"]),
        "clear": bool(row["clear"]),
    }


def _window_payload(
    rows: Mapping[tuple[str, str], Mapping[str, object]], signalname: str
) -> dict[str, dict[str, object]]:
    """The three window cells for one predictor; a missing window is null with 0 months."""
    windows: dict[str, dict[str, object]] = {}
    for window in WINDOWS:
        row = rows.get((signalname, window))
        if row is None:
            windows[window] = {"mean_ret": None, "se_mean": None, "n_months": 0}
        else:
            windows[window] = {
                "mean_ret": round_float(row["mean_ret"]),
                "se_mean": round_float(row["se_mean"]),
                "n_months": _int_value(row["n_months"]),
            }
    return windows


def _event_payload(
    rows: Mapping[str, list[Mapping[str, object]]], signalname: str
) -> dict[str, list[dict[str, object]]]:
    """The event-year series for both anchors; a missing anchor or year simply has no rows."""
    series: dict[str, list[dict[str, object]]] = {anchor: [] for anchor in ANCHORS}
    for row in rows.get(signalname, []):
        series[str(row["anchor"])].append(
            {
                "event_year": _int_value(row["event_year"]),
                "mean_ret": round_float(row["mean_ret"]),
                "n_months": _int_value(row["n_months"]),
            }
        )
    return series


def _write_json(path: Path, payload: object) -> int:
    """Write *payload* as deterministic JSON with LF endings; return the byte count."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path.stat().st_size


def write_site_data(results_dir: Path, out_dir: Path) -> SiteBuild:
    """Write ``predictors.json`` and one ``predictors/<signalname>.json`` per predictor.

    *results_dir* must hold ``predictor_info.csv``, ``predictor_summary.csv`` and
    ``predictor_event_time.csv`` (missing files raise :class:`FileNotFoundError`).
    Inconsistent universes raise :class:`SiteError`. Existing ``predictors/*.json`` files
    in *out_dir* are removed first, so a rebuild never leaves stale predictors behind, and
    the index is written last so it only advertises files that exist.
    """
    info, summary, event_time = _load(results_dir)
    info = info.sort("signalname")

    summary_rows = {
        (str(row["signalname"]), str(row["window"])): row for row in summary.iter_rows(named=True)
    }
    event_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in event_time.iter_rows(named=True):
        event_rows.setdefault(str(row["signalname"]), []).append(row)

    out = Path(out_dir)
    predictors_dir = out / "predictors"
    predictors_dir.mkdir(parents=True, exist_ok=True)
    for stale in predictors_dir.glob("*.json"):
        stale.unlink()

    n_bytes = 0
    index: list[dict[str, object]] = []
    for row in info.iter_rows(named=True):
        signalname = str(row["signalname"])
        index.append(_index_entry(row))
        profile = {
            **_index_entry(row),
            "cat_economic": row["cat_economic"],
            "windows": _window_payload(summary_rows, signalname),
            "event_time": _event_payload(event_rows, signalname),
        }
        n_bytes += _write_json(predictors_dir / f"{signalname}.json", profile)

    n_bytes += _write_json(out / "predictors.json", index)
    return SiteBuild(out_dir=out, n_predictors=info.height, n_bytes=n_bytes)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the companion page's JSON data files from the committed results."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.site",
        description="Write the per-predictor JSON data for the companion page.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="results directory (default: results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report/_site/data"),
        help="output directory (default: report/_site/data)",
    )
    args = parser.parse_args(argv)

    try:
        build = write_site_data(args.results_dir, args.out)
    except FileNotFoundError as exc:
        print(f"site: missing results file: {exc}", file=sys.stderr)
        return 1
    except SiteError as exc:
        print(f"site: {exc}", file=sys.stderr)
        return 1

    print(
        f"site: wrote {build.n_predictors} predictor files and predictors.json "
        f"to {build.out_dir} ({build.n_bytes} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
