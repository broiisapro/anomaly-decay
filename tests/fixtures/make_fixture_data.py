"""Generate the committed fixture data under tests/fixtures/data/.

Two sources, selected by ``--source`` (default ``auto``):

- ``osap``: subset the real pinned 202510 canonical cache
  (``data/raw/202510/signal_doc.csv`` and ``port_<name>.csv``) to the 15 fixture
  predictors, keeping only ``port == "LS"`` rows. Requires ``make fetch`` to have
  succeeded for the pinned vintage.
- ``synthetic``: deterministic synthetic monthly returns (seeded
  ``numpy.random.default_rng(20261016)``, normal(0.5, 4.0) percent) for the same 15
  predictors, using the real predictor metadata from the Chen-Zimmermann SignalDoc
  (downloaded once from
  https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv and
  committed as ``data/signaldoc_raw.csv`` inside the fixture so reruns are offline).
  Deliberate edge cases: ``MaxRet`` has three null-return months in 1995 and
  ``dVolPut`` returns are shifted by -1.0 percent per month so its in-sample mean is
  negative and its ``ret_norm`` is null by construction.

Outputs (all under ``tests/fixtures``): ``data/osap/202510/signal_doc.csv``,
``data/osap/202510/port_<name>.csv`` for op/deciles_ew/deciles_vw, copies of the pinned
Ken French CSVs under ``data/pinned/``, ``data/FIXTURE_SOURCE.txt`` (first line
``osap-202510`` or ``synthetic``) and ``manifest.lock`` covering everything under
``data/``. Reruns are byte-deterministic given the same source inputs.
"""

from __future__ import annotations

import argparse
import calendar
import shutil
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from anomaly_decay.data.fetch import sha256_file
from anomaly_decay.io import write_csv_canonical

PREDICTORS: tuple[str, ...] = (
    "Mom12m",
    "Size",
    "BM",
    "Accruals",
    "AM",
    "AssetGrowth",
    "GP",
    "IdioVol3F",
    "STreversal",
    "EP",
    "Beta",
    "BetaFP",
    "TrendFactor",
    "dVolPut",
    "MaxRet",
)
PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")
VINTAGE = 202510
VINTAGE_LAST_YYYYMM = 202412
SEED = 20261016
SIGNAL_DOC_COLUMNS: tuple[str, ...] = (
    "Acronym",
    "Cat.Signal",
    "Predictability in OP",
    "Authors",
    "Year",
    "Journal",
    "Cat.Economic",
    "Cat.Data",
    "SampleStartYear",
    "SampleEndYear",
)
PORT_COLUMNS: tuple[str, ...] = (
    "signalname",
    "port",
    "date",
    "ret",
    "signallag",
    "Nlong",
    "Nshort",
)
SIGNAL_DOC_URL = "https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/SignalDoc.csv"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _osap_files_present(repo_root: Path) -> bool:
    raw = repo_root / "data" / "raw" / str(VINTAGE)
    needed = [raw / "signal_doc.csv", *(raw / f"port_{name}.csv" for name in PORTFOLIOS)]
    return all(path.is_file() for path in needed)


def _write_signal_doc(source: pl.DataFrame, out_path: Path) -> None:
    subset = source.filter(pl.col("Acronym").is_in(PREDICTORS)).select(SIGNAL_DOC_COLUMNS)
    missing = sorted(set(PREDICTORS) - set(subset["Acronym"].to_list()))
    if missing:
        raise SystemExit(f"fixture predictors missing from the signal doc: {missing}")
    write_csv_canonical(subset, out_path, sort_by=["Acronym"])


def _months_between(first: int, last: int) -> list[int]:
    months = []
    yyyymm = first
    while yyyymm <= last:
        months.append(yyyymm)
        year, month = divmod(yyyymm, 100)
        if month == 12:
            yyyymm = (year + 1) * 100 + 1
        else:
            yyyymm += 1
    return months


def _last_day(yyyymm: int) -> str:
    year, month = divmod(yyyymm, 100)
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _from_osap(repo_root: Path, fixtures_dir: Path) -> None:
    raw = repo_root / "data" / "raw" / str(VINTAGE)
    signal_doc = pl.read_csv(raw / "signal_doc.csv", infer_schema_length=500)
    _write_signal_doc(signal_doc, fixtures_dir / "data" / "osap" / str(VINTAGE) / "signal_doc.csv")
    for name in PORTFOLIOS:
        frame = pl.read_csv(raw / f"port_{name}.csv")
        subset = frame.filter(
            pl.col("signalname").is_in(PREDICTORS) & (pl.col("port") == "LS")
        ).select(PORT_COLUMNS)
        missing = sorted(set(PREDICTORS) - set(subset["signalname"].to_list()))
        if missing:
            raise SystemExit(f"predictors missing from port_{name}.csv: {missing}")
        write_csv_canonical(
            subset,
            fixtures_dir / "data" / "osap" / str(VINTAGE) / f"port_{name}.csv",
            sort_by=["signalname", "port", "date"],
        )


def _signaldoc_raw(fixtures_dir: Path) -> pl.DataFrame:
    raw_path = fixtures_dir / "data" / "signaldoc_raw.csv"
    if not raw_path.is_file():
        with urllib.request.urlopen(SIGNAL_DOC_URL) as response:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(response.read())
    return pl.read_csv(raw_path, infer_schema_length=500)


def _synthetic_port_frame(
    rng: np.random.Generator, predictor: str, sample_start_year: int
) -> pl.DataFrame:
    months = _months_between((sample_start_year - 1) * 100 + 1, VINTAGE_LAST_YYYYMM)
    returns = rng.normal(0.5, 4.0, len(months))
    if predictor == "dVolPut":
        returns = returns - 1.0  # in-sample mean <= 0 on purpose (ret_norm is null)
    values: list[float | None] = [float(value) for value in returns]
    if predictor == "MaxRet":
        for yyyymm in (199501, 199502, 199503):
            values[months.index(yyyymm)] = None  # dropped rows, counted by the builder
    return pl.DataFrame(
        {
            "signalname": [predictor] * len(months),
            "port": ["LS"] * len(months),
            "date": [_last_day(yyyymm) for yyyymm in months],
            "ret": values,
            "signallag": [1] * len(months),
            "Nlong": [100] * len(months),
            "Nshort": [100] * len(months),
        }
    )


def _from_synthetic(fixtures_dir: Path) -> None:
    signal_doc = _signaldoc_raw(fixtures_dir)
    _write_signal_doc(signal_doc, fixtures_dir / "data" / "osap" / str(VINTAGE) / "signal_doc.csv")
    metadata = signal_doc.filter(pl.col("Acronym").is_in(PREDICTORS)).select(
        ["Acronym", "SampleStartYear"]
    )
    starts = dict(
        zip(metadata["Acronym"].to_list(), metadata["SampleStartYear"].to_list(), strict=True)
    )
    rng = np.random.default_rng(SEED)
    for name in PORTFOLIOS:
        frames = [
            _synthetic_port_frame(rng, predictor, int(starts[predictor]))
            for predictor in PREDICTORS
        ]
        frame = pl.concat(frames).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        write_csv_canonical(
            frame,
            fixtures_dir / "data" / "osap" / str(VINTAGE) / f"port_{name}.csv",
            sort_by=["signalname", "port", "date"],
        )


def _copy_pinned(repo_root: Path, fixtures_dir: Path) -> None:
    target = fixtures_dir / "data" / "pinned"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("ff5_monthly.csv", "umd_monthly.csv"):
        shutil.copyfile(repo_root / "data" / "pinned" / name, target / name)


def _write_fixture_manifest(fixtures_dir: Path, repo_root: Path) -> None:
    data_dir = fixtures_dir / "data"
    entries = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        entries.append((rel, sha256_file(path), path.stat().st_size))
    entries.sort(key=lambda entry: entry[0])
    text = "".join(f"{sha}  {rel}  {size}\n" for rel, sha, size in entries)
    (fixtures_dir / "manifest.lock").write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("auto", "osap", "synthetic"), default="auto")
    parser.add_argument("--fixtures-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    fixtures_dir = args.fixtures_dir.resolve()
    source = args.source
    if source == "auto":
        source = "osap" if _osap_files_present(repo_root) else "synthetic"
    if source == "osap" and not _osap_files_present(repo_root):
        print(
            "osap source requested but data/raw/202510 canonical files are absent", file=sys.stderr
        )
        return 1

    if source == "osap":
        _from_osap(repo_root, fixtures_dir)
    else:
        _from_synthetic(fixtures_dir)
    _copy_pinned(repo_root, fixtures_dir)

    source_path = fixtures_dir / "data" / "FIXTURE_SOURCE.txt"
    if source == "osap":
        detail = f"source: OSAP release {VINTAGE} canonical cache (port == LS rows)"
    else:
        detail = (
            f"seed: {SEED}; distribution: normal(0.5, 4.0) percent per month; "
            f"metadata: {SIGNAL_DOC_URL}"
        )
    source_path.write_text(
        f"{'osap-' + str(VINTAGE) if source == 'osap' else 'synthetic'}\n"
        f"predictors: {len(PREDICTORS)}\n"
        f"{detail}\n",
        encoding="utf-8",
    )
    _write_fixture_manifest(fixtures_dir, repo_root)
    print(f"fixture source: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
