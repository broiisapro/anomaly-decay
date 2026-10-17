"""Generate the committed fixture data under tests/fixtures/data/.

The fixture is a subset of the pinned OSAP 202510 canonical cache: the 15
pre-registered predictors below, ``port == "LS"`` rows only, all months kept exactly
as the cache has them (the production panel drops pre-sample months, the fixture does
not). Predictors without long-short rows in a given portfolio file are simply absent
from that file -- never padded. Requires the cache populated by ``make fetch`` for the
pinned vintage.

Outputs (all under ``tests/fixtures``): ``data/osap/202510/signal_doc.csv``,
``data/osap/202510/port_<name>.csv`` for op/deciles_ew/deciles_vw, copies of the pinned
Ken French CSVs under ``data/pinned/``, ``data/FIXTURE_SOURCE.txt`` (first line
``osap-202510``; the source files and their SHA-256 hashes) and ``manifest.lock``
covering everything under ``data/``. Reruns are byte-deterministic given the same
source inputs.

Edge cases that the retired synthetic fixture injected on purpose (null returns,
a predictor-portfolio with a non-positive in-sample mean) do not occur in the real
202510 cache; unit tests cover them with hand-built frames instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

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
# The R3 pitfall: canonical CSVs store `port` as text ("01", ..., "LS") and the date as
# ISO text; leave both as text on the way in.
CSV_OVERRIDES: dict[str, pl.DataTypeClass] = {"port": pl.Utf8, "date": pl.Utf8}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_files(repo_root: Path) -> list[Path]:
    raw = repo_root / "data" / "raw" / str(VINTAGE)
    files = [raw / "signal_doc.csv", *(raw / f"port_{name}.csv" for name in PORTFOLIOS)]
    pinned = repo_root / "data" / "pinned"
    files += [pinned / name for name in ("ff5_monthly.csv", "umd_monthly.csv")]
    return files


def _missing_sources(repo_root: Path) -> list[Path]:
    return [path for path in _source_files(repo_root) if not path.is_file()]


def _write_signal_doc(source: pl.DataFrame, out_path: Path) -> None:
    subset = source.filter(pl.col("Acronym").is_in(PREDICTORS)).select(SIGNAL_DOC_COLUMNS)
    missing = sorted(set(PREDICTORS) - set(subset["Acronym"].to_list()))
    if missing:
        raise SystemExit(f"fixture predictors missing from the signal doc: {missing}")
    write_csv_canonical(subset, out_path, sort_by=["Acronym"])


def _from_osap(repo_root: Path, fixtures_dir: Path) -> None:
    raw = repo_root / "data" / "raw" / str(VINTAGE)
    signal_doc = pl.read_csv(raw / "signal_doc.csv", infer_schema_length=500)
    _write_signal_doc(signal_doc, fixtures_dir / "data" / "osap" / str(VINTAGE) / "signal_doc.csv")
    for name in PORTFOLIOS:
        frame = pl.read_csv(raw / f"port_{name}.csv", schema_overrides=CSV_OVERRIDES)
        subset = frame.filter(
            pl.col("signalname").is_in(PREDICTORS) & (pl.col("port") == "LS")
        ).select(PORT_COLUMNS)
        present = set(subset["signalname"].to_list())
        missing = sorted(set(PREDICTORS) - present)
        if missing and name == "op":
            raise SystemExit(f"predictors missing from port_{name}.csv: {missing}")
        if missing:
            # Real coverage: only 179 of 212 predictors have deciles long-short rows;
            # a predictor without them is absent from this file by construction.
            print(f"port_{name}.csv: no LS rows for {missing} (real coverage, not padded)")
        write_csv_canonical(
            subset,
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


def _write_fixture_source(fixtures_dir: Path, repo_root: Path) -> None:
    lines = [
        f"osap-{VINTAGE}",
        f"predictors: {len(PREDICTORS)}",
        f"source: OSAP release {VINTAGE} canonical cache (port == LS rows), all months",
    ]
    for path in _source_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        lines.append(f"source-sha256: {rel} {sha256_file(path)}")
    (fixtures_dir / "data" / "FIXTURE_SOURCE.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("osap",), default="osap")
    parser.add_argument("--fixtures-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    fixtures_dir = args.fixtures_dir.resolve()
    missing = _missing_sources(repo_root)
    if missing:
        print(
            "the OSAP canonical cache is incomplete; run make fetch first. Missing: "
            + ", ".join(path.relative_to(repo_root).as_posix() for path in missing),
            file=sys.stderr,
        )
        return 1

    _from_osap(repo_root, fixtures_dir)
    _copy_pinned(repo_root, fixtures_dir)
    _write_fixture_source(fixtures_dir, repo_root)
    _write_fixture_manifest(fixtures_dir, repo_root)
    print(f"fixture source: osap-{VINTAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
