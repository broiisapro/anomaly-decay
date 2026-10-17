"""Download OSAP and Ken French data, cache it canonically and maintain the manifest.

OSAP is fetched through the ``openassetpricing`` package (Google Drive backend) wrapped
with retries/backoff and frame validation: the package itself has no retries and does not
detect Drive's "quota exceeded" HTML page. Canonical copies are written to
``data/raw/<vintage>/``; Ken French canonical copies to ``data/pinned/`` (committed, see
O16). ``data/manifest.lock`` records ``<sha256>  <path relative to repo root>  <bytes>``
lines sorted by path.

CLI::

    python -m anomaly_decay.data.fetch [--osap] [--french] [--refresh-french]
                                       [--verify] [--manifest-scope {all,pinned,fixture}]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
import requests

from anomaly_decay.config import DataConfig, load_data_config
from anomaly_decay.data.french import (
    FF5_RENAME,
    UMD_RENAME,
    FrenchParseError,
    parse_french_zip,
)
from anomaly_decay.io import write_csv_canonical

PORT_COLUMNS: tuple[str, ...] = (
    "signalname",
    "port",
    "date",
    "ret",
    "signallag",
    "Nlong",
    "Nshort",
)
SIGNAL_DOC_REQUIRED: tuple[str, ...] = (
    "Acronym",
    "Cat.Signal",
    "SampleStartYear",
    "SampleEndYear",
    "Year",
)
MIN_PORT_ROWS = 1000
MIN_SIGNAL_DOC_ROWS = 100
MAX_ATTEMPTS = 5
MANIFEST_NAME = "data/manifest.lock"


class FetchError(RuntimeError):
    """Raised when a download fails or its content fails validation."""


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry[T](
    fn: Callable[[], T],
    *,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    description: str = "download",
) -> T:
    """Call *fn*, retrying on any exception with exponential backoff (2**n seconds)."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                sleep(float(2 ** (attempt - 1)))
    raise FetchError(
        f"{description}: failed after {attempts} attempts: {last_error}"
    ) from last_error


def _default_openap(vintage: int) -> Any:
    import openassetpricing

    return openassetpricing.OpenAP(vintage)


def _validate_port(df: pl.DataFrame, name: str) -> pl.DataFrame:
    if df.height < MIN_PORT_ROWS:
        raise FetchError(
            f"portfolio '{name}': only {df.height} rows (expected >= {MIN_PORT_ROWS}); "
            "Google Drive may have returned a quota or error page"
        )
    missing = [column for column in PORT_COLUMNS if column not in df.columns]
    if missing:
        raise FetchError(
            f"portfolio '{name}': missing columns {missing}; got {df.columns}; "
            "Google Drive may have returned a quota or error page"
        )
    if not df.schema["ret"].is_numeric():
        raise FetchError(
            f"portfolio '{name}': column ret is not numeric (dtype {df.schema['ret']})"
        )
    return df


def _validate_signal_doc(df: pl.DataFrame) -> pl.DataFrame:
    if df.height < MIN_SIGNAL_DOC_ROWS:
        raise FetchError(
            f"signal doc: only {df.height} rows (expected >= {MIN_SIGNAL_DOC_ROWS}); "
            "Google Drive may have returned a quota or error page"
        )
    missing = [column for column in SIGNAL_DOC_REQUIRED if column not in df.columns]
    if missing:
        raise FetchError(f"signal doc: missing columns {missing}; got {df.columns}")
    return df


def fetch_osap(
    cfg: DataConfig,
    *,
    openap_factory: Callable[[int], Any] = _default_openap,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Download the OSAP signal doc and configured portfolios; write canonical CSVs.

    Returns the canonical file paths written under ``data/raw/<vintage>/``. Validation and
    retries happen per call; a frame that fails validation (wrong columns, too few rows,
    non-numeric ``ret``) raises :class:`FetchError`.
    """
    api = _retry(
        lambda: openap_factory(cfg.osap.vintage),
        sleep=sleep,
        description=f"open OSAP release {cfg.osap.vintage}",
    )
    osap_dir = cfg.osap.raw_dir / str(cfg.osap.vintage)
    written: list[Path] = []

    def load_signal_doc() -> pl.DataFrame:
        return _validate_signal_doc(api.dl_signal_doc(df_backend="polars"))

    doc = _retry(load_signal_doc, sleep=sleep, description="OSAP signal doc")
    doc_path = osap_dir / "signal_doc.csv"
    write_csv_canonical(doc, doc_path, sort_by=["Acronym"])
    written.append(doc_path)

    for name in cfg.osap.portfolios:

        def load_port(portfolio: str = name) -> pl.DataFrame:
            return _validate_port(api.dl_port(portfolio, df_backend="polars"), portfolio)

        frame = _retry(load_port, sleep=sleep, description=f"OSAP portfolio {name}")
        path = osap_dir / f"port_{name}.csv"
        write_csv_canonical(
            frame.select(PORT_COLUMNS), path, sort_by=["signalname", "port", "date"]
        )
        written.append(path)
    return written


def _download_bytes(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"PK"):
        content_type = response.headers.get("content-type", "unknown")
        raise FetchError(
            f"{url}: response is not a zip archive (content-type {content_type!r}); "
            "the host may be serving an error page"
        )
    return content


def update_pinned_metadata(
    config_path: Path, *, ff5_sha256: str, umd_sha256: str, downloaded_on: str
) -> None:
    """Rewrite the pinned Ken French hashes and download date in ``config/data.toml``.

    Only ``make refresh-french`` reaches this; the surrounding file text is preserved.
    """
    replacements = {
        "ff5_sha256": ff5_sha256,
        "umd_sha256": umd_sha256,
        "downloaded_on": downloaded_on,
    }
    text = Path(config_path).read_text(encoding="utf-8")
    for key, value in replacements.items():
        pattern = re.compile(rf'^({re.escape(key)}\s*=\s*")[^"]*(")', re.MULTILINE)
        if not pattern.search(text):
            raise FetchError(f"key '{key}' not found in {config_path}")
        text = pattern.sub(rf"\g<1>{value}\g<2>", text, count=1)
    Path(config_path).write_text(text, encoding="utf-8")


def fetch_french(
    cfg: DataConfig,
    *,
    refresh: bool = False,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Download both Ken French zips, verify (or refresh) their SHA-256, write canonical CSVs.

    With ``refresh=False`` a hash mismatch against ``config/data.toml`` raises
    :class:`FetchError` (the library revises its files monthly). With ``refresh=True`` the
    downloaded bytes win and ``config/data.toml`` is updated. Raw zips are cached under
    ``data/raw/ff/<downloaded_on>/``; canonical monthly CSVs are written to ``data/pinned/``.
    """
    http = session if session is not None else requests.Session()
    downloaded_on = time.strftime("%Y-%m-%d") if refresh else cfg.french.downloaded_on
    plans = (
        ("ff5", cfg.french.ff5_url, cfg.french.ff5_sha256, FF5_RENAME),
        ("umd", cfg.french.umd_url, cfg.french.umd_sha256, UMD_RENAME),
    )
    hashes: dict[str, str] = {}
    written: list[Path] = []
    for name, url, expected, rename in plans:
        content = _retry(
            partial(_download_bytes, http, url),
            sleep=sleep,
            description=f"Ken French {name} zip",
        )
        actual = sha256_bytes(content)
        hashes[name] = actual
        if not refresh and actual != expected:
            raise FetchError(
                f"Ken French {name} zip SHA-256 mismatch: config pins {expected}, "
                f"download is {actual}. The library revises files monthly; "
                "run 'make refresh-french' to update the pinned copy."
            )
        try:
            frame = parse_french_zip(content, rename)
        except FrenchParseError as exc:
            raise FetchError(f"Ken French {name} zip: {exc}") from exc
        canonical = cfg.french.pinned_dir / f"{name}_monthly.csv"
        write_csv_canonical(frame, canonical, sort_by=["yyyymm"])
        zip_path = cfg.osap.raw_dir / "ff" / downloaded_on / f"{name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(content)
        written.append(canonical)

    if refresh:
        update_pinned_metadata(
            cfg.root / "config" / "data.toml",
            ff5_sha256=hashes["ff5"],
            umd_sha256=hashes["umd"],
            downloaded_on=downloaded_on,
        )
    return written


@dataclass(frozen=True)
class ManifestEntry:
    """One line of a manifest: hash, repo-root-relative path, size in bytes."""

    sha256: str
    path: str
    size: int


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying a manifest against the working tree."""

    checked: int
    mismatches: tuple[str, ...]
    missing: tuple[str, ...]
    not_cached: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when no present file mismatched and no committed file is missing."""
        return not self.mismatches and not self.missing


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read a manifest file into entries; malformed lines raise :class:`FetchError`."""
    path = Path(path)
    if not path.is_file():
        raise FetchError(f"manifest not found: {path}")
    entries: list[ManifestEntry] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("  ")
        if len(parts) != 3:
            raise FetchError(f"{path}:{number}: malformed manifest line: {line!r}")
        sha256, rel_path, size_text = parts
        try:
            size = int(size_text)
        except ValueError as exc:
            raise FetchError(f"{path}:{number}: bad byte count: {size_text!r}") from exc
        entries.append(ManifestEntry(sha256=sha256, path=rel_path, size=size))
    return entries


def manifest_path_for(cfg: DataConfig, scope: str) -> Path:
    """Manifest file for a scope: ``all``/``pinned`` use ``data/manifest.lock``."""
    if scope in ("all", "pinned"):
        return cfg.root / MANIFEST_NAME
    if scope == "fixture":
        return cfg.root / "tests" / "fixtures" / "manifest.lock"
    raise ValueError(f"unknown manifest scope: {scope!r}")


def _collect_files(cfg: DataConfig, scope: str) -> list[Path]:
    if scope == "pinned":
        roots = [cfg.french.pinned_dir]
    elif scope == "all":
        roots = [cfg.french.pinned_dir, cfg.osap.raw_dir]
    elif scope == "fixture":
        roots = [cfg.root / "tests" / "fixtures"]
    else:
        raise ValueError(f"unknown manifest scope: {scope!r}")
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name in ("manifest.lock", ".DS_Store") or "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def write_manifest(cfg: DataConfig, scope: str = "all") -> Path:
    """Hash every file in *scope* and rewrite its manifest, sorted by path.

    Entries already in the manifest whose file is absent from disk are preserved (the
    OSAP cache is gitignored while its manifest is committed, so a failed or absent cache
    must not drop entries).
    """
    manifest_path = manifest_path_for(cfg, scope)
    root = cfg.root
    entries: dict[str, ManifestEntry] = {}
    for path in _collect_files(cfg, scope):
        rel = path.relative_to(root).as_posix()
        entries[rel] = ManifestEntry(sha256=sha256_file(path), path=rel, size=path.stat().st_size)
    if manifest_path.is_file():
        for entry in read_manifest(manifest_path):
            if entry.path not in entries and not (root / entry.path).exists():
                entries[entry.path] = entry
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{entry.sha256}  {entry.path}  {entry.size}\n"
        for entry in sorted(entries.values(), key=lambda item: item.path)
    ]
    manifest_path.write_text("".join(lines), encoding="utf-8")
    return manifest_path


def verify_manifest(manifest_path: Path, root: Path) -> VerifyResult:
    """Check every manifest entry against the working tree.

    Files present must match hash and size. Absent files under ``data/raw/`` are reported
    as ``not_cached`` (gitignored cache; run ``make fetch``); any other absent file is a
    ``missing`` failure.
    """
    mismatches: list[str] = []
    missing: list[str] = []
    not_cached: list[str] = []
    entries = read_manifest(manifest_path)
    for entry in entries:
        path = Path(root) / entry.path
        if not path.is_file():
            if entry.path.startswith("data/raw/"):
                not_cached.append(entry.path)
            else:
                missing.append(entry.path)
            continue
        if path.stat().st_size != entry.size or sha256_file(path) != entry.sha256:
            mismatches.append(entry.path)
    return VerifyResult(
        checked=len(entries),
        mismatches=tuple(mismatches),
        missing=tuple(missing),
        not_cached=tuple(not_cached),
    )


def verify_scope(cfg: DataConfig, scope: str) -> VerifyResult:
    """Verify the manifest for *scope*; see :func:`verify_manifest`."""
    return verify_manifest(manifest_path_for(cfg, scope), cfg.root)


def _print_verify(result: VerifyResult) -> None:
    for path in result.not_cached:
        print(f"not cached (run make fetch): {path}")
    for path in result.missing:
        print(f"missing: {path}")
    for path in result.mismatches:
        print(f"mismatch: {path}")
    if result.ok:
        print("all hashes match")
    else:
        print(
            f"hash verification failed: {len(result.mismatches)} mismatched, "
            f"{len(result.missing)} missing"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; see module docstring for usage."""
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.data.fetch",
        description="Fetch, cache and verify OSAP and Ken French data.",
    )
    parser.add_argument("--osap", action="store_true", help="fetch the OSAP files only")
    parser.add_argument("--french", action="store_true", help="fetch the Ken French files only")
    parser.add_argument(
        "--refresh-french",
        action="store_true",
        help="re-download Ken French files and update the pinned hashes in config/data.toml",
    )
    parser.add_argument("--verify", action="store_true", help="verify manifest hashes")
    parser.add_argument("--manifest-scope", choices=("all", "pinned", "fixture"), default="all")
    parser.add_argument("--config", type=Path, default=Path("config/data.toml"))
    args = parser.parse_args(argv)

    try:
        cfg = load_data_config(args.config)
    except Exception as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1

    if args.verify:
        try:
            result = verify_scope(cfg, args.manifest_scope)
        except FetchError as exc:
            print(f"verification failed: {exc}", file=sys.stderr)
            return 1
        _print_verify(result)
        return 0 if result.ok else 1

    selectors = args.osap or args.french or args.refresh_french
    do_osap = args.osap or not selectors
    do_french = args.french or args.refresh_french
    if not selectors:
        pinned = [cfg.french.pinned_dir / f"{name}_monthly.csv" for name in ("ff5", "umd")]
        do_french = not all(path.is_file() for path in pinned)

    try:
        if do_osap:
            paths = fetch_osap(cfg)
            for path in paths:
                print(f"wrote {path.relative_to(cfg.root)}")
        if do_french:
            paths = fetch_french(cfg, refresh=args.refresh_french)
            for path in paths:
                print(f"wrote {path.relative_to(cfg.root)}")
        manifest_path = write_manifest(cfg, "all")
        print(f"wrote {manifest_path.relative_to(cfg.root)}")
    except (FetchError, FrenchParseError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
