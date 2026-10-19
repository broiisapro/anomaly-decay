"""Download OSAP and Ken French data, cache it canonically and maintain the manifest.

OSAP is fetched through the ``openassetpricing`` package (Google Drive backend) wrapped
with retries/backoff and frame validation: the package itself has no retries and does not
detect Drive's "quota exceeded" HTML page. Canonical copies are written to
``data/raw/<vintage>/``; Ken French canonical copies to ``data/pinned/`` (committed, see
O16). ``data/manifest.lock`` records ``<sha256>  <path relative to repo root>  <bytes>``
lines sorted by path.

The cache is verified before it is trusted: a file whose size and hash match its manifest
entry is never re-downloaded (``cached: <path>``), and a fresh download that disagrees
with an existing entry is kept as ``<path>.rejected``, fails the run and leaves the
manifest untouched. Entries are added when a file has none (first-time cache population);
changing an existing entry requires ``--update-manifest``.

Most calls fetch the primary ``[osap] vintage`` and its configured portfolios. Phase 2's
vintage comparison additionally fetches an earlier release into its own
``data/raw/<vintage>/`` directory from the same validated pipeline; those calls pass an
explicit ``--vintage`` (the CLI fetches the signal doc and the ``op`` portfolios only —
the comparison needs nothing else). Legacy signal-doc headers are renamed by the explicit
:data:`SIGNAL_DOC_RENAMES` table; a release whose header is not covered by it fails
validation loudly with the observed columns.

CLI::

    python -m anomaly_decay.data.fetch [--osap] [--french] [--refresh-french]
                                       [--vintage N] [--portfolios a,b] [--verify]
                                       [--update-manifest]
                                       [--manifest-scope {all,pinned,fixture}]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import tomllib
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
# Explicit legacy SignalDoc column renames, applied for every release before validation.
# The 2023.08, 2024.08 and 2024.10 headers were checked against the OSAP CrossSection
# repository history and match the canonical header, so no entry is needed for the
# vintages in `compare_vintages`; a release that renames a required column without an
# entry here fails loudly in `_validate_signal_doc` with the columns it did find.
SIGNAL_DOC_RENAMES: dict[str, str] = {}
# A non-primary vintage is fetched for the vintage comparison, which needs the signal
# doc and the `op` long-short rows only; the decile files are large and unused there.
NON_PRIMARY_PORTFOLIOS: tuple[str, ...] = ("op",)
MIN_PORT_ROWS = 1000
MIN_SIGNAL_DOC_ROWS = 100
MAX_ATTEMPTS = 5
MANIFEST_NAME = "data/manifest.lock"
# Working files that must never enter a manifest scan.
IGNORED_SUFFIXES: tuple[str, ...] = (".rejected", ".tmp")


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


def _apply_signal_doc_renames(df: pl.DataFrame) -> pl.DataFrame:
    """Rename legacy signal-doc columns per :data:`SIGNAL_DOC_RENAMES`.

    Only names in the table are touched, and only when the canonical name is absent, so
    a release that already uses canonical names is passed through unchanged.
    """
    renames = {
        old: new
        for old, new in SIGNAL_DOC_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(renames) if renames else df


def _validate_signal_doc(df: pl.DataFrame) -> pl.DataFrame:
    if df.height < MIN_SIGNAL_DOC_ROWS:
        raise FetchError(
            f"signal doc: only {df.height} rows (expected >= {MIN_SIGNAL_DOC_ROWS}); "
            "Google Drive may have returned a quota or error page"
        )
    missing = [column for column in SIGNAL_DOC_REQUIRED if column not in df.columns]
    if missing:
        raise FetchError(
            f"signal doc: missing columns {missing} after applying "
            f"SIGNAL_DOC_RENAMES; got {df.columns}"
        )
    return df


def load_compare_vintages(config_path: Path) -> tuple[int, ...]:
    """Read ``[osap] compare_vintages`` from ``config/data.toml``.

    The list is the pre-registered preference order of earlier OSAP releases for the
    vintage comparison (Phase 2). Unknown or absent keys yield an empty tuple so the
    loader can be used against minimal test configurations.
    """
    config_path = Path(config_path)
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)
    osap = data.get("osap", {})
    raw = osap.get("compare_vintages", [])
    if not isinstance(raw, list) or not all(isinstance(v, int) for v in raw):
        raise FetchError(f"{config_path}: [osap] compare_vintages must be a list of integers")
    return tuple(raw)


@dataclass(frozen=True)
class ManifestEntry:
    """One line of a manifest: hash, repo-root-relative path, size in bytes."""

    sha256: str
    path: str
    size: int


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


def _manifest_text(entries: Sequence[ManifestEntry]) -> str:
    return "".join(
        f"{entry.sha256}  {entry.path}  {entry.size}\n"
        for entry in sorted(entries, key=lambda item: item.path)
    )


class ManifestUpdater:
    """Stage first-time additions and explicit updates to ``data/manifest.lock``.

    Entries that already exist are never changed unless *update* is true (the
    ``--update-manifest`` flag): a downloaded file that disagrees with its entry is kept
    as ``<path>.rejected`` and the run fails. Files that had no entry are added. Nothing
    is written until :meth:`save` is called.
    """

    def __init__(self, manifest_path: Path, root: Path, *, update: bool = False) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root)
        self.update = update
        self.added: list[str] = []
        self.updated: list[str] = []
        self.entries: dict[str, ManifestEntry] = {}
        if self.manifest_path.is_file():
            for entry in read_manifest(self.manifest_path):
                self.entries[entry.path] = entry

    def entry(self, rel_path: str) -> ManifestEntry | None:
        """Current entry for *rel_path*, or None when the file has no entry."""
        return self.entries.get(rel_path)

    def is_cached(self, rel_path: str) -> bool:
        """True when the file exists and its size and hash match its manifest entry."""
        entry = self.entries.get(rel_path)
        if entry is None:
            return False
        path = self.root / rel_path
        return (
            path.is_file()
            and path.stat().st_size == entry.size
            and sha256_file(path) == entry.sha256
        )

    def adopt(self, rel_path: str) -> ManifestEntry:
        """Record an existing file that has no manifest entry (first-time cache)."""
        path = self.root / rel_path
        entry = ManifestEntry(sha256_file(path), rel_path, path.stat().st_size)
        self.entries[rel_path] = entry
        self.added.append(rel_path)
        return entry

    def accept(self, rel_path: str, tmp_path: Path, dest: Path, *, force: bool = False) -> str:
        """Validate a freshly written temporary file and move it into place.

        Returns ``added`` (new to the manifest), ``updated`` (entry rewritten) or
        ``unchanged``. Without ``update``/``force``, a file that disagrees with an
        existing entry is kept as ``<dest>.rejected`` and :class:`FetchError` is raised.
        """
        sha = sha256_file(tmp_path)
        size = tmp_path.stat().st_size
        entry = self.entries.get(rel_path)
        if (
            entry is not None
            and not (self.update or force)
            and (entry.sha256 != sha or entry.size != size)
        ):
            rejected = dest.with_name(dest.name + ".rejected")
            os.replace(tmp_path, rejected)
            raise FetchError(
                f"{rel_path}: SHA-256 mismatch against the manifest: expected "
                f"{entry.sha256}, got {sha}; the download was kept as "
                f"{rejected.relative_to(self.root).as_posix()} and the manifest was "
                "not changed (re-run with --update-manifest to accept a revised file)"
            )
        os.replace(tmp_path, dest)
        if entry is None:
            self.entries[rel_path] = ManifestEntry(sha, rel_path, size)
            self.added.append(rel_path)
            return "added"
        if entry.sha256 != sha or entry.size != size:
            self.entries[rel_path] = ManifestEntry(sha, rel_path, size)
            self.updated.append(rel_path)
            return "updated"
        return "unchanged"

    def save(self) -> Path | None:
        """Write the manifest when anything was added or updated; else return None."""
        if not self.added and not self.updated:
            return None
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(_manifest_text(list(self.entries.values())), encoding="utf-8")
        return self.manifest_path


def fetch_osap(
    cfg: DataConfig,
    *,
    vintage: int | None = None,
    portfolios: Sequence[str] | None = None,
    openap_factory: Callable[[int], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    updater: ManifestUpdater | None = None,
) -> list[Path]:
    """Fetch one OSAP release's signal doc and portfolios into the canonical cache.

    *vintage* defaults to the pinned ``[osap] vintage``; passing an earlier release
    fetches it into its own ``data/raw/<vintage>/`` directory through the same
    validation and canonical write. *portfolios* defaults to the configured list for
    the pinned vintage and to :data:`NON_PRIMARY_PORTFOLIOS` (``op`` only) for any
    other release, which is all the vintage comparison needs.

    Returns the paths written by this call. With *updater*, a file whose cached copy
    matches its manifest entry is skipped and reported as ``cached: <path>``; an existing
    file without an entry is hashed and recorded; every other file is downloaded,
    validated and written atomically. A frame that fails validation (wrong columns, too
    few rows, non-numeric ``ret``) raises :class:`FetchError`.
    """
    release = cfg.osap.vintage if vintage is None else vintage
    if portfolios is None:
        names = cfg.osap.portfolios if release == cfg.osap.vintage else NON_PRIMARY_PORTFOLIOS
    else:
        names = tuple(portfolios)
    names = tuple(dict.fromkeys(names))
    if not names:
        raise FetchError("fetch_osap: at least one portfolio is required")
    osap_dir = cfg.osap.raw_dir / str(release)
    targets: list[tuple[str, Path]] = [("signal_doc", osap_dir / "signal_doc.csv")]
    targets += [(f"port_{name}", osap_dir / f"port_{name}.csv") for name in names]

    todo: list[tuple[str, Path]] = []
    for label, dest in targets:
        if updater is None:
            todo.append((label, dest))
            continue
        rel = dest.relative_to(cfg.root).as_posix()
        if updater.is_cached(rel):
            print(f"cached: {rel}")
            continue
        if dest.is_file() and updater.entry(rel) is None:
            updater.adopt(rel)
            print(f"adopted: {rel} (existing file had no manifest entry)")
            continue
        todo.append((label, dest))

    api: Any = None
    if todo:
        factory = _default_openap if openap_factory is None else openap_factory
        api = _retry(
            lambda: factory(release),
            sleep=sleep,
            description=f"open OSAP release {release}",
        )

    written: list[Path] = []
    for label, dest in todo:
        if label == "signal_doc":
            frame = _retry(
                lambda: _validate_signal_doc(
                    _apply_signal_doc_renames(api.dl_signal_doc(df_backend="polars"))
                ),
                sleep=sleep,
                description=f"OSAP {release} signal doc",
            )
            sort_by = ["Acronym"]
        else:
            name = label.removeprefix("port_")

            def load_port(portfolio: str = name) -> pl.DataFrame:
                return _validate_port(api.dl_port(portfolio, df_backend="polars"), portfolio)

            frame = _retry(load_port, sleep=sleep, description=f"OSAP {release} portfolio {name}")
            frame = frame.select(PORT_COLUMNS)
            sort_by = ["signalname", "port", "date"]
        rel = dest.relative_to(cfg.root).as_posix()
        tmp = dest.with_name(dest.name + ".tmp")
        write_csv_canonical(frame, tmp, sort_by=sort_by)
        if updater is not None:
            updater.accept(rel, tmp, dest)
        else:
            os.replace(tmp, dest)
        print(f"wrote {rel}")
        written.append(dest)
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
    updater: ManifestUpdater | None = None,
) -> list[Path]:
    """Download both Ken French zips, verify (or refresh) their SHA-256, write canonical CSVs.

    With *updater* and ``refresh=False``, a plan whose canonical CSV matches its manifest
    entry is skipped and reported as ``cached: <path>``. With ``refresh=False`` a hash
    mismatch against ``config/data.toml`` raises :class:`FetchError` (the library revises
    its files monthly). With ``refresh=True`` the downloaded bytes win, the manifest
    entries for the two zips and canonical CSVs are updated even without
    ``--update-manifest`` (this is the explicit refresh path), and ``config/data.toml``
    is updated. Raw zips are cached under ``data/raw/ff/<downloaded_on>/``; canonical
    monthly CSVs are written to ``data/pinned/``.
    """
    downloaded_on = time.strftime("%Y-%m-%d") if refresh else cfg.french.downloaded_on
    plans = (
        ("ff5", cfg.french.ff5_url, cfg.french.ff5_sha256, FF5_RENAME),
        ("umd", cfg.french.umd_url, cfg.french.umd_sha256, UMD_RENAME),
    )
    canonical = {name: cfg.french.pinned_dir / f"{name}_monthly.csv" for name, *_ in plans}

    pending: list[tuple[str, str, str, dict[str, str]]] = []
    for name, url, expected, rename in plans:
        rel = canonical[name].relative_to(cfg.root).as_posix()
        if updater is not None and not refresh and updater.is_cached(rel):
            print(f"cached: {rel}")
            continue
        pending.append((name, url, expected, rename))

    if not pending:
        return []
    http = session if session is not None else requests.Session()
    hashes: dict[str, str] = {}
    written: list[Path] = []
    for name, url, expected, rename in pending:
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
        dest = canonical[name]
        tmp = dest.with_name(dest.name + ".tmp")
        write_csv_canonical(frame, tmp, sort_by=["yyyymm"])
        zip_path = cfg.osap.raw_dir / "ff" / downloaded_on / f"{name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_tmp = zip_path.with_name(zip_path.name + ".tmp")
        zip_tmp.write_bytes(content)
        if updater is not None:
            updater.accept(dest.relative_to(cfg.root).as_posix(), tmp, dest, force=refresh)
            updater.accept(
                zip_path.relative_to(cfg.root).as_posix(), zip_tmp, zip_path, force=refresh
            )
        else:
            os.replace(tmp, dest)
            os.replace(zip_tmp, zip_path)
        written.append(dest)

    if refresh:
        update_pinned_metadata(
            cfg.root / "config" / "data.toml",
            ff5_sha256=hashes["ff5"],
            umd_sha256=hashes["umd"],
            downloaded_on=downloaded_on,
        )
    return written


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
            if path.name.endswith(IGNORED_SUFFIXES):
                continue
            if path.name in ("manifest.lock", ".DS_Store") or "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def write_manifest(cfg: DataConfig, scope: str = "all") -> Path:
    """Rebuild a manifest from every file in *scope* on disk, sorted by path.

    A full rebuild for tests and fixture regeneration; the fetch CLI never calls it
    (it uses :class:`ManifestUpdater` so existing entries are not rewritten). Entries
    already in the manifest whose file is absent from disk are preserved (the OSAP
    cache is gitignored while its manifest is committed, so a failed or absent cache
    must not drop entries). ``.rejected``/``.tmp`` working files are ignored.
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
    manifest_path.write_text(_manifest_text(list(entries.values())), encoding="utf-8")
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
        "--vintage",
        type=int,
        default=None,
        help=(
            "OSAP release to fetch (default: the pinned vintage); a non-primary "
            "release fetches the signal doc and op portfolios only"
        ),
    )
    parser.add_argument(
        "--portfolios",
        default=None,
        help="comma-separated portfolio names (default: config portfolios, or op)",
    )
    parser.add_argument(
        "--refresh-french",
        action="store_true",
        help="re-download Ken French files and update the pinned hashes in config/data.toml",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="rewrite manifest entries for re-downloaded files that changed",
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
    do_french = args.french or args.refresh_french or not selectors

    portfolios: tuple[str, ...] | None = None
    if args.portfolios is not None:
        portfolios = tuple(name for name in args.portfolios.split(",") if name)
        if not portfolios:
            print("fetch failed: --portfolios is empty", file=sys.stderr)
            return 1

    try:
        updater = ManifestUpdater(
            manifest_path_for(cfg, "all"), cfg.root, update=args.update_manifest
        )
        if do_osap:
            fetch_osap(cfg, vintage=args.vintage, portfolios=portfolios, updater=updater)
        if do_french:
            fetch_french(cfg, refresh=args.refresh_french, updater=updater)
        manifest_path = updater.save()
    except (FetchError, FrenchParseError) as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    if manifest_path is None:
        print("manifest unchanged")
    else:
        print(f"wrote {manifest_path.relative_to(cfg.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
