"""Load and validate project configuration files.

`config/data.toml` holds data-source settings (pinned OSAP vintage, Ken French URLs and
hashes). Errors are raised as :class:`ConfigError` with the offending key named.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or holds invalid values."""


@dataclass(frozen=True)
class OsapConfig:
    """Pinned OSAP vintage and portfolios to fetch."""

    vintage: int
    portfolios: tuple[str, ...]
    raw_dir: Path


@dataclass(frozen=True)
class FrenchConfig:
    """Ken French source URLs, pinned hashes and output directory."""

    ff5_url: str
    umd_url: str
    pinned_dir: Path
    downloaded_on: str
    ff5_sha256: str
    umd_sha256: str


@dataclass(frozen=True)
class DataConfig:
    """Everything `anomaly_decay.data.fetch` needs, with paths resolved against the repo root."""

    root: Path
    osap: OsapConfig
    french: FrenchConfig


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing key '{key}' in [{where}]")
    return section[key]


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = _require(data, name, "top level")
    if not isinstance(section, dict):
        raise ConfigError(f"[{name}] must be a table")
    return section


def load_data_config(path: Path) -> DataConfig:
    """Load `config/data.toml`; paths are resolved relative to the repository root."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    osap_raw = _section(data, "osap")
    french_raw = _section(data, "french")
    root = path.resolve().parent.parent

    vintage = _require(osap_raw, "vintage", "osap")
    if not isinstance(vintage, int):
        raise ConfigError("[osap] vintage must be an integer release year, e.g. 202510")
    portfolios = _require(osap_raw, "portfolios", "osap")
    if (
        not isinstance(portfolios, list)
        or not portfolios
        or not all(isinstance(item, str) for item in portfolios)
    ):
        raise ConfigError("[osap] portfolios must be a non-empty list of strings")
    raw_dir = _require(osap_raw, "raw_dir", "osap")
    if not isinstance(raw_dir, str):
        raise ConfigError("[osap] raw_dir must be a string")

    urls = {}
    for key in ("ff5_url", "umd_url"):
        value = _require(french_raw, key, "french")
        if not isinstance(value, str) or not value.startswith("https://"):
            raise ConfigError(f"[french] {key} must be an https URL")
        urls[key] = value
    pinned_dir = _require(french_raw, "pinned_dir", "french")
    if not isinstance(pinned_dir, str):
        raise ConfigError("[french] pinned_dir must be a string")
    downloaded_on = _require(french_raw, "downloaded_on", "french")
    if not isinstance(downloaded_on, str) or not _DATE_RE.match(downloaded_on):
        raise ConfigError("[french] downloaded_on must be a YYYY-MM-DD date")
    hashes = {}
    for key in ("ff5_sha256", "umd_sha256"):
        value = _require(french_raw, key, "french")
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise ConfigError(f"[french] {key} must be a lowercase 64-character SHA-256 hex digest")
        hashes[key] = value

    return DataConfig(
        root=root,
        osap=OsapConfig(
            vintage=vintage,
            portfolios=tuple(portfolios),
            raw_dir=root / raw_dir,
        ),
        french=FrenchConfig(
            ff5_url=urls["ff5_url"],
            umd_url=urls["umd_url"],
            pinned_dir=root / pinned_dir,
            downloaded_on=downloaded_on,
            ff5_sha256=hashes["ff5_sha256"],
            umd_sha256=hashes["umd_sha256"],
        ),
    )
