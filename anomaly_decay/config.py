"""Load and validate project configuration files.

`config/data.toml` holds data-source settings (pinned OSAP vintage, Ken French URLs and
hashes); `config/specifications.yaml` holds the pre-registered specifications. Errors are
raised as :class:`ConfigError` with the offending key named.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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


@dataclass(frozen=True)
class BootstrapPrereg:
    """The pre-registered moving-block bootstrap of the main decay estimates."""

    id: str
    base_spec: str
    method: str
    block_length_months: int
    reps: int
    seed: int
    statistics: tuple[str, ...]
    interval: str
    level: float
    exploratory: bool


@dataclass(frozen=True)
class MultipleTestingSpecPrereg:
    """One pre-registered multiple-testing specification (a base spec plus a weak flag)."""

    id: str
    base_spec: str
    tstat_source: str
    extra_terms: tuple[str, ...]
    exploratory: bool


@dataclass(frozen=True)
class MultipleTestingPrereg:
    """The pre-registered multiple-testing lens on the main decay estimates."""

    hurdle_t: float
    weak_rule: str
    missing_tstat: str
    tstat_sources: Mapping[str, str]
    specs: tuple[MultipleTestingSpecPrereg, ...]


@dataclass(frozen=True)
class VintagesPrereg:
    """The pre-registered OSAP vintage comparison."""

    primary: int
    compare_preference: tuple[int, ...]
    predictor_set: str
    specs: tuple[str, ...]
    metrics: tuple[str, ...]
    exploratory: bool


@dataclass(frozen=True)
class Phase2Prereg:
    """The three pre-registered Phase 2 analyses, as loaded from the config file."""

    version: int
    bootstrap: BootstrapPrereg
    multiple_testing: MultipleTestingPrereg
    vintages: VintagesPrereg


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


SPEC_GROUPS: tuple[str, ...] = ("main", "robustness", "extension")
SPEC_PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")
SPEC_DEP_VARS: tuple[str, ...] = ("ret", "ret_norm", "ret_factor_adj")
SPEC_CONVENTIONS: tuple[str, ...] = ("default", "alt")
SPEC_COVS: tuple[str, ...] = ("clustered_time", "clustered_twoway", "driscoll_kraay")
SPEC_TOP_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "base",
        "label",
        "group",
        "portfolio",
        "dep_var",
        "convention",
        "cov",
        "cov_params",
        "universe",
        "months",
        "extra_terms",
        "exploratory",
    }
)
SPEC_UNIVERSE_KEYS: frozenset[str] = frozenset(
    {"pub_year_max", "sample_start_year_min", "clear_only"}
)
SPEC_MONTHS_KEYS: frozenset[str] = frozenset({"start_yyyymm", "end_yyyymm"})


@dataclass(frozen=True)
class Spec:
    """One resolved, validated entry of `config/specifications.yaml`."""

    id: str
    label: str
    group: str
    portfolio: str
    dep_var: str
    convention: str
    cov: str
    cov_params: Mapping[str, object]
    pub_year_max: int | None
    sample_start_year_min: int | None
    clear_only: bool
    start_yyyymm: int | None
    end_yyyymm: int | None
    extra_terms: tuple[str, ...]
    exploratory: bool


def _merge_spec_mappings(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_spec_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_spec_entry(
    entries: Mapping[str, Mapping[str, Any]], spec_id: str, chain: tuple[str, ...]
) -> dict[str, Any]:
    if spec_id in chain:
        raise ConfigError(f"circular 'base' chain: {' -> '.join((*chain, spec_id))}")
    entry = entries[spec_id]
    resolved: dict[str, Any] = {}
    base_id = entry.get("base")
    if base_id is not None:
        if not isinstance(base_id, str) or base_id not in entries:
            raise ConfigError(f"specification '{spec_id}': unknown base {base_id!r}")
        resolved = _resolve_spec_entry(entries, base_id, (*chain, spec_id))
    return _merge_spec_mappings(resolved, entry)


def _require_choice(spec_id: str, key: str, value: Any, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"specification '{spec_id}': {key} must be one of {list(choices)}")
    return value


def _optional_int(spec_id: str, key: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"specification '{spec_id}': {key} must be an integer or null")
    return value


def _to_spec(spec_id: str, resolved: Mapping[str, Any]) -> Spec:
    unknown = sorted(set(resolved) - SPEC_TOP_KEYS)
    if unknown:
        raise ConfigError(f"specification '{spec_id}': unknown keys {unknown}")

    label = resolved.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ConfigError(f"specification '{spec_id}': label must be a non-empty string")

    universe = resolved.get("universe", {})
    months = resolved.get("months", {})
    if not isinstance(universe, Mapping) or not isinstance(months, Mapping):
        raise ConfigError(f"specification '{spec_id}': universe and months must be mappings")
    unknown_universe = sorted(set(universe) - SPEC_UNIVERSE_KEYS)
    if unknown_universe:
        raise ConfigError(f"specification '{spec_id}': unknown universe keys {unknown_universe}")
    unknown_months = sorted(set(months) - SPEC_MONTHS_KEYS)
    if unknown_months:
        raise ConfigError(f"specification '{spec_id}': unknown months keys {unknown_months}")

    clear_only = universe.get("clear_only")
    if not isinstance(clear_only, bool):
        raise ConfigError(f"specification '{spec_id}': universe.clear_only must be a boolean")
    extra_terms = resolved.get("extra_terms")
    if not isinstance(extra_terms, list) or not all(isinstance(term, str) for term in extra_terms):
        raise ConfigError(f"specification '{spec_id}': extra_terms must be a list of strings")
    exploratory = resolved.get("exploratory")
    if not isinstance(exploratory, bool):
        raise ConfigError(f"specification '{spec_id}': exploratory must be a boolean")

    cov_params = resolved.get("cov_params")
    if not isinstance(cov_params, Mapping):
        raise ConfigError(f"specification '{spec_id}': cov_params must be a mapping")
    for param, param_value in cov_params.items():
        if param == "kernel":
            if not isinstance(param_value, str):
                raise ConfigError(f"specification '{spec_id}': cov_params.kernel must be a string")
        elif param == "bandwidth":
            if isinstance(param_value, bool) or not isinstance(param_value, int):
                raise ConfigError(
                    f"specification '{spec_id}': cov_params.bandwidth must be an integer"
                )
        else:
            raise ConfigError(f"specification '{spec_id}': unknown cov_params key {param!r}")

    return Spec(
        id=spec_id,
        label=label,
        group=_require_choice(spec_id, "group", resolved.get("group"), SPEC_GROUPS),
        portfolio=_require_choice(spec_id, "portfolio", resolved.get("portfolio"), SPEC_PORTFOLIOS),
        dep_var=_require_choice(spec_id, "dep_var", resolved.get("dep_var"), SPEC_DEP_VARS),
        convention=_require_choice(
            spec_id, "convention", resolved.get("convention"), SPEC_CONVENTIONS
        ),
        cov=_require_choice(spec_id, "cov", resolved.get("cov"), SPEC_COVS),
        cov_params=dict(cov_params),
        pub_year_max=_optional_int(spec_id, "universe.pub_year_max", universe.get("pub_year_max")),
        sample_start_year_min=_optional_int(
            spec_id, "universe.sample_start_year_min", universe.get("sample_start_year_min")
        ),
        clear_only=clear_only,
        start_yyyymm=_optional_int(spec_id, "months.start_yyyymm", months.get("start_yyyymm")),
        end_yyyymm=_optional_int(spec_id, "months.end_yyyymm", months.get("end_yyyymm")),
        extra_terms=tuple(extra_terms),
        exploratory=exploratory,
    )


def load_specifications(path: Path) -> list[Spec]:
    """Load and validate `config/specifications.yaml` into resolved :class:`Spec` objects.

    Resolution starts from ``defaults``, applies the ``base`` entry recursively, then the
    entry itself; nested mappings (``universe``, ``months``, ``cov_params``) merge key by
    key with the later value winning. Unknown keys and values outside the documented
    enums raise :class:`ConfigError`. File order is preserved.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"specifications file not found: {path}")
    with path.open("rb") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    raw_entries = data.get("specifications")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ConfigError(f"{path}: 'specifications' must be a non-empty list")
    defaults = data.get("defaults")
    if not isinstance(defaults, Mapping):
        raise ConfigError(f"{path}: 'defaults' must be a mapping")

    entries: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{path}: specifications[{index}] must be a mapping")
        spec_id = entry.get("id")
        if not isinstance(spec_id, str) or not spec_id:
            raise ConfigError(f"{path}: specifications[{index}] has no string 'id'")
        if spec_id in entries:
            raise ConfigError(f"{path}: duplicate specification id {spec_id!r}")
        entries[spec_id] = entry

    specs: list[Spec] = []
    for entry in raw_entries:
        spec_id = entry["id"]
        resolved = _merge_spec_mappings(dict(defaults), _resolve_spec_entry(entries, spec_id, ()))
        specs.append(_to_spec(spec_id, resolved))
    return specs


PHASE2_TOP_KEYS: frozenset[str] = frozenset(
    {"version", "bootstrap", "multiple_testing", "vintages"}
)
BOOTSTRAP_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "base_spec",
        "method",
        "block_length_months",
        "reps",
        "seed",
        "statistics",
        "interval",
        "level",
        "exploratory",
    }
)
BOOTSTRAP_METHODS: tuple[str, ...] = ("moving_block_by_month",)
BOOTSTRAP_STATISTICS: tuple[str, ...] = (
    "coef_post_sample",
    "coef_post_publication",
    "decay_pct_post_sample",
    "decay_pct_post_publication",
)
BOOTSTRAP_INTERVALS: tuple[str, ...] = ("percentile",)
MULTIPLE_TESTING_KEYS: frozenset[str] = frozenset(
    {"hurdle_t", "weak_rule", "missing_tstat", "tstat_sources", "specs"}
)
MULTIPLE_TESTING_SPEC_KEYS: frozenset[str] = frozenset(
    {"id", "base_spec", "tstat_source", "extra_terms", "exploratory"}
)
WEAK_RULES: tuple[str, ...] = ("tstat_below_hurdle",)
MISSING_TSTAT_RULES: tuple[str, ...] = ("exclude_and_count",)
VINTAGES_KEYS: frozenset[str] = frozenset(
    {"primary", "compare_preference", "predictor_set", "specs", "metrics", "exploratory"}
)
VINTAGE_PREDICTOR_SETS: tuple[str, ...] = ("intersection",)


def _phase2_require(section: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"{where}: missing key '{key}'")
    return section[key]


def _phase2_mapping(section: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = _phase2_require(section, key, where)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: '{key}' must be a mapping")
    return value


def _phase2_reject_unknown(section: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")


def _phase2_string(section: Mapping[str, Any], key: str, where: str) -> str:
    value = _phase2_require(section, key, where)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: {key} must be a non-empty string")
    return value


def _phase2_choice(
    section: Mapping[str, Any], key: str, where: str, choices: tuple[str, ...]
) -> str:
    value = _phase2_string(section, key, where)
    if value not in choices:
        raise ConfigError(f"{where}: {key} must be one of {list(choices)}, got {value!r}")
    return value


def _phase2_int(
    section: Mapping[str, Any], key: str, where: str, *, minimum: int | None = None
) -> int:
    value = _phase2_require(section, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: {key} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{where}: {key} must be >= {minimum}")
    return value


def _phase2_float(section: Mapping[str, Any], key: str, where: str) -> float:
    value = _phase2_require(section, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: {key} must be a number")
    return float(value)


def _phase2_bool(section: Mapping[str, Any], key: str, where: str) -> bool:
    value = _phase2_require(section, key, where)
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: {key} must be a boolean")
    return value


def _phase2_string_list(
    section: Mapping[str, Any],
    key: str,
    where: str,
    *,
    choices: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    value = _phase2_require(section, key, where)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: {key} must be a non-empty list of strings")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{where}: {key} must contain only non-empty strings")
        if choices is not None and item not in choices:
            raise ConfigError(f"{where}: {key} entry {item!r} must be one of {list(choices)}")
    return tuple(value)


def _to_bootstrap_prereg(section: Mapping[str, Any]) -> BootstrapPrereg:
    where = "[bootstrap]"
    _phase2_reject_unknown(section, BOOTSTRAP_KEYS, where)
    level = _phase2_float(section, "level", where)
    if not 0.0 < level < 1.0:
        raise ConfigError(f"{where}: level must be strictly between 0 and 1, got {level}")
    return BootstrapPrereg(
        id=_phase2_string(section, "id", where),
        base_spec=_phase2_string(section, "base_spec", where),
        method=_phase2_choice(section, "method", where, BOOTSTRAP_METHODS),
        block_length_months=_phase2_int(section, "block_length_months", where, minimum=1),
        reps=_phase2_int(section, "reps", where, minimum=1),
        seed=_phase2_int(section, "seed", where, minimum=0),
        statistics=_phase2_string_list(section, "statistics", where, choices=BOOTSTRAP_STATISTICS),
        interval=_phase2_choice(section, "interval", where, BOOTSTRAP_INTERVALS),
        level=level,
        exploratory=_phase2_bool(section, "exploratory", where),
    )


def _to_multiple_testing_spec(
    section: Mapping[str, Any], index: int, tstat_sources: set[str]
) -> MultipleTestingSpecPrereg:
    where = f"[multiple_testing] specs[{index}]"
    _phase2_reject_unknown(section, MULTIPLE_TESTING_SPEC_KEYS, where)
    tstat_source = _phase2_string(section, "tstat_source", where)
    if tstat_source not in tstat_sources:
        raise ConfigError(
            f"{where}: tstat_source {tstat_source!r} is not in tstat_sources "
            f"{sorted(tstat_sources)}"
        )
    return MultipleTestingSpecPrereg(
        id=_phase2_string(section, "id", where),
        base_spec=_phase2_string(section, "base_spec", where),
        tstat_source=tstat_source,
        extra_terms=_phase2_string_list(section, "extra_terms", where),
        exploratory=_phase2_bool(section, "exploratory", where),
    )


def _to_multiple_testing_prereg(section: Mapping[str, Any]) -> MultipleTestingPrereg:
    where = "[multiple_testing]"
    _phase2_reject_unknown(section, MULTIPLE_TESTING_KEYS, where)
    sources = _phase2_mapping(section, "tstat_sources", where)
    if not sources:
        raise ConfigError(f"{where}: tstat_sources must be a non-empty mapping")
    for source, description in sources.items():
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise ConfigError(
                f"{where}: tstat_sources must map non-empty strings to non-empty strings"
            )
    raw_specs = _phase2_require(section, "specs", where)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ConfigError(f"{where}: specs must be a non-empty list of mappings")
    specs: list[MultipleTestingSpecPrereg] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where}: specs[{index}] must be a mapping")
        spec = _to_multiple_testing_spec(raw, index, set(sources))
        if spec.id in seen_ids:
            raise ConfigError(f"{where}: duplicate spec id {spec.id!r}")
        seen_ids.add(spec.id)
        specs.append(spec)
    return MultipleTestingPrereg(
        hurdle_t=_phase2_float(section, "hurdle_t", where),
        weak_rule=_phase2_choice(section, "weak_rule", where, WEAK_RULES),
        missing_tstat=_phase2_choice(section, "missing_tstat", where, MISSING_TSTAT_RULES),
        tstat_sources=dict(sources),
        specs=tuple(specs),
    )


def _to_vintages_prereg(section: Mapping[str, Any]) -> VintagesPrereg:
    where = "[vintages]"
    _phase2_reject_unknown(section, VINTAGES_KEYS, where)
    raw_preference = _phase2_require(section, "compare_preference", where)
    if not isinstance(raw_preference, list) or not raw_preference:
        raise ConfigError(f"{where}: compare_preference must be a non-empty list of vintages")
    for item in raw_preference:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{where}: compare_preference must contain only integers")
    return VintagesPrereg(
        primary=_phase2_int(section, "primary", where, minimum=1),
        compare_preference=tuple(raw_preference),
        predictor_set=_phase2_choice(section, "predictor_set", where, VINTAGE_PREDICTOR_SETS),
        specs=_phase2_string_list(section, "specs", where),
        metrics=_phase2_string_list(section, "metrics", where),
        exploratory=_phase2_bool(section, "exploratory", where),
    )


def load_phase2_preregistration(path: Path) -> Phase2Prereg:
    """Load and validate `config/phase2_preregistration.yaml`.

    Returns the three pre-registered Phase 2 analyses as frozen dataclasses:
    :class:`BootstrapPrereg`, :class:`MultipleTestingPrereg` and
    :class:`VintagesPrereg`. Validation is strict — unknown keys, missing keys, wrong
    types, invalid enum values, duplicate specification ids and a specification that
    references an undeclared ``tstat_source`` all raise :class:`ConfigError`, so a typo
    can never silently change a registered analysis.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    where = str(path)
    _phase2_reject_unknown(data, PHASE2_TOP_KEYS, where)
    return Phase2Prereg(
        version=_phase2_int(data, "version", where, minimum=1),
        bootstrap=_to_bootstrap_prereg(_phase2_mapping(data, "bootstrap", where)),
        multiple_testing=_to_multiple_testing_prereg(
            _phase2_mapping(data, "multiple_testing", where)
        ),
        vintages=_to_vintages_prereg(_phase2_mapping(data, "vintages", where)),
    )
