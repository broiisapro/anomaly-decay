"""Tests for loading and resolving config/specifications.yaml."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from anomaly_decay.config import ConfigError, Spec, load_specifications

REPO_ROOT = Path(__file__).parents[1]
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"

EXPECTED_IDS = [
    "main",
    "main_twoway",
    "main_dk",
    "pubconv_alt",
    "start_1963",
    "clear_only",
    "ew_deciles",
    "vw_deciles",
    "factor_adj",
    "normalized",
    "full_extension",
    "ext_pub_after_2000",
]


@pytest.fixture(scope="module")
def specs() -> list[Spec]:
    return load_specifications(SPECS_PATH)


def _by_id(specs: list[Spec], spec_id: str) -> Spec:
    return next(spec for spec in specs if spec.id == spec_id)


def test_all_twelve_specs_load_in_file_order(specs: list[Spec]) -> None:
    assert [spec.id for spec in specs] == EXPECTED_IDS


def test_defaults_and_main_fields(specs: list[Spec]) -> None:
    main = _by_id(specs, "main")
    assert main.group == "main"
    assert main.portfolio == "op"
    assert main.dep_var == "ret"
    assert main.convention == "default"
    assert main.cov == "clustered_time"
    assert main.cov_params == {}
    assert main.pub_year_max == 2013
    assert main.sample_start_year_min is None
    assert main.clear_only is False
    assert main.start_yyyymm is None
    assert main.end_yyyymm == 201312
    assert main.extra_terms == ()
    assert main.exploratory is False
    assert main.label == "Main: predictors published by 2013, returns through 2013-12"


def test_base_inheritance_keeps_universe_from_base(specs: list[Spec]) -> None:
    twoway = _by_id(specs, "main_twoway")
    assert twoway.cov == "clustered_twoway"
    assert twoway.pub_year_max == 2013  # inherited from main
    assert twoway.end_yyyymm == 201312

    dk = _by_id(specs, "main_dk")
    assert dk.cov == "driscoll_kraay"
    assert dk.cov_params == {"kernel": "bartlett", "bandwidth": 12}
    assert dk.pub_year_max == 2013

    alt = _by_id(specs, "pubconv_alt")
    assert alt.convention == "alt"
    assert alt.pub_year_max == 2013


def test_nested_mapping_merges_key_by_key(specs: list[Spec]) -> None:
    start = _by_id(specs, "start_1963")
    assert start.pub_year_max == 2013
    assert start.sample_start_year_min == 1963

    clear = _by_id(specs, "clear_only")
    assert clear.clear_only is True
    assert clear.pub_year_max == 2013

    factor = _by_id(specs, "factor_adj")
    assert factor.dep_var == "ret_factor_adj"
    assert factor.start_yyyymm == 196307
    assert factor.end_yyyymm == 201312  # override of main's end


def test_portfolio_and_extension_specs(specs: list[Spec]) -> None:
    assert _by_id(specs, "ew_deciles").portfolio == "deciles_ew"
    assert _by_id(specs, "vw_deciles").portfolio == "deciles_vw"
    assert _by_id(specs, "normalized").dep_var == "ret_norm"

    extension = _by_id(specs, "full_extension")
    assert extension.group == "extension"
    assert extension.pub_year_max is None
    assert extension.end_yyyymm is None

    split = _by_id(specs, "ext_pub_after_2000")
    assert split.group == "extension"
    assert split.extra_terms == ("post_publication_x_pub_after_2000",)


def test_specs_are_frozen(specs: list[Spec]) -> None:
    with pytest.raises(FrozenInstanceError):
        _by_id(specs, "main").portfolio = "deciles_ew"  # type: ignore[misc]


def test_loading_is_deterministic(specs: list[Spec]) -> None:
    assert load_specifications(SPECS_PATH) == specs


def _write_specs(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "specifications.yaml"
    path.write_text(body, encoding="utf-8")
    return path


DEFAULTS = """
version: 1
defaults:
  group: robustness
  portfolio: op
  dep_var: ret
  convention: default
  cov: clustered_time
  cov_params: {}
  universe:
    pub_year_max: null
    sample_start_year_min: null
    clear_only: false
  months:
    start_yyyymm: null
    end_yyyymm: null
  extra_terms: []
  exploratory: false
specifications:
"""


def test_unknown_key_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: bad
    label: "Bad"
    mysterious_option: 1
""",
    )
    with pytest.raises(ConfigError, match="unknown keys"):
        load_specifications(path)


def test_unknown_base_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: bad
    label: "Bad"
    base: does_not_exist
""",
    )
    with pytest.raises(ConfigError, match="unknown base"):
        load_specifications(path)


def test_circular_base_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: a
    label: "A"
    base: b
  - id: b
    label: "B"
    base: a
""",
    )
    with pytest.raises(ConfigError, match="circular"):
        load_specifications(path)


def test_bad_enum_value_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: bad
    label: "Bad"
    portfolio: equally_weighted
""",
    )
    with pytest.raises(ConfigError, match="portfolio"):
        load_specifications(path)


def test_bad_cov_params_key_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: bad
    label: "Bad"
    cov: driscoll_kraay
    cov_params: {window: 12}
""",
    )
    with pytest.raises(ConfigError, match="cov_params"):
        load_specifications(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: same
    label: "One"
  - id: same
    label: "Two"
""",
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_specifications(path)


def test_missing_label_raises(tmp_path: Path) -> None:
    path = _write_specs(
        tmp_path,
        DEFAULTS
        + """
  - id: bad
    group: main
""",
    )
    with pytest.raises(ConfigError, match="label"):
        load_specifications(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_specifications(tmp_path / "nope.yaml")
