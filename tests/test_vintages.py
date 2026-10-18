"""Offline tests for the vintage comparison (Phase 2, FR-2.3).

A synthetic second vintage is built inside each test from the committed 202510 fixture:
predictors are dropped and returns or calendar years are perturbed, so the tests check
the intersection logic, the sample counts and the output schema without any network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.config import load_specifications
from anomaly_decay.estimate.vintages import (
    VINTAGE_SPEC_IDS,
    VintageComparison,
    VintageComparisonError,
    build_vintage_comparison,
    main,
    metadata_date_changes,
)
from anomaly_decay.io import RESULTS_SORT_KEYS, read_results, write_csv_canonical
from anomaly_decay.schemas import RESULTS_SCHEMAS

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE_VINTAGE = 202510
COMPARE_VINTAGE = 202410
SUPPLEMENT = REPO / "config" / "predictor_dates_supplement.csv"
SPECS = REPO / "config" / "specifications.yaml"
PINNED = FIXTURES / "data" / "pinned"
CSV_OVERRIDES: dict[str, pl.DataTypeClass] = {"port": pl.Utf8, "date": pl.Utf8}

pytestmark = pytest.mark.skipif(
    "vintage_comparison.csv" not in RESULTS_SCHEMAS,
    reason="Phase 2 results schemas not merged yet",
)


def _fixture_acronyms() -> list[str]:
    doc = pl.read_csv(FIXTURES / "data" / "osap" / str(SOURCE_VINTAGE) / "signal_doc.csv")
    return list(doc["Acronym"])


def _fixture_count() -> int:
    return len(_fixture_acronyms())


def _write_vintage(
    data_dir: Path,
    vintage: int,
    *,
    keep: Iterable[str] | None = None,
    shift_post_sample_ret: Mapping[str, float] | None = None,
    date_changes: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    """Derive a vintage directory from the committed fixture.

    *keep* restricts the signal doc and the long-short rows (default: everything);
    *shift_post_sample_ret* adds a constant to one predictor's returns from 2014-01 on
    (the estimators absorb a shift of every month, so the shift must change the
    post-sample months); *date_changes* overwrites signal-doc columns (``Year``,
    ``SampleStartYear``, ``SampleEndYear``).
    """
    out = data_dir / str(vintage)
    out.mkdir(parents=True, exist_ok=True)
    doc = pl.read_csv(FIXTURES / "data" / "osap" / str(SOURCE_VINTAGE) / "signal_doc.csv")
    ports = pl.read_csv(
        FIXTURES / "data" / "osap" / str(SOURCE_VINTAGE) / "port_op.csv",
        schema_overrides=CSV_OVERRIDES,
    )
    if keep is not None:
        names = list(keep)
        doc = doc.filter(pl.col("Acronym").is_in(names))
        ports = ports.filter(pl.col("signalname").is_in(names))
    if date_changes:
        for signalname, changes in date_changes.items():
            for field, value in changes.items():
                doc = doc.with_columns(
                    pl.when(pl.col("Acronym") == signalname)
                    .then(pl.lit(value))
                    .otherwise(pl.col(field))
                    .alias(field)
                )
    if shift_post_sample_ret:
        yyyymm = pl.col("date").str.slice(0, 4).cast(pl.Int32) * 100 + pl.col("date").str.slice(
            5, 2
        ).cast(pl.Int32)
        for signalname, shift in shift_post_sample_ret.items():
            ports = ports.with_columns(
                pl.when((pl.col("signalname") == signalname) & (yyyymm >= 201401))
                .then(pl.col("ret") + shift)
                .otherwise(pl.col("ret"))
                .alias("ret")
            )
    write_csv_canonical(doc, out / "signal_doc.csv", sort_by=["Acronym"])
    write_csv_canonical(ports, out / "port_op.csv", sort_by=["signalname", "port", "date"])


def _comparison(
    tmp_path: Path,
    *,
    keep_410: Iterable[str] | None = None,
    keep_510: Iterable[str] | None = None,
    shift: Mapping[str, float] | None = None,
    date_changes: Mapping[str, Mapping[str, int]] | None = None,
    write: bool = True,
) -> tuple[VintageComparison, Path]:
    data_dir = tmp_path / "data" / "raw"
    _write_vintage(data_dir, COMPARE_VINTAGE, keep=keep_410, shift_post_sample_ret=shift)
    _write_vintage(data_dir, SOURCE_VINTAGE, keep=keep_510, date_changes=date_changes)
    results_dir = tmp_path / "results"
    comparison = build_vintage_comparison(
        data_dir,
        (COMPARE_VINTAGE, SOURCE_VINTAGE),
        pinned_dir=PINNED,
        specs=load_specifications(SPECS),
        supplement_path=SUPPLEMENT,
        results_dir=results_dir if write else None,
    )
    return comparison, results_dir


def test_comparison_writes_the_preregistered_rows(tmp_path: Path) -> None:
    comparison, results_dir = _comparison(tmp_path)
    results = comparison.results

    written = read_results("vintage_comparison.csv", results_dir)
    assert written.equals(results)
    assert written.columns == list(RESULTS_SCHEMAS["vintage_comparison.csv"])
    assert RESULTS_SORT_KEYS["vintage_comparison.csv"] == ("spec", "vintage", "term")
    assert written.height == len(VINTAGE_SPEC_IDS) * 2 * 2
    assert set(written["spec"]) == set(VINTAGE_SPEC_IDS)
    assert set(written["term"]) == {"post_sample", "post_publication"}
    assert set(written["vintage"]) == {COMPARE_VINTAGE, SOURCE_VINTAGE}
    assert set(written["predictor_set"]) == {"intersection"}
    assert written.select("spec", "vintage", "term").n_unique() == written.height


def test_dropped_predictor_leaves_the_intersection(tmp_path: Path) -> None:
    total = _fixture_count()
    keep = [name for name in _fixture_acronyms() if name != "MaxRet"]
    comparison, results_dir = _comparison(tmp_path, keep_410=keep)

    assert total == 15
    assert len(comparison.intersection) == total - 1
    assert "MaxRet" not in comparison.intersection

    written = read_results("vintage_comparison.csv", results_dir)
    extension = written.filter(pl.col("spec") == "full_extension")
    assert set(extension["n_predictors"]) == {total - 1}
    main_rows = written.filter(pl.col("spec") == "main")
    assert main_rows["n_predictors"].max() < total - 1


def test_second_run_is_byte_identical(tmp_path: Path) -> None:
    first, first_dir = _comparison(tmp_path / "first")
    second, second_dir = _comparison(tmp_path / "second")
    assert first.results.equals(second.results)
    assert (first_dir / "vintage_comparison.csv").read_bytes() == (
        second_dir / "vintage_comparison.csv"
    ).read_bytes()


def test_returns_perturbation_moves_the_estimates(tmp_path: Path) -> None:
    comparison, _ = _comparison(tmp_path, shift={"Mom12m": 1.0})
    full = comparison.results.filter(
        (pl.col("spec") == "full_extension") & (pl.col("term") == "post_sample")
    )
    by_vintage = {row["vintage"]: row for row in full.iter_rows(named=True)}
    assert by_vintage[COMPARE_VINTAGE]["coef"] != by_vintage[SOURCE_VINTAGE]["coef"]
    assert by_vintage[COMPARE_VINTAGE]["decay_pct"] != by_vintage[SOURCE_VINTAGE]["decay_pct"]


def test_metadata_date_changes_are_reported(tmp_path: Path) -> None:
    comparison, _ = _comparison(
        tmp_path,
        date_changes={"MaxRet": {"SampleEndYear": 2011, "Year": 2012}},
    )
    changes = comparison.date_changes
    assert set(changes["field"]) == {"sample_end_year", "pub_year"}
    assert set(changes["signalname"]) == {"MaxRet"}
    assert set(changes["vintage"]) == {COMPARE_VINTAGE}
    assert changes.height == 2
    # A date-only revision does not change the intersection; `main` still filters to
    # predictors published by 2013, so its count stays below the full intersection.
    assert len(comparison.intersection) == _fixture_count()
    assert comparison.results["n_predictors"].max() == _fixture_count()
    assert comparison.results["n_predictors"].min() < _fixture_count()


def test_metadata_date_changes_helper(tmp_path: Path) -> None:
    base = pl.DataFrame(
        {
            "signalname": ["A", "B"],
            "sample_start_year": [1963, 1970],
            "sample_end_year": [2000, 2010],
            "pub_year": [2001, 2011],
        }
    )
    revised = base.with_columns(
        pl.when(pl.col("signalname") == "B")
        .then(pl.lit(2009))
        .otherwise(pl.col("sample_end_year"))
        .alias("sample_end_year")
    )
    changes = metadata_date_changes({202510: base, 202410: revised}, primary=202510)
    assert changes.height == 1
    row = changes.row(0, named=True)
    assert row == {
        "vintage": 202410,
        "signalname": "B",
        "field": "sample_end_year",
        "primary_value": 2010,
        "vintage_value": 2009,
    }
    restricted = metadata_date_changes(
        {202510: base, 202410: revised}, primary=202510, predictors=["A"]
    )
    assert restricted.height == 0
    only_primary = metadata_date_changes({202510: base}, primary=202510)
    assert only_primary.height == 0
    with pytest.raises(VintageComparisonError, match="primary vintage"):
        metadata_date_changes({202510: base}, primary=202410)


def test_single_vintage_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    _write_vintage(data_dir, SOURCE_VINTAGE)
    with pytest.raises(VintageComparisonError, match="at least two"):
        build_vintage_comparison(
            data_dir,
            (SOURCE_VINTAGE,),
            pinned_dir=PINNED,
            specs=load_specifications(SPECS),
        )


def test_empty_intersection_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    _write_vintage(data_dir, COMPARE_VINTAGE, keep=["Size", "BM"])
    _write_vintage(data_dir, SOURCE_VINTAGE, keep=["Mom12m"])
    with pytest.raises(VintageComparisonError, match="share no predictors"):
        build_vintage_comparison(
            data_dir,
            (COMPARE_VINTAGE, SOURCE_VINTAGE),
            pinned_dir=PINNED,
            specs=load_specifications(SPECS),
        )


def test_missing_signal_doc_is_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    _write_vintage(data_dir, SOURCE_VINTAGE)
    with pytest.raises(VintageComparisonError, match="canonical file not found"):
        build_vintage_comparison(
            data_dir,
            (COMPARE_VINTAGE, SOURCE_VINTAGE),
            pinned_dir=PINNED,
            specs=load_specifications(SPECS),
        )


def _config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "config" / "data.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "[osap]\n"
            f"vintage = {SOURCE_VINTAGE}\n"
            'portfolios = ["op"]\n'
            f"compare_vintages = [{COMPARE_VINTAGE}]\n"
            'raw_dir = "data/raw"\n\n'
            "[french]\n"
            'ff5_url = "https://example.invalid/ff5.zip"\n'
            'umd_url = "https://example.invalid/umd.zip"\n'
            'pinned_dir = "data/pinned"\n'
            'downloaded_on = "2026-10-16"\n'
            f'ff5_sha256 = "{"0" * 64}"\n'
            f'umd_sha256 = "{"0" * 64}"\n'
        ),
        encoding="utf-8",
    )
    return config_path


def test_cli_runs_on_the_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = tmp_path / "data" / "raw"
    _write_vintage(data_dir, COMPARE_VINTAGE)
    _write_vintage(data_dir, SOURCE_VINTAGE)
    results_dir = tmp_path / "results"
    config_path = _config_path(tmp_path)

    exit_code = main(
        [
            "--vintages",
            f"{COMPARE_VINTAGE},{SOURCE_VINTAGE}",
            "--data-dir",
            str(data_dir),
            "--pinned-dir",
            str(PINNED),
            "--specs",
            str(SPECS),
            "--results-dir",
            str(results_dir),
            "--config",
            str(config_path),
            "--supplement",
            str(SUPPLEMENT),
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert f"intersection={_fixture_count()} predictors" in out
    assert "metadata date changes" in out
    written = read_results("vintage_comparison.csv", results_dir)
    assert written.height == 8


def test_cli_rejects_a_vintage_outside_the_config(tmp_path: Path) -> None:
    config_path = _config_path(tmp_path)
    with pytest.raises(SystemExit) as info:
        main(["--vintages", f"{COMPARE_VINTAGE},202001", "--config", str(config_path)])
    assert info.value.code == 2


def test_cli_rejects_a_single_vintage(tmp_path: Path) -> None:
    config_path = _config_path(tmp_path)
    with pytest.raises(SystemExit) as info:
        main(["--vintages", str(SOURCE_VINTAGE), "--config", str(config_path)])
    assert info.value.code == 2


def test_repo_config_allows_the_preference_order() -> None:
    from anomaly_decay.data.fetch import load_compare_vintages

    assert load_compare_vintages(REPO / "config" / "data.toml") == (202410, 202408, 2023)
