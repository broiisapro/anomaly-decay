"""Tests for the report helpers.

The helpers decide where results live and how numbers are formatted, so the tests pin the
repository-root path resolution (never the process working directory), the schema-typed
read (with and without ``anomaly_decay.io.read_results``), the pending-value rules, the
table builders on the committed results fixture, and the structure of ``report/index.qmd``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay import io as io_module
from anomaly_decay import reporting
from anomaly_decay.schemas import RESULTS_SCHEMAS, validate_frame
from IPython.display import Image, Markdown

MAIN_CSV = (
    "spec,term,coef,se,t,n_predictors,n_obs\n"
    "main,post_sample,0.15,0.05,3.0,100,5000\n"
    "main,post_publication,-0.02,0.04,-0.5,100,4000\n"
)
QMD_PATH = reporting.REPO_ROOT / "report" / "index.qmd"
FIXTURE_DIR = reporting.REPO_ROOT / "tests" / "fixtures" / "results_fixture"
FIXTURE_FILES = (
    "coverage.csv",
    "decay.csv",
    "event_time.csv",
    "extension.csv",
    "main.csv",
    "predictor_summary.csv",
    "robustness.csv",
    "specs_run.csv",
)
"""The files the committed fictional fixture covers (the eight Phase 1 results files)."""
SECTIONS = (
    "# Question",
    "# Data",
    "# Method",
    "# Main result vs the original paper",
    "# Robustness",
    "# Extension",
    "# Deviations from the original paper",
    "# Limitations",
    "# Reproduction",
)


def _write_main(results_dir: Path) -> None:
    (results_dir / "main.csv").write_text(MAIN_CSV, encoding="utf-8")


def test_results_dir_defaults_to_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(reporting.RESULTS_ENV_VAR, raising=False)
    assert reporting.results_dir() == reporting.REPO_ROOT / "results"


def test_results_dir_absolute_env_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(reporting.RESULTS_ENV_VAR, str(tmp_path))
    assert reporting.results_dir() == tmp_path


def test_results_dir_relative_env_resolves_against_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(reporting.RESULTS_ENV_VAR, "tests/fixtures/dummy_results")
    assert reporting.results_dir() == reporting.REPO_ROOT / "tests" / "fixtures" / "dummy_results"


def test_results_dir_ignores_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(reporting.RESULTS_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert reporting.results_dir() == reporting.REPO_ROOT / "results"


def test_results_dir_empty_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(reporting.RESULTS_ENV_VAR, "")
    assert reporting.results_dir() == reporting.REPO_ROOT / "results"


def test_load_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown results file"):
        reporting.load("not-a-results-file.csv", where=tmp_path)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        reporting.load("main.csv", where=tmp_path)


def test_load_falls_back_to_schema_read_without_io_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_main(tmp_path)
    monkeypatch.delattr(io_module, "read_results", raising=False)
    frame = reporting.load("main.csv", where=tmp_path)
    assert frame.schema == {
        "spec": pl.Utf8,
        "term": pl.Utf8,
        "coef": pl.Float64,
        "se": pl.Float64,
        "t": pl.Float64,
        "n_predictors": pl.Int64,
        "n_obs": pl.Int64,
    }
    assert frame.height == 2


def test_load_prefers_io_read_results_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_reader(name: str, base: Path) -> pl.DataFrame:
        calls.append((name, base))
        return pl.DataFrame({"spec": ["main"]}, schema={"spec": pl.Utf8})

    monkeypatch.setattr(io_module, "read_results", fake_reader, raising=False)
    frame = reporting.load("main.csv", where=tmp_path)
    assert calls == [("main.csv", tmp_path)]
    assert frame.columns == ["spec"]


def test_load_optional_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert reporting.load_optional("main.csv", where=tmp_path) is None


def test_load_uses_environment_results_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_main(tmp_path)
    monkeypatch.setenv(reporting.RESULTS_ENV_VAR, str(tmp_path))
    frame = reporting.load("main.csv")
    assert frame.height == 2


def test_lookup_returns_first_match() -> None:
    frame = pl.DataFrame({"term": ["a", "b", "b"], "coef": [1.0, 2.0, 3.0]})
    assert reporting.lookup(frame, "coef", term="b") == 2.0


def test_lookup_returns_none_without_match() -> None:
    frame = pl.DataFrame({"term": ["a"], "coef": [1.0]})
    assert reporting.lookup(frame, "coef", term="missing") is None


def test_lookup_rejects_unknown_column() -> None:
    frame = pl.DataFrame({"term": ["a"]})
    with pytest.raises(ValueError, match="column not in frame"):
        reporting.lookup(frame, "coef", term="a")


def test_lookup_rejects_unknown_condition_column() -> None:
    frame = pl.DataFrame({"term": ["a"], "coef": [1.0]})
    with pytest.raises(ValueError, match="condition column not in frame"):
        reporting.lookup(frame, "coef", spec="main")


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (1.2345, 2, "1.23"),
        (1, 3, "1.000"),
        (0.0, 1, "0.0"),
        (None, 2, "n/a"),
        (float("nan"), 2, "n/a"),
    ],
)
def test_num(value: float | int | None, digits: int, expected: str) -> None:
    assert reporting.num(value, digits) == expected


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (26.0, 1, "26.0%"),
        (-3.456, 2, "-3.46%"),
        (None, 1, "n/a"),
        (float("nan"), 1, "n/a"),
    ],
)
def test_pct(value: float | None, digits: int, expected: str) -> None:
    assert reporting.pct(value, digits) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.345, "2.35"), (-0.5, "-0.50"), (None, "n/a"), (float("nan"), "n/a")],
)
def test_tstat(value: float | None, expected: str) -> None:
    assert reporting.tstat(value) == expected


def test_table_renders_markdown_pipe_table() -> None:
    frame = pl.DataFrame({"term": ["post_sample"], "coef": [0.15]})
    text = reporting.table(frame, ["term", "coef"], ["Term", "Coef"])
    assert text.splitlines() == [
        "| Term | Coef |",
        "| --- | --- |",
        "| post_sample | 0.15 |",
    ]


def test_table_defaults_headers_to_column_names() -> None:
    frame = pl.DataFrame({"spec": ["main"]})
    assert reporting.table(frame, ["spec"]).splitlines()[0] == "| spec |"


def test_table_renders_none_bool_and_escapes_pipes() -> None:
    frame = pl.DataFrame(
        {
            "term": ["a", "b"],
            "value": [None, True],
            "label": ["x | y", "plain"],
        }
    )
    text = reporting.table(frame, ["term", "value", "label"])
    rows = text.splitlines()[2:]
    assert rows[0] == "| a | n/a | x \\| y |"
    assert rows[1] == "| b | yes | plain |"


def test_table_applies_per_column_formats() -> None:
    frame = pl.DataFrame({"coef": [0.15], "t": [None]})
    text = reporting.table(
        frame,
        ["coef", "t"],
        formats={"coef": lambda value: reporting.num(value, 3), "t": reporting.tstat},
    )
    assert "| 0.150 | n/a |" in text


def test_table_selects_requested_columns_in_order() -> None:
    frame = pl.DataFrame({"spec": ["main"], "term": ["post_sample"], "coef": [0.15]})
    text = reporting.table(frame, ["term", "coef"])
    assert text.splitlines()[2] == "| post_sample | 0.15 |"


def test_table_rejects_unknown_column() -> None:
    frame = pl.DataFrame({"spec": ["main"]})
    with pytest.raises(ValueError, match="columns not in frame"):
        reporting.table(frame, ["spec", "coef"])


def test_table_rejects_header_length_mismatch() -> None:
    frame = pl.DataFrame({"spec": ["main"]})
    with pytest.raises(ValueError, match="equal length"):
        reporting.table(frame, ["spec"], ["Spec", "Extra"])


def test_figure_prefers_png_over_svg(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "event_time.svg").write_text("<svg/>", encoding="utf-8")
    assert reporting.figure("event_time", where=tmp_path) == figures / "event_time.svg"
    (figures / "event_time.png").write_bytes(b"\x89PNG\r\n")
    assert reporting.figure("event_time", where=tmp_path) == figures / "event_time.png"


def test_figure_missing_returns_none(tmp_path: Path) -> None:
    assert reporting.figure("event_time", where=tmp_path) is None


def test_figure_block_notes_missing_figure(tmp_path: Path) -> None:
    block = reporting.figure_block("event_time", "Event time", where=tmp_path)
    assert isinstance(block, Markdown)
    assert "Figure not generated" in block.data
    assert "event_time" in block.data


def test_figure_block_embeds_existing_raster_figure(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    png = figures / "coefplot.png"
    png.write_bytes(b"\x89PNG\r\n")
    block = reporting.figure_block("coefplot", "Robustness", where=tmp_path)
    assert isinstance(block, Image)
    assert block.filename == str(png)
    assert block.alt == "Robustness"


def test_figure_block_references_svg_only_figure_as_markdown(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    svg = figures / "event_time.svg"
    svg.write_text("<svg/>", encoding="utf-8")
    block = reporting.figure_block("event_time", "Event time", where=tmp_path)
    assert isinstance(block, Markdown)
    assert f"![Event time]({svg.as_posix()})" in block.data


def test_original_paper_values_are_verified() -> None:
    data = reporting.original_paper()
    citation = data["citation"]
    assert citation["authors"] == "McLean, R. David, and Jeffrey Pontiff"
    assert citation["year"] == 2016
    assert citation["doi"] == "10.1111/jofi.12365"
    assert data["status"] == "verified"
    assert data["table1"]["in_sample_mean_pct"] == 0.582
    table = data["table2_col1"]
    assert table["status"] == "verified"
    assert table["post_sample"] == {"coef": -0.150, "se": 0.077}
    assert table["post_publication"] == {"coef": -0.337, "se": 0.090}
    assert data["headline"] == {
        "post_sample_decline_pct": 26,
        "post_publication_decline_pct": 58,
    }


def test_original_paper_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "original_paper.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        reporting.original_paper(path)


def test_report_contains_the_nine_sections() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert f"\n{heading}\n" in text, f"missing section: {heading}"


def test_report_reads_results_only_through_reporting() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    assert "read_csv" not in text
    assert "read_parquet" not in text


def test_report_references_every_figure() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    for name in ("event_time", "distribution", "coefplot"):
        assert f'"{name}"' in text, f"figure not referenced: {name}"


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_results_fixture_loads_against_its_schema(name: str) -> None:
    frame = reporting.load(name, where=FIXTURE_DIR)
    validate_frame(frame, RESULTS_SCHEMAS[name])


def test_main_table_on_results_fixture() -> None:
    main = reporting.load("main.csv", where=FIXTURE_DIR)
    text = reporting.table(
        main,
        ["term", "coef", "se", "t", "n_predictors", "n_obs"],
        ["Term", "Coef (%/mo)", "SE", "t", "N pred.", "N obs"],
        formats={
            "coef": lambda value: reporting.num(value, 3),
            "se": lambda value: reporting.num(value, 3),
            "t": reporting.tstat,
        },
    )
    assert "| post_sample | -0.333 | 0.155 | -2.14 | 12 | 5000 |" in text
    assert "| post_publication | -0.666 | 0.200 | -3.33 | 12 | 5000 |" in text


def test_decay_table_on_results_fixture() -> None:
    decay = reporting.load("decay.csv", where=FIXTURE_DIR)
    text = reporting.table(
        decay,
        ["method", "term", "decay_pct", "se_pct", "in_sample_mean", "n_predictors"],
        formats={
            "decay_pct": lambda value: reporting.pct(value, 1),
            "se_pct": lambda value: reporting.num(value, 1),
            "in_sample_mean": lambda value: reporting.num(value, 3),
        },
    )
    assert "| pooled_ratio | post_publication | -60.0% | 18.0 | 1.110 | 12 |" in text
    assert "| normalized | post_sample | -33.3% | 11.1 | 1.000 | 12 |" in text


def test_robustness_and_extension_tables_on_results_fixture() -> None:
    robustness = reporting.load("robustness.csv", where=FIXTURE_DIR)
    extension = reporting.load("extension.csv", where=FIXTURE_DIR)
    for frame in (robustness, extension):
        text = reporting.table(
            frame,
            ["label", "term", "coef", "se", "t", "n_predictors", "n_obs"],
            formats={
                "coef": lambda value: reporting.num(value, 3),
                "se": lambda value: reporting.num(value, 3),
                "t": reporting.tstat,
            },
        )
        assert len(text.splitlines()) == frame.height + 2
    interaction = reporting.lookup(
        extension, "coef", spec="ext_pub_after_2000", term="post_publication_x_pub_after_2000"
    )
    assert interaction == -0.222


def test_specs_table_lists_every_specification_from_the_fixture() -> None:
    specs = reporting.load("specs_run.csv", where=FIXTURE_DIR)
    text = reporting.table(specs, ["spec", "label", "status", "exploratory"])
    assert len(specs) == 12
    for spec_id in specs["spec"].to_list():
        assert f"| {spec_id} |" in text
    assert "| no |" in text


def test_coverage_table_on_results_fixture() -> None:
    coverage = reporting.load("coverage.csv", where=FIXTURE_DIR)
    text = reporting.table(coverage, ["item", "value"], ["Item", "Value"])
    assert "| vintage | 202510 |" in text
    assert "| n_predictors_op | 15 |" in text
