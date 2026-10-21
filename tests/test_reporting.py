"""Tests for the report helpers.

The helpers decide where results live and how numbers are formatted, so the tests pin the
repository-root path resolution (never the process working directory), the schema-typed
read (with and without ``anomaly_decay.io.read_results``), the pending-value rules, the
table builders on the committed results fixture, and the structure of ``report/index.qmd``.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest
import yaml
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
MAKEFILE_PATH = reporting.REPO_ROOT / "Makefile"
QUARTO_CONFIG_PATH = reporting.REPO_ROOT / "report" / "_quarto.yml"
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
    "# Headline result",
    "# Data",
    "# Method",
    "# Main result vs the original paper",
    "# Robustness",
    "# Extension",
    "# Phase 2: Bootstrap inference",
    "# Phase 2: Multiple-testing lens",
    "# Phase 2: Data-vintage sensitivity",
    "# Deviations from the original paper",
    "# Limitations",
    "# Reproduction",
)
PHASE2_FRAMES = {
    "bootstrap.csv": {
        "spec": ["main"],
        "statistic": ["coef_post_sample"],
        "estimate": [0.1],
        "boot_mean": [0.1],
        "boot_se": [0.02],
        "ci_low": [0.06],
        "ci_high": [0.14],
        "reps": [2000],
        "block_length": [12],
        "seed": [2027],
    },
    "multiple_testing.csv": {
        "spec": ["mt_weak_original"],
        "tstat_source": ["original_paper"],
        "term": ["post_publication_x_weak"],
        "coef": [-0.05],
        "se": [0.04],
        "t": [-1.25],
        "n_predictors": [90],
        "n_obs": [40000],
        "n_weak": [30],
        "n_excluded_missing_t": [7],
    },
    "tstat_summary.csv": {
        "tstat_source": ["original_paper"],
        "n_predictors": [97],
        "n_missing": [7],
        "n_below_hurdle": [30],
        "median_t": [2.5],
        "hurdle_t": [3.0],
    },
    "vintage_comparison.csv": {
        "spec": ["main"],
        "vintage": [202410],
        "term": ["post_sample"],
        "coef": [-0.14],
        "se": [0.08],
        "decay_pct": [-25.0],
        "n_predictors": [190],
        "n_obs": [50000],
        "predictor_set": ["intersection"],
    },
}
CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_EXPRESSION = re.compile(r"\{python\}[^\n]*")
INLINE_CODE = re.compile(r"`[^`\n]*`")
BACKTICK_EXPRESSION = re.compile(r"`\{python\} [^`]+`")
DECIMAL_NUMBER = re.compile(r"\b\d+\.\d+\b")


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.5, "lower"),
        (26.0, "higher"),
        (0, "unchanged"),
        (None, "n/a"),
        (float("nan"), "n/a"),
    ],
)
def test_direction(value: float | int | None, expected: str) -> None:
    assert reporting.direction(value) == expected


def test_direction_accepts_custom_words() -> None:
    assert reporting.direction(-1.0, positive="more", negative="less", zero="same") == "less"
    assert reporting.direction(1.0, positive="more") == "more"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-2.5, 2.5), (3, 3), (0.0, 0.0), (None, None), (float("nan"), float("nan"))],
)
def test_magnitude(value: float | int | None, expected: float | int | None) -> None:
    result = reporting.magnitude(value)
    if expected is None or (isinstance(expected, float) and expected != expected):
        assert result is None or result != result
    else:
        assert result == expected


def test_direction_and_magnitude_compose_for_sign_prose() -> None:
    coef = -0.333
    assert (
        f"returns are {reporting.direction(coef)} by {reporting.num(reporting.magnitude(coef), 3)}"
        == "returns are lower by 0.333"
    )


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (-0.150, 0.582, -25.77319587628866),
        (1.0, 4.0, 25.0),
        (0.0, 0.582, 0.0),
        (1.0, 0.0, None),
        (None, 0.582, None),
        (0.150, None, None),
        (float("nan"), 0.582, None),
        (0.150, float("nan"), None),
    ],
)
def test_ratio_pct(
    numerator: float | None, denominator: float | None, expected: float | None
) -> None:
    result = reporting.ratio_pct(numerator, denominator)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_ratio_pct_feeds_pct_for_the_headline_decay() -> None:
    assert reporting.pct(reporting.ratio_pct(-0.150, 0.582)) == "-25.8%"


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (1.0, 4.0, 0.25),
        (-0.150, 0.077, -1.948051948051948),
        (0.0, 4.0, 0.0),
        (1.0, 0.0, None),
        (None, 4.0, None),
        (1.0, None, None),
        (float("nan"), 4.0, None),
        (1.0, float("nan"), None),
    ],
)
def test_ratio(numerator: float | None, denominator: float | None, expected: float | None) -> None:
    result = reporting.ratio(numerator, denominator)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_ordered_follows_the_requested_order_and_keeps_unknown_rows() -> None:
    frame = pl.DataFrame(
        {
            "spec": ["z", "main", "unknown", "main"],
            "term": ["post_publication", "post_publication", "post_sample", "post_sample"],
            "value": [1, 2, 3, 4],
        }
    )
    ordered = reporting.ordered(
        frame, {"spec": ["main", "z"], "term": ["post_sample", "post_publication"]}
    )
    assert ordered["value"].to_list() == [4, 2, 1, 3]
    assert ordered.columns == frame.columns


def test_ordered_rejects_unknown_column() -> None:
    with pytest.raises(ValueError, match="order column not in frame"):
        reporting.ordered(pl.DataFrame({"spec": ["main"]}), {"label": ["main"]})


def test_specifications_preserve_the_preregistered_order() -> None:
    specs = reporting.specifications()
    assert [spec.id for spec in specs] == [
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


def test_grouped_table_prints_each_group_label_once() -> None:
    frame = pl.DataFrame(
        {
            "spec": ["main", "main", "main_twoway", "main_twoway"],
            "term": ["post_sample", "post_publication", "post_sample", "post_publication"],
            "coef": [-0.239, -0.337, -0.24, -0.34],
        }
    )
    text = reporting.grouped_table(
        frame,
        "spec",
        ["spec", "term", "coef"],
        ["Specification", "Term", "Coef"],
        formats={"term": reporting.term_label, "coef": lambda value: reporting.num(value, 3)},
        labels={"main": "Main", "main_twoway": "Two-way"},
    )
    rows = text.splitlines()
    assert rows[0] == "| Specification | Term | Coef |"
    assert rows[1] == "| --- | --- | --- |"
    assert rows[2] == "| **Main** | Post-sample | -0.239 |"
    assert rows[3] == "|  | Post-publication | -0.337 |"
    assert rows[4] == "| **Two-way** | Post-sample | -0.240 |"
    assert rows[5] == "|  | Post-publication | -0.340 |"
    assert len(rows) == frame.height + 2


def test_grouped_table_rejects_a_non_first_group_column() -> None:
    frame = pl.DataFrame({"spec": ["main"], "term": ["post_sample"]})
    with pytest.raises(ValueError, match="must be the first column"):
        reporting.grouped_table(frame, "spec", ["term", "spec"])


def test_grouped_table_validates_columns_and_headers() -> None:
    frame = pl.DataFrame({"spec": ["main"], "term": ["post_sample"]})
    with pytest.raises(ValueError, match="columns not in frame"):
        reporting.grouped_table(frame, "spec", ["spec", "coef"])
    with pytest.raises(ValueError, match="equal length"):
        reporting.grouped_table(frame, "spec", ["spec", "term"], ["Spec"])


def test_data_config_reads_the_pinned_sources() -> None:
    data = reporting.data_config()
    assert data["osap"]["vintage"] == 202510
    assert data["french"]["downloaded_on"]
    assert data["french"]["ff5_url"].startswith("https://")


def test_phase2_preregistration_missing_file_is_none(tmp_path: Path) -> None:
    assert reporting.phase2_preregistration(tmp_path / "prereg.yaml") is None


def test_phase2_preregistration_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "prereg.yaml"
    path.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        reporting.phase2_preregistration(path)


def test_phase2_preregistration_committed_file() -> None:
    prereg = reporting.phase2_preregistration()
    if prereg is None:
        pytest.skip("config/phase2_preregistration.yaml is not committed yet")
    assert prereg["bootstrap"]["reps"] == 2000
    assert prereg["bootstrap"]["seed"] == 2027
    assert prereg["multiple_testing"]["hurdle_t"] == 3.0
    assert len(prereg["multiple_testing"]["specs"]) == 3
    assert prereg["vintages"]["primary"] == 202510


@pytest.mark.parametrize("name", sorted(PHASE2_FRAMES))
def test_phase2_results_load_through_load_optional(name: str, tmp_path: Path) -> None:
    """The Phase 2 sections read their files through ``load_optional`` and the schemas."""
    assert reporting.load_optional(name, where=tmp_path) is None
    frame = pl.DataFrame(PHASE2_FRAMES[name])
    io_module.write_results(frame, name, tmp_path)
    loaded = reporting.load_optional(name, where=tmp_path)
    assert loaded is not None
    assert loaded.schema == RESULTS_SCHEMAS[name]
    assert loaded.height == 1


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


def test_report_contains_the_required_sections() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    for heading in SECTIONS:
        assert f"\n{heading}\n" in text, f"missing section: {heading}"


def test_report_prose_has_no_typed_decimal_results() -> None:
    """Result-like decimals must live in inline expressions, never in typed prose."""
    text = QMD_PATH.read_text(encoding="utf-8")
    prose = INLINE_CODE.sub("", INLINE_EXPRESSION.sub("", CODE_FENCE.sub("", text)))
    typed = DECIMAL_NUMBER.findall(prose)
    assert not typed, f"typed decimal numbers in report prose: {typed}"


def test_report_inline_expressions_are_backticked() -> None:
    """A bare ``{python}`` in the prose renders literally, so every one needs backticks."""
    text = CODE_FENCE.sub("", QMD_PATH.read_text(encoding="utf-8"))
    assert "{python}" in text
    leftover = BACKTICK_EXPRESSION.sub("", text)
    assert "{python}" not in leftover


def test_report_reads_results_only_through_reporting() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    assert "read_csv" not in text
    assert "read_parquet" not in text


def test_report_references_every_figure() -> None:
    text = QMD_PATH.read_text(encoding="utf-8")
    for name in ("event_time", "distribution", "coefplot"):
        assert f'"{name}"' in text, f"figure not referenced: {name}"


def test_report_puts_the_headline_result_first() -> None:
    """The main table and the event-time figure come before the details sections."""
    text = QMD_PATH.read_text(encoding="utf-8")
    question = text.index("# Question")
    headline = text.index("# Headline result")
    data = text.index("# Data")
    assert question < headline < data
    section = text[headline:data]
    assert "reporting.figure_block(" in section
    assert '"event_time"' in section
    assert '"n_predictors"' in section


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


def test_coverage_rows_on_results_fixture_are_readable_and_ordered() -> None:
    coverage = reporting.load("coverage.csv", where=FIXTURE_DIR)
    rows = reporting.coverage_rows(coverage)
    items = rows["item"].to_list()
    values = rows["value"].to_list()
    assert items[0] == "OSAP release"
    assert items[1] == "First return month"
    assert items[2] == "Last return month"
    assert values[1] == "January 1926"
    assert values[2] == "December 2024"
    assert "Predictors with original-paper returns" in items
    assert "Clear predictors" in items
    assert all("_" not in item for item in items)


def test_report_tables_set_column_widths_that_fit_the_text_block() -> None:
    """Every rendered table carries column widths; each set sums to the text block width."""
    text = QMD_PATH.read_text(encoding="utf-8")
    widths = re.findall(r"#\| tbl-colwidths: \[([0-9, ]+)\]", text)
    assert len(widths) == text.count("= render_table(\n")
    for specification in widths:
        assert sum(int(value) for value in specification.split(",")) == 100


MAKEFILE_TARGET = re.compile(r"^([A-Za-z0-9_.-]+):(?![:=])")
MAKEFILE_VARIABLE = re.compile(r"^([A-Za-z0-9_.-]+)\s*(?::=|\?=|=)\s*(.*)$")
MAKEFILE_REFERENCE = re.compile(r"\$\(([A-Za-z0-9_.-]+)\)")


def _makefile_recipes() -> dict[str, list[str]]:
    """Recipe lines of every Makefile rule, with backslash continuations joined."""
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    continued: list[str] | None = None
    for raw in MAKEFILE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if continued is not None:
            continued.append(line.strip())
            if line.endswith("\\"):
                continue
            assert current is not None
            joined = " ".join(continued).replace("\\", " ")
            recipes[current].append(" ".join(joined.split()))
            continued = None
            continue
        target = MAKEFILE_TARGET.match(line)
        if target:
            current = target.group(1)
            recipes.setdefault(current, [])
            continue
        if line.startswith("\t"):
            if current is not None:
                command = line.strip()
                if command.endswith("\\"):
                    continued = [command]
                else:
                    recipes[current].append(command)
            continue
        if line and not line.startswith("#"):
            current = None
    return recipes


def _makefile_variables() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in MAKEFILE_PATH.read_text(encoding="utf-8").splitlines():
        match = MAKEFILE_VARIABLE.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _expand_makefile(text: str, values: dict[str, str]) -> str:
    for _ in range(10):
        expanded = MAKEFILE_REFERENCE.sub(
            lambda match: values.get(match.group(1), match.group(0)), text
        )
        if expanded == text:
            break
        text = expanded
    return text


def test_fixture_site_directory_resolves_under_data_processed() -> None:
    """The fixture render directory the Makefile uses is data/processed/fixture/_site."""
    values = _makefile_variables()
    assert _expand_makefile("$(FIXTURE_SITE_DIR)", values) == "data/processed/fixture/_site"


def test_fixture_targets_render_into_their_own_output_directory() -> None:
    """Fixture renders pass an absolute --output-dir; none writes report/_site."""
    recipes = _makefile_recipes()
    for target in ("report-fixture", "report-placeholder", "reproduce-fixture"):
        assert target in recipes, f"missing Makefile target: {target}"
        renders = [cmd for cmd in recipes[target] if "quarto render report" in cmd]
        assert renders, f"{target} never renders the report"
        for command in renders:
            assert "--output-dir" in command, command
            assert "$(FIXTURE_SITE_DIR)" in command, command
            assert "report/_site" not in command, command


def test_report_target_is_the_only_one_writing_report_site() -> None:
    """Every report render except `make report` overrides the project output directory."""
    recipes = _makefile_recipes()
    for target, commands in recipes.items():
        for command in commands:
            if "quarto render report" not in command:
                continue
            if target == "report":
                assert "--output-dir" not in command, command
            else:
                assert "--output-dir" in command, f"{target}: {command}"
                assert "$(FIXTURE_SITE_DIR)" in command, f"{target}: {command}"


def test_report_project_still_writes_the_real_site_by_default() -> None:
    config = yaml.safe_load(QUARTO_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["project"]["output-dir"] == "_site"
