"""Tests for the predictor metadata table: mapping, supplement, validation, CLI."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.data.metadata import (
    MetadataError,
    MetadataWarning,
    build_metadata,
    main,
)
from anomaly_decay.schemas import PREDICTORS_SCHEMA, validate_frame

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_SIGNAL_DOC = (
    REPO_ROOT / "tests" / "fixtures" / "data" / "osap" / "202510" / "signal_doc.csv"
)
REAL_SIGNAL_DOC = REPO_ROOT / "data" / "raw" / "202510" / "signal_doc.csv"
SUPPLEMENT = REPO_ROOT / "config" / "predictor_dates_supplement.csv"

DOC_COLUMNS = (
    "Acronym",
    "Cat.Signal",
    "Predictability in OP",
    "SampleStartYear",
    "SampleEndYear",
    "Year",
    "Journal",
    "Authors",
    "Cat.Economic",
    "Cat.Data",
)
SUPPLEMENT_COLUMNS = ("signalname", "field", "value", "citation_key", "citation")


def _row(
    acronym: str,
    *,
    cat_signal: str = "Predictor",
    clear: str = "1_clear",
    start: str = "1963",
    end: str = "1990",
    year: str = "1992",
    journal: str = "JF",
    authors: str = "Author",
    cat_economic: str = "other",
    cat_data: str = "Price",
) -> dict[str, str]:
    return {
        "Acronym": acronym,
        "Cat.Signal": cat_signal,
        "Predictability in OP": clear,
        "SampleStartYear": start,
        "SampleEndYear": end,
        "Year": year,
        "Journal": journal,
        "Authors": authors,
        "Cat.Economic": cat_economic,
        "Cat.Data": cat_data,
    }


def _write_doc(path: Path, rows: list[dict[str, str]]) -> Path:
    frame = pl.DataFrame(rows, schema={column: pl.Utf8 for column in DOC_COLUMNS})
    frame.select(list(DOC_COLUMNS)).write_csv(path, line_terminator="\n")
    return path


def _write_supplement(path: Path, rows: list[dict[str, str]]) -> Path:
    frame = pl.DataFrame(rows, schema={column: pl.Utf8 for column in SUPPLEMENT_COLUMNS})
    frame.select(list(SUPPLEMENT_COLUMNS)).write_csv(path, line_terminator="\n")
    return path


# --- mapping -----------------------------------------------------------------


def test_mapping_on_fixture_signal_doc() -> None:
    frame = build_metadata(FIXTURE_SIGNAL_DOC)
    validate_frame(frame, PREDICTORS_SCHEMA, key=["signalname"])
    assert list(frame.columns) == list(PREDICTORS_SCHEMA)
    assert frame.height == 15
    assert frame["signalname"].to_list() == sorted(frame["signalname"].to_list())

    row = frame.filter(pl.col("signalname") == "Mom12m").row(0, named=True)
    assert row == {
        "signalname": "Mom12m",
        "sample_start_year": 1964,
        "sample_end_year": 1989,
        "pub_year": 1993,
        "journal": "JF",
        "authors": "Jegadeesh and Titman",
        "cat_economic": "momentum",
        "cat_data": "Price",
        "clear": True,
        "date_source": "signaldoc",
    }
    likely = frame.filter(pl.col("signalname") == "BetaFP").row(0, named=True)
    assert likely["clear"] is False
    assert likely["sample_start_year"] == 1929
    assert likely["sample_end_year"] == 2012
    assert likely["pub_year"] == 2014


def test_non_predictor_rows_are_excluded(tmp_path: Path) -> None:
    doc = _write_doc(
        tmp_path / "signal_doc.csv",
        [
            _row("KeepMe"),
            _row("PlaceboMe", cat_signal="Placebo"),
            _row("DropMe", cat_signal="Drop"),
        ],
    )
    frame = build_metadata(doc)
    assert frame["signalname"].to_list() == ["KeepMe"]


def test_signal_names_are_sorted(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("zebra"), _row("Alpha"), _row("beta")])
    frame = build_metadata(doc)
    assert frame["signalname"].to_list() == ["Alpha", "beta", "zebra"]


# --- supplement --------------------------------------------------------------


def test_supplement_fills_only_null_fields_and_records_citation(tmp_path: Path) -> None:
    doc = _write_doc(
        tmp_path / "signal_doc.csv",
        [
            _row("MissingStart", start=""),
            _row("MissingJournal", journal=""),
            _row("Complete"),
        ],
    )
    supplement = _write_supplement(
        tmp_path / "supplement.csv",
        [
            {
                "signalname": "MissingStart",
                "field": "sample_start_year",
                "value": "1965",
                "citation_key": "mp2016",
                "citation": "McLean and Pontiff (2016), Table 1.",
            },
            {
                "signalname": "MissingJournal",
                "field": "journal",
                "value": "JFE",
                "citation_key": "mp2016",
                "citation": "McLean and Pontiff (2016), Table 1.",
            },
            {
                "signalname": "Complete",
                "field": "sample_start_year",
                "value": "1900",
                "citation_key": "wrong",
                "citation": "This row must be ignored because the field is not null.",
            },
        ],
    )
    frame = build_metadata(doc, supplement)
    rows = {row["signalname"]: row for row in frame.iter_rows(named=True)}
    assert rows["MissingStart"]["sample_start_year"] == 1965
    assert rows["MissingStart"]["date_source"] == "supplement:mp2016"
    assert rows["MissingJournal"]["journal"] == "JFE"
    assert rows["MissingJournal"]["date_source"] == "supplement:mp2016"
    assert rows["Complete"]["sample_start_year"] == 1963
    assert rows["Complete"]["date_source"] == "signaldoc"


def test_supplement_multiple_keys_are_recorded(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Both", start="", journal="")])
    supplement = _write_supplement(
        tmp_path / "supplement.csv",
        [
            {
                "signalname": "Both",
                "field": "sample_start_year",
                "value": "1970",
                "citation_key": "b_key",
                "citation": "B.",
            },
            {
                "signalname": "Both",
                "field": "journal",
                "value": "RFS",
                "citation_key": "a_key",
                "citation": "A.",
            },
        ],
    )
    row = build_metadata(doc, supplement).row(0, named=True)
    assert row["date_source"] == "supplement:a_key+b_key"


@pytest.mark.parametrize(
    ("split", "message"),
    [
        ("citation", "citation"),
        ("citation_key", "citation"),
    ],
)
def test_supplement_without_citation_raises(tmp_path: Path, split: str, message: str) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Foo", start="")])
    entry = {
        "signalname": "Foo",
        "field": "sample_start_year",
        "value": "1965",
        "citation_key": "key",
        "citation": "Source (2016).",
    }
    entry[split] = ""
    supplement = _write_supplement(tmp_path / "supplement.csv", [entry])
    with pytest.raises(MetadataError, match=message) as info:
        build_metadata(doc, supplement)
    assert "citation" in str(info.value)


def test_supplement_unknown_signalname_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Known")])
    supplement = _write_supplement(
        tmp_path / "supplement.csv",
        [
            {
                "signalname": "Unknown",
                "field": "sample_start_year",
                "value": "1965",
                "citation_key": "k",
                "citation": "Source.",
            }
        ],
    )
    with pytest.raises(MetadataError, match="unknown predictor"):
        build_metadata(doc, supplement)


def test_supplement_unknown_field_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Known", start="")])
    supplement = _write_supplement(
        tmp_path / "supplement.csv",
        [
            {
                "signalname": "Known",
                "field": "clear",
                "value": "True",
                "citation_key": "k",
                "citation": "Source.",
            }
        ],
    )
    with pytest.raises(MetadataError, match="unknown field"):
        build_metadata(doc, supplement)


def test_supplement_duplicate_row_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Known", start="")])
    entry = {
        "signalname": "Known",
        "field": "sample_start_year",
        "value": "1965",
        "citation_key": "k",
        "citation": "Source.",
    }
    supplement = _write_supplement(tmp_path / "supplement.csv", [entry, entry])
    with pytest.raises(MetadataError, match="duplicate supplement row"):
        build_metadata(doc, supplement)


def test_committed_supplement_has_the_documented_header() -> None:
    first_line = SUPPLEMENT.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "signalname,field,value,citation_key,citation"


def test_empty_committed_supplement_is_accepted() -> None:
    frame = build_metadata(FIXTURE_SIGNAL_DOC, SUPPLEMENT)
    assert frame.height == 15


# --- validation --------------------------------------------------------------


def test_duplicate_signalname_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Same"), _row("Same")])
    with pytest.raises(MetadataError, match="more than once"):
        build_metadata(doc)


def test_missing_required_field_after_supplement_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Gap", year="")])
    with pytest.raises(MetadataError, match="missing pub_year"):
        build_metadata(doc)


def test_sample_start_after_sample_end_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Backwards", start="1991", end="1990")])
    with pytest.raises(MetadataError, match="sample_start_year 1991 > sample_end_year 1990"):
        build_metadata(doc)


def test_pub_year_before_sample_end_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Early", end="1995", year="1990")])
    with pytest.raises(MetadataError, match="pub_year 1990 < sample_end_year 1995"):
        build_metadata(doc)


def test_pub_year_equal_sample_end_warns(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("SameEnd", end="1990", year="1990")])
    with pytest.warns(MetadataWarning, match="SameEnd"):
        frame = build_metadata(doc)
    assert frame.height == 1


def test_non_integer_year_raises(tmp_path: Path) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Float", start="1963.0")])
    with pytest.raises(MetadataError, match="not an integer year"):
        build_metadata(doc)


def test_missing_signal_doc_raises(tmp_path: Path) -> None:
    with pytest.raises(MetadataError, match="not found"):
        build_metadata(tmp_path / "nope.csv")


def test_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "signal_doc.csv"
    pl.DataFrame({"Acronym": ["A"], "Cat.Signal": ["Predictor"]}).write_csv(
        path, line_terminator="\n"
    )
    with pytest.raises(MetadataError, match="missing columns"):
        build_metadata(path)


# --- real signal doc ---------------------------------------------------------

realdata_only = pytest.mark.skipif(
    not REAL_SIGNAL_DOC.is_file(), reason="OSAP cache not present under data/raw/202510"
)


@realdata_only
def test_real_signal_doc_counts_and_schema() -> None:
    frame = build_metadata(REAL_SIGNAL_DOC, SUPPLEMENT)
    validate_frame(frame, PREDICTORS_SCHEMA, key=["signalname"])
    assert frame.height == 212
    assert int(frame["clear"].sum()) == 165
    assert frame.filter(pl.col("pub_year") <= 2013).height == 198
    assert frame["sample_start_year"].null_count() == 0
    assert frame["sample_end_year"].null_count() == 0
    assert frame["pub_year"].null_count() == 0
    assert set(frame["date_source"].unique().to_list()) == {"signaldoc"}


# --- CLI ---------------------------------------------------------------------


def test_cli_writes_parquet_and_prints_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "processed" / "predictors.parquet"
    exit_code = main(
        [
            "--signal-doc",
            str(FIXTURE_SIGNAL_DOC),
            "--supplement",
            str(SUPPLEMENT),
            "--out",
            str(out),
        ]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out.splitlines()
    assert "predictors: 15" in printed
    assert "clear: 12" in printed  # Beta, BetaFP and dVolPut are 2_likely
    assert "published by 2013: 12" in printed  # BetaFP (2014), dVolPut (2014), TrendFactor (2016)
    frame = pl.read_parquet(out)
    validate_frame(frame, PREDICTORS_SCHEMA, key=["signalname"])


def test_cli_reports_invalid_signal_doc(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    doc = _write_doc(tmp_path / "signal_doc.csv", [_row("Backwards", start="1991", end="1990")])
    exit_code = main(["--signal-doc", str(doc), "--out", str(tmp_path / "out.parquet")])
    assert exit_code == 1
    assert "metadata failed:" in capsys.readouterr().err
