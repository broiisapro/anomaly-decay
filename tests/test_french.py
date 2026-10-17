"""Tests for the Ken French parser, against the real (small) source zips."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.data.french import (
    FF5_RENAME,
    UMD_RENAME,
    FrenchParseError,
    parse_french_csv,
    parse_french_zip,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "french"


def _zip_with(text: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


def test_ff5_monthly_block_only() -> None:
    df = parse_french_zip((FIXTURE_DIR / "ff5.zip").read_bytes(), FF5_RENAME)
    assert df.columns == ["yyyymm", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert df.schema == {
        "yyyymm": pl.Int32,
        "mkt_rf": pl.Float64,
        "smb": pl.Float64,
        "hml": pl.Float64,
        "rmw": pl.Float64,
        "cma": pl.Float64,
        "rf": pl.Float64,
    }
    assert df["yyyymm"].min() == 196307
    assert df["yyyymm"].max() == 202608
    assert df.height == 758  # 196307..202608 inclusive; the annual block is excluded


def test_umd_monthly_block_only() -> None:
    df = parse_french_zip((FIXTURE_DIR / "umd.zip").read_bytes(), UMD_RENAME)
    assert df.columns == ["yyyymm", "umd"]
    assert df["yyyymm"].min() == 192701
    assert df["yyyymm"].max() == 202608
    assert df.height == 1196


def test_known_value_ff5_first_month_is_percent() -> None:
    df = parse_french_zip((FIXTURE_DIR / "ff5.zip").read_bytes(), FF5_RENAME)
    row = df.filter(df["yyyymm"] == 196307).row(0, named=True)
    assert row == {
        "yyyymm": 196307,
        "mkt_rf": -0.39,
        "smb": -0.48,
        "hml": -0.84,
        "rmw": 0.64,
        "cma": -1.15,
        "rf": 0.27,
    }


def test_known_value_umd_first_month() -> None:
    df = parse_french_zip((FIXTURE_DIR / "umd.zip").read_bytes(), UMD_RENAME)
    row = df.filter(df["yyyymm"] == 192701).row(0, named=True)
    assert row == {"yyyymm": 192701, "umd": 0.57}


def test_real_files_contain_no_missing_codes_after_parsing() -> None:
    ff5 = parse_french_zip((FIXTURE_DIR / "ff5.zip").read_bytes(), FF5_RENAME)
    umd = parse_french_zip((FIXTURE_DIR / "umd.zip").read_bytes(), UMD_RENAME)
    for df in (ff5, umd):
        factors = [column for column in df.columns if column != "yyyymm"]
        assert df.select(pl.all_horizontal(pl.col(factors) > -99.98).all()).row(0) == (True,)


def test_annual_block_is_excluded() -> None:
    text = (
        "note line\r\n"
        "\r\n"
        ",Mom\r\n"
        "192701,   0.57\r\n"
        "192702,  -1.51\r\n"
        "\r\n"
        "Annual Factors:\r\n"
        "January-December\r\n"
        ",Mom\r\n"
        "  1927,  24.52\r\n"
        "  1928,  26.43\r\n"
        "\r\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
    )
    df = parse_french_csv(text, UMD_RENAME)
    assert df["yyyymm"].to_list() == [192701, 192702]
    assert df["umd"].to_list() == [0.57, -1.51]


def test_missing_codes_become_null() -> None:
    text = ",Mom\n192701,   0.57\n192702, -99.99\n192703, -999\n192704,   1.25\n"
    df = parse_french_csv(text, UMD_RENAME)
    assert df["umd"].to_list() == [0.57, None, None, 1.25]


def test_crlf_and_padding_are_stripped() -> None:
    text = ",Mkt-RF,RF\r\n196307,   -0.39,    0.27\r\n"
    df = parse_french_csv(text, FF5_RENAME)
    assert df.row(0, named=True) == {"yyyymm": 196307, "mkt_rf": -0.39, "rf": 0.27}


def test_missing_header_raises() -> None:
    with pytest.raises(FrenchParseError, match="header"):
        parse_french_csv("no,comma,header\n196307,1\n", FF5_RENAME)


def test_unexpected_factor_column_raises() -> None:
    with pytest.raises(FrenchParseError, match="unexpected"):
        parse_french_csv(",Mkt-RF,Bogus\n196307,-0.39,1\n", FF5_RENAME)


def test_not_a_zip_raises() -> None:
    with pytest.raises(FrenchParseError, match="not a zip"):
        parse_french_zip(b"<html>quota exceeded</html>", FF5_RENAME)


def test_zip_without_csv_member_raises() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(FrenchParseError, match="no CSV"):
        parse_french_zip(buffer.getvalue(), FF5_RENAME)
