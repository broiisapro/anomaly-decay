"""Tests for the canonical CSV writer (byte-identical determinism)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.io import write_csv_canonical


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "signalname": ["Mom12m", "Size", "Accruals"],
            "yyyymm": pl.Series([196307, 196301, 199912], dtype=pl.Int32),
            "ret": [1.2345678901234567, -0.0, 4.567890123456789e-05],
        }
    )


def test_same_input_writes_byte_identical_output(tmp_path: Path) -> None:
    first = tmp_path / "a" / "out.csv"
    second = tmp_path / "b" / "out.csv"
    third = tmp_path / "c.csv"
    write_csv_canonical(_frame(), first, sort_by=["signalname", "yyyymm"])
    write_csv_canonical(_frame(), second, sort_by=["signalname", "yyyymm"])
    shuffled = _frame().sample(fraction=1.0, shuffle=True, seed=7)
    write_csv_canonical(shuffled, third, sort_by=["signalname", "yyyymm"])
    assert first.read_bytes() == second.read_bytes() == third.read_bytes()


def test_sorts_by_key_and_preserves_column_order(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_csv_canonical(_frame(), path, sort_by=["signalname", "yyyymm"])
    parsed = pl.read_csv(path)
    assert parsed.columns == ["signalname", "yyyymm", "ret"]
    assert parsed["signalname"].to_list() == ["Accruals", "Mom12m", "Size"]
    assert parsed["yyyymm"].to_list() == [199912, 196307, 196301]


def test_lf_endings_utf8_and_float_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    frame = _frame().with_columns(pl.lit("naïve-é").alias("journal"))
    write_csv_canonical(frame, path, sort_by=["signalname"])
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert "naïve-é" in raw.decode("utf-8")
    parsed = pl.read_csv(path)
    assert parsed["ret"].to_list() == frame.sort("signalname")["ret"].to_list()


def test_unknown_sort_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sort_by"):
        write_csv_canonical(_frame(), tmp_path / "out.csv", sort_by=["nope"])
