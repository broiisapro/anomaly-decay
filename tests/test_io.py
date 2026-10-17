"""Tests for the canonical CSV writer and the results I/O layer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.io import (
    RESULTS_SORT_KEYS,
    read_results,
    write_csv_canonical,
    write_results,
)
from anomaly_decay.schemas import RESULTS_SCHEMAS, SchemaError

RESULTS_SORT_KEY_ORDERS: dict[str, list[list[object]]] = {
    "main.csv": [["b_spec", "pb"], ["a_spec", "ps"]],
    "decay.csv": [["decay", "pooled_ratio", "ps"], ["decay", "normalized", "pp"]],
    "robustness.csv": [["main_twoway", "pb"], ["main_twoway", "ps"]],
    "extension.csv": [["full_extension", "pb"], ["full_extension", "ps"]],
    "specs_run.csv": [["main_twoway"], ["main"]],
    "predictor_summary.csv": [["Mom12m", "post_sample"], ["Mom12m", "in_sample"]],
    "event_time.csv": [["publication", 1], ["publication", 0]],
    "coverage.csv": [["z_item"], ["a_item"]],
}


def _results_frame(name: str) -> pl.DataFrame:
    """A valid, multi-row, deliberately unsorted frame for results file *name*."""
    schema = RESULTS_SCHEMAS[name]
    rows = []
    for index, key_values in enumerate(RESULTS_SORT_KEY_ORDERS[name]):
        row: dict[str, object] = {}
        for column, dtype in schema.items():
            if column == "window":
                row[column] = "in_sample"
            elif dtype == pl.Utf8:
                if name == "coverage.csv" and column == "value":
                    row[column] = f"value-{index}"
                else:
                    row[column] = f"x{index}"
            elif dtype == pl.Boolean:
                row[column] = False
            elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                row[column] = 1
            else:
                row[column] = 0.1 + 1 / 3 * (index + 1)
        for column, value in zip(RESULTS_SORT_KEYS[name], key_values, strict=True):
            row[column] = value
        rows.append(row)
    return pl.DataFrame(rows, schema=schema)


def test_results_sort_keys_cover_every_file_and_column() -> None:
    assert set(RESULTS_SORT_KEYS) == set(RESULTS_SCHEMAS)
    for name, keys in RESULTS_SORT_KEYS.items():
        assert keys
        for column in keys:
            assert column in RESULTS_SCHEMAS[name], (name, column)


@pytest.mark.parametrize("name", sorted(RESULTS_SCHEMAS))
def test_results_round_trip_preserves_values_and_dtypes(name: str, tmp_path: Path) -> None:
    frame = _results_frame(name)
    write_results(frame, name, tmp_path)
    back = read_results(name, tmp_path)
    assert back.schema == frame.schema
    key = list(RESULTS_SORT_KEYS[name])
    expected = frame.sort(key)
    assert back.equals(expected)


@pytest.mark.parametrize("name", sorted(RESULTS_SCHEMAS))
def test_results_rewrite_is_byte_identical(name: str, tmp_path: Path) -> None:
    frame = _results_frame(name)
    first = write_results(frame, name, tmp_path / "first")
    second = write_results(frame, name, tmp_path / "second")
    assert first.read_bytes() == second.read_bytes()
    reshuffled = frame.reverse()
    third = write_results(reshuffled, name, tmp_path / "third")
    assert first.read_bytes() == third.read_bytes()
    again = write_results(read_results(name, tmp_path / "first"), name, tmp_path / "fourth")
    assert first.read_bytes() == again.read_bytes()
    assert b"\r" not in first.read_bytes()


@pytest.mark.parametrize("name", sorted(RESULTS_SCHEMAS))
def test_results_columns_written_in_schema_order(name: str, tmp_path: Path) -> None:
    schema = RESULTS_SCHEMAS[name]
    frame = _results_frame(name)
    reordered = frame.select(list(schema)[::-1])
    path = write_results(reordered, name, tmp_path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(schema)


@pytest.mark.parametrize("name", sorted(RESULTS_SCHEMAS))
def test_write_results_rejects_wrong_dtype(name: str, tmp_path: Path) -> None:
    schema = RESULTS_SCHEMAS[name]
    frame = _results_frame(name)
    column = next((c for c, dtype in schema.items() if dtype == pl.Float64), next(iter(schema)))
    corrupted = {name_: frame[name_].to_list() for name_ in frame.columns}
    if schema[column] == pl.Float64:
        wrong_dtype, wrong_values = pl.Utf8, ["not-a-number"] * frame.height
    else:
        wrong_dtype, wrong_values = pl.Int64, [1] * frame.height
    corrupted[column] = wrong_values
    misdtyped = pl.DataFrame(corrupted, schema={**frame.schema, column: wrong_dtype})
    with pytest.raises(SchemaError, match="expected dtype"):
        write_results(misdtyped, name, tmp_path)


@pytest.mark.parametrize("name", sorted(RESULTS_SCHEMAS))
def test_write_results_rejects_missing_column(name: str, tmp_path: Path) -> None:
    frame = _results_frame(name)
    with pytest.raises(SchemaError, match="missing columns"):
        write_results(frame.drop(frame.columns[-1]), name, tmp_path)


def test_results_rejects_unknown_name(tmp_path: Path) -> None:
    frame = _results_frame("main.csv")
    with pytest.raises(SchemaError, match="unknown results file"):
        write_results(frame, "nope.csv", tmp_path)
    with pytest.raises(SchemaError, match="unknown results file"):
        read_results("nope.csv", tmp_path)


def test_read_results_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_results("main.csv", tmp_path)


def test_result_text_columns_survive_zero_padding(tmp_path: Path) -> None:
    frame = _results_frame("coverage.csv").with_columns(pl.lit("01").alias("value"))
    path = write_results(frame, "coverage.csv", tmp_path)
    assert "01" in path.read_text(encoding="utf-8")
    back = read_results("coverage.csv", tmp_path)
    assert back.schema["value"] == pl.Utf8
    assert back["value"].to_list() == ["01", "01"]


def test_read_results_validates_dtypes(tmp_path: Path) -> None:
    (tmp_path / "event_time.csv").write_text(
        "anchor,event_month,mean_ret,se,n_predictors\npublication,0.5,1.0,0.2,10\n",
        encoding="utf-8",
    )
    with pytest.raises(SchemaError, match="expected dtype"):
        read_results("event_time.csv", tmp_path)


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
