"""Tests for the schema contracts (docs/architecture.md §5.1-§5.3)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.schemas import (
    ALLOWED_WINDOWS,
    PANEL_KEY,
    PANEL_SCHEMA,
    PREDICTORS_SCHEMA,
    RESULTS_SCHEMAS,
    SchemaError,
    validate_frame,
)

REPO_ROOT = Path(__file__).parents[1]
PANEL_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "panel_fixture.parquet"


def _panel() -> pl.DataFrame:
    return pl.read_parquet(PANEL_FIXTURE)


def _valid_results_frame(name: str) -> pl.DataFrame:
    schema = RESULTS_SCHEMAS[name]
    row = {}
    for column, dtype in schema.items():
        if dtype == pl.Utf8:
            row[column] = "in_sample" if column == "window" else "x"
        elif dtype == pl.Boolean:
            row[column] = False
        elif dtype in (pl.Int64, pl.Int8, pl.Int32, pl.Int16):
            row[column] = 1
        else:
            row[column] = 0.5
    return pl.DataFrame([row], schema=schema)


def test_results_schemas_cover_every_documented_file() -> None:
    assert set(RESULTS_SCHEMAS) == {
        "main.csv",
        "decay.csv",
        "robustness.csv",
        "extension.csv",
        "specs_run.csv",
        "predictor_summary.csv",
        "event_time.csv",
        "coverage.csv",
    }
    for name, schema in RESULTS_SCHEMAS.items():
        validate_frame(_valid_results_frame(name), schema)


def test_predictors_schema_validates_expected_frame() -> None:
    frame = pl.DataFrame(
        {
            "signalname": ["Size"],
            "sample_start_year": pl.Series([1926], dtype=pl.Int16),
            "sample_end_year": pl.Series([1975], dtype=pl.Int16),
            "pub_year": pl.Series([1981], dtype=pl.Int16),
            "journal": ["JFE"],
            "authors": ["Banz"],
            "cat_economic": ["size"],
            "cat_data": ["Price"],
            "clear": [True],
            "date_source": ["signaldoc"],
        }
    )
    validate_frame(frame, PREDICTORS_SCHEMA, key=["signalname"])


def test_fixture_panel_validates_against_panel_schema() -> None:
    panel = _panel()
    validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)
    assert panel.height > 0
    assert panel["signalname"].n_unique() == 15
    assert set(panel["portfolio"].unique().to_list()) == {"op", "deciles_ew", "deciles_vw"}


def test_every_panel_row_is_in_exactly_one_window() -> None:
    panel = _panel()
    flags = panel.select(
        pl.col("post_sample")
        + pl.col("post_publication")
        + (pl.col("window") == "in_sample").cast(pl.Int8)
    ).to_series()
    assert flags.min() == 1
    assert flags.max() == 1
    assert set(panel["window"].unique().to_list()) <= set(ALLOWED_WINDOWS)
    assert set(panel["window_alt"].unique().to_list()) <= set(ALLOWED_WINDOWS)
    for convention, window, post_sample, post_publication in (
        ("default", "window", "post_sample", "post_publication"),
        ("alt", "window_alt", "post_sample_alt", "post_publication_alt"),
    ):
        flags = panel.select(
            pl.col(post_sample)
            + pl.col(post_publication)
            + (pl.col(window) == "in_sample").cast(pl.Int8)
        ).to_series()
        assert flags.min() == 1, convention
        assert flags.max() == 1, convention


def test_unsorted_frame_is_rejected() -> None:
    panel = _panel().head(10).sort("yyyymm", descending=True)
    with pytest.raises(SchemaError, match="sorted"):
        validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)


def test_duplicate_key_is_rejected() -> None:
    panel = _panel()
    duplicated = pl.concat([panel.head(1), panel.head(1), panel.tail(1)])
    with pytest.raises(SchemaError, match="not unique"):
        validate_frame(duplicated, PANEL_SCHEMA, key=PANEL_KEY)


def test_null_key_is_rejected() -> None:
    panel = _panel().head(5).with_columns(pl.lit(None).cast(pl.Utf8).alias("signalname"))
    with pytest.raises(SchemaError, match="null"):
        validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)


def test_wrong_dtype_is_rejected() -> None:
    panel = _panel().head(5).with_columns(pl.col("ret").cast(pl.Float32))
    with pytest.raises(SchemaError, match="expected dtype"):
        validate_frame(panel, PANEL_SCHEMA)


def test_wrong_column_order_is_rejected() -> None:
    columns = list(PANEL_SCHEMA)
    reordered = [columns[1], columns[0], *columns[2:]]
    panel = _panel().select(reordered)
    with pytest.raises(SchemaError, match="columns differ"):
        validate_frame(panel, PANEL_SCHEMA)


def test_missing_column_is_rejected() -> None:
    panel = _panel().drop("ret_norm")
    with pytest.raises(SchemaError, match="columns differ"):
        validate_frame(panel, PANEL_SCHEMA)


def test_invalid_window_value_is_rejected() -> None:
    panel = _panel().head(5).with_columns(pl.lit("pre_sample").alias("window"))
    with pytest.raises(SchemaError, match="values outside"):
        validate_frame(panel, PANEL_SCHEMA)
