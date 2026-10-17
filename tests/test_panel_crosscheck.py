"""Exact cross-check: production panel build vs the independent reference fixture.

``tests/fixtures/panel_fixture.parquet`` comes from ``tests/fixtures/build_fixture_panel.py``,
a deliberately simple second implementation of the window rules and panel layout that
imports nothing from ``anomaly_decay`` (architecture §4.12). This module rebuilds the
same panel with production code and asserts exact equality — values, dtypes, column
order and row order — so a silent off-by-one in either implementation fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from anomaly_decay.panel.build import build_panel
from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, PREDICTORS_SCHEMA, validate_frame
from polars.testing import assert_frame_equal

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_DATA = FIXTURES / "data"
FIXTURE_OSAP = FIXTURE_DATA / "osap"
FIXTURE_PINNED = FIXTURE_DATA / "pinned"
REFERENCE_PANEL = FIXTURES / "panel_fixture.parquet"
VINTAGE = 202510
PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")


def _metadata_from_signal_doc(signal_doc_path: Path) -> pl.DataFrame:
    """Equivalent local metadata helper used until agent A's #11 lands.

    Reads the canonical signal doc, keeps ``Cat.Signal == "Predictor"`` and maps the
    columns to :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`.
    """
    doc = pl.read_csv(signal_doc_path, infer_schema_length=500).filter(
        pl.col("Cat.Signal") == "Predictor"
    )
    height = doc.height
    return pl.DataFrame(
        {
            "signalname": doc["Acronym"],
            "sample_start_year": doc["SampleStartYear"].cast(pl.Int16),
            "sample_end_year": doc["SampleEndYear"].cast(pl.Int16),
            "pub_year": doc["Year"].cast(pl.Int16),
            "journal": doc["Journal"],
            "authors": doc["Authors"],
            "cat_economic": doc["Cat.Economic"],
            "cat_data": doc["Cat.Data"],
            "clear": doc["Predictability in OP"] == "1_clear",
            "date_source": ["signaldoc"] * height,
        }
    ).select(list(PREDICTORS_SCHEMA))


def _fixture_metadata() -> pl.DataFrame:
    """Prefer the production metadata builder once #11 is merged; fall back locally."""
    try:
        from anomaly_decay.data.metadata import build_metadata

        return build_metadata(FIXTURE_OSAP / str(VINTAGE) / "signal_doc.csv")
    except NotImplementedError:
        return _metadata_from_signal_doc(FIXTURE_OSAP / str(VINTAGE) / "signal_doc.csv")


@pytest.fixture(scope="module")
def reference_panel() -> pl.DataFrame:
    panel = pl.read_parquet(REFERENCE_PANEL)
    validate_frame(panel, PANEL_SCHEMA, key=PANEL_KEY)
    return panel


def test_production_panel_reproduces_reference_panel_exactly(
    reference_panel: pl.DataFrame,
) -> None:
    production = build_panel(
        FIXTURE_OSAP,
        _fixture_metadata(),
        PORTFOLIOS,
        pinned_dir=FIXTURE_PINNED,
        vintage=VINTAGE,
    )
    # Exact equality including floats (check_exact): every boundary month, event time,
    # in-sample mean and factor join must agree bit for bit.
    assert_frame_equal(production, reference_panel, check_exact=True)
    assert production.height == reference_panel.height
    assert production.columns == list(PANEL_SCHEMA)


def test_reference_panel_covers_every_fixture_portfolio(
    reference_panel: pl.DataFrame,
) -> None:
    assert set(reference_panel["portfolio"].unique().to_list()) == set(PORTFOLIOS)
    grouped = reference_panel.group_by("portfolio").agg(pl.col("signalname").n_unique())
    assert grouped["signalname"].min() > 0
    assert reference_panel["window"].null_count() == 0
    assert set(reference_panel["window_alt"].unique().to_list()) <= {
        "in_sample",
        "post_sample",
        "post_publication",
    }
