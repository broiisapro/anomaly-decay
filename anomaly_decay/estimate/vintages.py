"""Vintage comparison: how much do OSAP data revisions move the decay estimates?

Phase 2 (pre-registered in ``config/phase2_preregistration.yaml``, ``vintages:``): for
each requested OSAP release build that release's own predictor metadata from its
canonical signal doc and its own ``op`` panel, keep only the predictors present in
**every** vintage (the intersection), and run the pre-registered ``main`` and
``full_extension`` specifications on it. The output rows per
``RESULTS_SCHEMAS["vintage_comparison.csv"]`` carry the coefficients, decay percentages
and sample counts per (spec, vintage, term) with ``predictor_set = "intersection"``.

:func:`metadata_date_changes` reports how many predictors' calendar years changed
between the primary release and each comparison release — the "revision" part of the
comparison that the coefficients alone cannot show.

CLI::

    python -m anomaly_decay.estimate.vintages --vintages 202410,202510
        [--data-dir data/raw] [--pinned-dir data/pinned]
        [--specs config/specifications.yaml] [--results-dir results]
        [--config config/data.toml] [--supplement config/predictor_dates_supplement.csv]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from anomaly_decay.config import Spec, load_data_config, load_specifications
from anomaly_decay.data.fetch import load_compare_vintages
from anomaly_decay.data.metadata import MetadataError, build_metadata
from anomaly_decay.estimate.main import EstimationError, decay_table, run_spec
from anomaly_decay.io import write_results
from anomaly_decay.panel.build import build_panel

__all__ = [
    "VINTAGE_SPEC_IDS",
    "VintageComparison",
    "build_vintage_comparison",
    "main",
    "metadata_date_changes",
]

VINTAGE_SPEC_IDS: tuple[str, ...] = ("main", "full_extension")
"""Specification ids the vintage comparison runs (pre-registered ``vintages.specs``)."""

PREDICTOR_SET = "intersection"
"""``predictor_set`` value for every row (pre-registered ``vintages.predictor_set``)."""

_DATE_FIELDS: tuple[str, ...] = ("sample_start_year", "sample_end_year", "pub_year")


class VintageComparisonError(ValueError):
    """Raised when the vintage comparison cannot be computed as pre-registered."""


@dataclass(frozen=True)
class VintageComparison:
    """Everything a comparison run produced.

    ``results`` is the frame written to ``vintage_comparison.csv``;
    ``date_changes`` lists one row per (vintage, signalname, field) whose calendar
    year differs from the primary release among the intersection predictors.
    """

    results: pl.DataFrame
    intersection: tuple[str, ...]
    vintages: tuple[int, ...]
    date_changes: pl.DataFrame


def metadata_date_changes(
    metadata: Mapping[int, pl.DataFrame],
    *,
    primary: int,
    predictors: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Predictors whose sample or publication years differ from the primary release.

    *metadata* maps each vintage to its predictor table (per
    :data:`anomaly_decay.schemas.PREDICTORS_SCHEMA`); *primary* must be one of the
    vintages. *predictors* restricts the comparison (the intersection when called from
    :func:`build_vintage_comparison`). Output: one row per
    (``vintage``, ``signalname``, ``field``) difference with ``field`` in
    ``sample_start_year | sample_end_year | pub_year``, sorted by key; empty when the
    primary table is the only input or nothing changed.
    """
    schema = {
        "vintage": pl.Int64,
        "signalname": pl.Utf8,
        "field": pl.Utf8,
        "primary_value": pl.Int64,
        "vintage_value": pl.Int64,
    }
    if primary not in metadata:
        raise VintageComparisonError(f"primary vintage {primary} has no metadata table")
    if len(metadata) < 2:
        return pl.DataFrame(schema=schema)
    reference = metadata[primary].filter(
        pl.col("signalname").is_in(list(predictors)) if predictors is not None else pl.lit(True)
    )
    rows: list[dict[str, object]] = []
    for vintage in sorted(metadata):
        if vintage == primary:
            continue
        other = metadata[vintage]
        joined = reference.select("signalname", *_DATE_FIELDS).join(
            other.select("signalname", *_DATE_FIELDS),
            on="signalname",
            how="inner",
            suffix="_other",
        )
        for field in _DATE_FIELDS:
            changed = joined.filter(pl.col(field).ne_missing(pl.col(f"{field}_other")))
            rows.extend(
                {
                    "vintage": int(vintage),
                    "signalname": row["signalname"],
                    "field": field,
                    "primary_value": row[field],
                    "vintage_value": row[f"{field}_other"],
                }
                for row in changed.iter_rows(named=True)
            )
    return pl.DataFrame(rows, schema=schema).sort(["vintage", "signalname", "field"])


def _select_specs(specs: Sequence[Spec]) -> list[Spec]:
    by_id = {spec.id: spec for spec in specs}
    missing = [spec_id for spec_id in VINTAGE_SPEC_IDS if spec_id not in by_id]
    if missing:
        raise VintageComparisonError(
            f"specifications.yaml is missing the vintage-comparison specs {missing}"
        )
    return [by_id[spec_id] for spec_id in VINTAGE_SPEC_IDS]


def _intersection(panels: Mapping[int, pl.DataFrame]) -> tuple[str, ...]:
    sets = [set(panel["signalname"].unique().to_list()) for panel in panels.values()]
    if not sets:
        raise VintageComparisonError("no panels to intersect")
    common = set.intersection(*sets)
    if not common:
        raise VintageComparisonError(
            "the vintages share no predictors with op long-short rows; nothing to compare"
        )
    return tuple(sorted(common))


def _comparison_rows(spec: Spec, vintage: int, panel: pl.DataFrame) -> pl.DataFrame:
    """Coefficient and decay rows for one (spec, vintage) with the output columns."""
    coefs = run_spec(panel, spec)
    decay = decay_table(panel, spec, coefs)
    joined = coefs.join(decay.select("term", "decay_pct"), on="term", how="left")
    return joined.select(
        pl.lit(spec.id).alias("spec"),
        pl.lit(int(vintage), dtype=pl.Int64).alias("vintage"),
        pl.col("term"),
        pl.col("coef"),
        pl.col("se"),
        pl.col("decay_pct"),
        pl.col("n_predictors"),
        pl.col("n_obs"),
        pl.lit(PREDICTOR_SET).alias("predictor_set"),
    )


def build_vintage_comparison(
    data_dir: Path,
    vintages: Sequence[int],
    *,
    pinned_dir: Path,
    specs: Sequence[Spec],
    supplement_path: Path | None = None,
    results_dir: Path | None = None,
    primary: int | None = None,
) -> VintageComparison:
    """Run the pre-registered vintage comparison and optionally write the results file.

    Each vintage is built from its own canonical cache under ``data_dir/<vintage>/``
    (signal doc through :func:`~anomaly_decay.data.metadata.build_metadata`, ``op``
    panel through :func:`~anomaly_decay.panel.build.build_panel`); only the predictors
    present in every vintage are kept. ``main`` and ``full_extension`` are estimated on
    that intersection with :func:`~anomaly_decay.estimate.main.run_spec` and the pooled
    in-sample-mean decay comes from
    :func:`~anomaly_decay.estimate.main.decay_table`. With *results_dir*, the frame is
    written as ``vintage_comparison.csv`` (sorted by spec, vintage, term). *primary*
    names the release the reported metadata date changes are relative to; it defaults
    to the latest requested vintage.

    Raises :class:`VintageComparisonError` for fewer than two vintages, a missing
    specification, or an empty intersection; estimation problems surface as
    :class:`~anomaly_decay.estimate.main.EstimationError`.
    """
    ordered = tuple(dict.fromkeys(int(vintage) for vintage in vintages))
    if len(ordered) < 2:
        raise VintageComparisonError("the comparison needs at least two vintages")
    reference = max(ordered) if primary is None else int(primary)
    if reference not in ordered:
        raise VintageComparisonError(f"primary vintage {reference} is not in {list(ordered)}")
    selected = _select_specs(specs)

    metadata: dict[int, pl.DataFrame] = {}
    panels: dict[int, pl.DataFrame] = {}
    for vintage in ordered:
        signal_doc = data_dir / str(vintage) / "signal_doc.csv"
        port_op = data_dir / str(vintage) / "port_op.csv"
        for path in (signal_doc, port_op):
            if not path.is_file():
                raise VintageComparisonError(f"canonical file not found: {path}")
        metadata[vintage] = build_metadata(signal_doc, supplement_path)
        panels[vintage] = build_panel(
            data_dir, metadata[vintage], ("op",), pinned_dir=pinned_dir, vintage=vintage
        )

    intersection = _intersection(panels)
    keep = list(intersection)
    restricted = {
        vintage: panel.filter(pl.col("signalname").is_in(keep)) for vintage, panel in panels.items()
    }

    parts: list[pl.DataFrame] = []
    for vintage in ordered:
        for spec in selected:
            parts.append(_comparison_rows(spec, vintage, restricted[vintage]))
    results = pl.concat(parts).sort(["spec", "vintage", "term"])
    if results.height == 0:
        raise VintageComparisonError("no comparison rows were produced")

    date_changes = metadata_date_changes(metadata, primary=reference, predictors=keep)

    if results_dir is not None:
        write_results(results, "vintage_comparison.csv", results_dir)
    return VintageComparison(
        results=results,
        intersection=intersection,
        vintages=ordered,
        date_changes=date_changes,
    )


def _print_summary(comparison: VintageComparison, primary: int) -> None:
    print(
        f"vintage comparison: vintages={list(comparison.vintages)}, "
        f"intersection={len(comparison.intersection)} predictors"
    )
    changes = comparison.date_changes
    for vintage in comparison.vintages:
        if vintage == primary:
            continue
        rows = changes.filter(pl.col("vintage") == vintage)
        predictors = rows["signalname"].n_unique()
        print(f"  metadata date changes {primary} -> {vintage}: {predictors} predictor(s)")
        for field in _DATE_FIELDS:
            count = rows.filter(pl.col("field") == field).height
            print(f"    {field}: {count}")
        examples = rows.head(3).iter_rows(named=True)
        for row in examples:
            print(
                f"    e.g. {row['signalname']}: {row['field']} "
                f"{row['primary_value']} -> {row['vintage_value']}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the vintage comparison and write ``vintage_comparison.csv``.

    ``--vintages`` must be a subset of the pinned ``[osap] vintage`` plus
    ``[osap] compare_vintages`` from ``config/data.toml`` and list at least two
    releases; the written comparison covers exactly the requested releases.
    """
    parser = argparse.ArgumentParser(
        prog="python -m anomaly_decay.estimate.vintages",
        description="Compare main and full-extension decay across OSAP releases.",
    )
    parser.add_argument(
        "--vintages",
        required=True,
        help="comma-separated releases, e.g. 202410,202510 (at least two)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--pinned-dir", type=Path, default=Path("data/pinned"))
    parser.add_argument("--specs", type=Path, default=Path("config/specifications.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("config/data.toml"))
    parser.add_argument(
        "--supplement",
        type=Path,
        default=Path("config/predictor_dates_supplement.csv"),
    )
    args = parser.parse_args(argv)

    try:
        vintages = tuple(int(part) for part in args.vintages.split(",") if part)
    except ValueError:
        parser.error("--vintages must be a comma-separated list of release years")
    if len(vintages) < 2:
        parser.error("--vintages needs at least two releases")

    cfg = load_data_config(args.config)
    allowed = {cfg.osap.vintage, *load_compare_vintages(args.config)}
    unknown = [vintage for vintage in vintages if vintage not in allowed]
    if unknown:
        parser.error(
            f"vintages {unknown} are outside the pinned vintage {cfg.osap.vintage} "
            f"and compare_vintages {sorted(allowed - {cfg.osap.vintage})}"
        )

    supplement = args.supplement if args.supplement.is_file() else None
    primary = cfg.osap.vintage if cfg.osap.vintage in vintages else None
    try:
        comparison = build_vintage_comparison(
            args.data_dir,
            vintages,
            pinned_dir=args.pinned_dir,
            specs=load_specifications(args.specs),
            supplement_path=supplement,
            results_dir=args.results_dir,
            primary=primary,
        )
    except (VintageComparisonError, EstimationError, MetadataError) as exc:
        print(f"vintage comparison failed: {exc}", file=sys.stderr)
        return 1
    _print_summary(comparison, primary if primary is not None else max(vintages))
    print(f"wrote {args.results_dir / 'vintage_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
