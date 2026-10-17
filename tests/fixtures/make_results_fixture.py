"""Build the committed fictional results fixture used by figure and report tests.

The fixture covers every results file in ``RESULTS_SCHEMAS`` and every specification
in ``config/specifications.yaml``. Values are plausible in sign and scale but
unmistakably fictional: every coefficient in ``main.csv``, ``robustness.csv`` and
``extension.csv`` is an integer multiple of :data:`FICTIONAL_UNIT`, as are the
per-predictor mean returns in ``predictor_summary.csv``. The script is seeded and
uses :func:`anomaly_decay.io.write_results`, so two runs produce byte-identical files
(asserted in ``tests/test_results_fixture.py``).
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from anomaly_decay.config import Spec, load_specifications
from anomaly_decay.io import write_results
from anomaly_decay.schemas import RESULTS_SCHEMAS

FICTIONAL_UNIT = 0.111
"""Marker for fictional coefficients and per-predictor mean returns."""

SEED = 202510

_HERE = Path(__file__).resolve()
DEFAULT_OUT_DIR = _HERE.parent / "results_fixture"
REPO_ROOT = _HERE.parents[2]
SPECS_PATH = REPO_ROOT / "config" / "specifications.yaml"

PREDICTORS = (
    "AM",
    "Accruals",
    "AssetGrowth",
    "BM",
    "Beta",
    "BetaFP",
    "EP",
    "GP",
    "IdioVol3F",
    "MaxRet",
    "Mom12m",
    "STreversal",
    "Size",
    "TrendFactor",
    "dVolPut",
)

# (post_sample coefficient steps, post_publication coefficient steps)
_COEF_STEPS: dict[str, tuple[int, int]] = {
    "main": (-3, -6),
    "main_twoway": (-3, -6),
    "main_dk": (-3, -6),
    "pubconv_alt": (-3, -5),
    "start_1963": (-2, -5),
    "clear_only": (-4, -7),
    "ew_deciles": (-2, -4),
    "vw_deciles": (-1, -3),
    "factor_adj": (-2, -5),
    "normalized": (-3, -6),
}

# (post_sample SE steps, post_publication SE steps); one step is 0.0111.
_SE_STEPS: dict[str, tuple[int, int]] = {
    "main": (14, 18),
    "main_twoway": (15, 19),
    "main_dk": (13, 17),
    "pubconv_alt": (14, 17),
    "start_1963": (13, 17),
    "clear_only": (16, 21),
    "ew_deciles": (12, 15),
    "vw_deciles": (11, 14),
    "factor_adj": (13, 17),
    "normalized": (10, 12),
}

_EXTENSION_TERMS = {
    "full_extension": ("post_sample", "post_publication"),
    "ext_pub_after_2000": (
        "post_sample",
        "post_publication",
        "post_publication_x_pub_after_2000",
    ),
}


def _coefficient(steps: int) -> float:
    return round(steps * FICTIONAL_UNIT, 12)


def _standard_error(steps: int) -> float:
    return round(steps * FICTIONAL_UNIT / 10.0, 12)


def _specs() -> list[Spec]:
    return load_specifications(SPECS_PATH)


def _spec_by_id(specs: Sequence[Spec]) -> dict[str, Spec]:
    return {spec.id: spec for spec in specs}


def _estimate_row(
    spec: str,
    label: str,
    term: str,
    coef: float,
    se: float,
    n_predictors: int,
    n_obs: int,
) -> dict[str, object]:
    return {
        "spec": spec,
        "label": label,
        "term": term,
        "coef": coef,
        "se": se,
        "t": coef / se,
        "n_predictors": n_predictors,
        "n_obs": n_obs,
    }


def _main_rows() -> list[dict[str, object]]:
    coefs = _COEF_STEPS["main"]
    ses = _SE_STEPS["main"]
    rows = []
    for index, term in enumerate(("post_sample", "post_publication")):
        rows.append(
            {
                "spec": "main",
                "term": term,
                "coef": _coefficient(coefs[index]),
                "se": _standard_error(ses[index]),
                "t": _coefficient(coefs[index]) / _standard_error(ses[index]),
                "n_predictors": 12,
                "n_obs": 5000,
            }
        )
    return rows


def _robustness_rows(specs: Sequence[Spec]) -> list[dict[str, object]]:
    by_id = _spec_by_id(specs)
    extension_ids = {spec.id for spec in specs if spec.group == "extension"}
    # Every non-main, non-extension spec is a robustness spec. Deriving the list this
    # way keeps the fixture stable while the base-inheritance group fix (#17) lands.
    order = ["main", *[spec.id for spec in specs if spec.id not in extension_ids | {"main"}]]
    rows = []
    for index, spec_id in enumerate(order):
        spec = by_id[spec_id]
        coef_steps = _COEF_STEPS[spec_id]
        se_steps = _SE_STEPS[spec_id]
        for term_index, term in enumerate(("post_sample", "post_publication")):
            rows.append(
                _estimate_row(
                    spec_id,
                    spec.label,
                    term,
                    _coefficient(coef_steps[term_index]),
                    _standard_error(se_steps[term_index]),
                    12,
                    5000 + 13 * index,
                )
            )
    return rows


def _extension_rows(specs: Sequence[Spec]) -> list[dict[str, object]]:
    by_id = _spec_by_id(specs)
    rows = []
    for spec_id, terms in _EXTENSION_TERMS.items():
        spec = by_id[spec_id]
        for index, term in enumerate(terms):
            if spec_id == "full_extension":
                coef_steps, se_steps, n_obs = (-2, -5)[index], (10, 12)[index], 6400
            else:
                coef_steps = (-2, -4, -2)[index]
                se_steps = (10, 12, 6)[index]
                n_obs = 6300
            rows.append(
                _estimate_row(
                    spec_id,
                    spec.label,
                    term,
                    _coefficient(coef_steps),
                    _standard_error(se_steps),
                    15,
                    n_obs,
                )
            )
    return rows


def _decay_rows() -> list[dict[str, object]]:
    in_sample_mean = _coefficient(10)
    coefs = _COEF_STEPS["main"]
    ses = _SE_STEPS["main"]
    rows = []
    for index, term in enumerate(("post_sample", "post_publication")):
        coef = _coefficient(coefs[index])
        se = _standard_error(ses[index])
        rows.append(
            {
                "spec": "main",
                "method": "pooled_ratio",
                "term": term,
                "decay_pct": round(100.0 * coef / in_sample_mean, 12),
                "se_pct": round(100.0 * se / in_sample_mean, 12),
                "in_sample_mean": in_sample_mean,
                "n_predictors": 12,
                "n_obs": 5000,
            }
        )
    normalized_coefs = _COEF_STEPS["normalized"]
    normalized_ses = _SE_STEPS["normalized"]
    for index, term in enumerate(("post_sample", "post_publication")):
        coef = _coefficient(normalized_coefs[index])
        se = _standard_error(normalized_ses[index])
        rows.append(
            {
                "spec": "normalized",
                "method": "normalized",
                "term": term,
                "decay_pct": round(100.0 * coef, 12),
                "se_pct": round(100.0 * se, 12),
                "in_sample_mean": 1.0,
                "n_predictors": 12,
                "n_obs": 5000,
            }
        )
    return rows


def _specs_run_rows(specs: Sequence[Spec]) -> list[dict[str, object]]:
    return [
        {
            "spec": spec.id,
            "label": spec.label,
            "status": "ok",
            "exploratory": spec.exploratory,
        }
        for spec in specs
    ]


def _predictor_summary_rows() -> list[dict[str, object]]:
    rows = []
    for index, signalname in enumerate(PREDICTORS):
        in_sample_steps = 3 + index
        means = {
            "in_sample": in_sample_steps,
            "post_sample": in_sample_steps - 2,
            "post_publication": in_sample_steps - 4,
        }
        se_steps = {
            "in_sample": 20 + index % 3,
            "post_sample": 30 + index % 5,
            "post_publication": 28 + index % 5,
        }
        n_months = {
            "in_sample": 300 + 5 * index,
            "post_sample": 20 + 2 * index,
            "post_publication": 60 + 3 * index,
        }
        for window in ("in_sample", "post_sample", "post_publication"):
            rows.append(
                {
                    "signalname": signalname,
                    "window": window,
                    "n_months": n_months[window],
                    "mean_ret": _coefficient(means[window]),
                    "se_mean": _standard_error(se_steps[window]),
                }
            )
    return rows


def _event_time_rows() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    rows = []
    for anchor in ("sample_end", "publication"):
        for event_month in range(-60, 121):
            if event_month < 0:
                if anchor == "sample_end":
                    mean_steps = 42 + rng.randint(-2, 2)
                else:
                    mean_steps = 35 + round(3 * (event_month + 60) / 60) + rng.randint(-2, 2)
                se_steps = 12 + min(abs(event_month) // 30, 4) + rng.randint(0, 2)
            else:
                if anchor == "sample_end":
                    mean_steps = 40 - round(20 * event_month / 120) + rng.randint(-2, 2)
                else:
                    mean_steps = 26 - round(14 * event_month / 120) + rng.randint(-2, 2)
                se_steps = 12 + min(event_month // 30, 4) + rng.randint(0, 2)
            rows.append(
                {
                    "anchor": anchor,
                    "event_month": event_month,
                    "mean_ret": round(mean_steps * FICTIONAL_UNIT / 10.0, 12),
                    "se": _standard_error(se_steps),
                    "n_predictors": 12,
                }
            )
    return rows


def _coverage_rows() -> list[dict[str, object]]:
    items = {
        "vintage": "202510",
        "first_month": "192601",
        "last_month": "202412",
        "n_predictors_op": "15",
        "n_predictors_deciles_ew": "15",
        "n_predictors_deciles_vw": "15",
        "n_clear": "12",
        "n_published_by_2013": "12",
        "n_predictor_months_op": "12801",
    }
    return [{"item": item, "value": value} for item, value in items.items()]


def build_results_fixture(out_dir: Path) -> list[Path]:
    """Write all eight results CSVs under *out_dir* and return their paths."""
    specs = _specs()
    frames: dict[str, list[dict[str, object]]] = {
        "main.csv": _main_rows(),
        "decay.csv": _decay_rows(),
        "robustness.csv": _robustness_rows(specs),
        "extension.csv": _extension_rows(specs),
        "specs_run.csv": _specs_run_rows(specs),
        "predictor_summary.csv": _predictor_summary_rows(),
        "event_time.csv": _event_time_rows(),
        "coverage.csv": _coverage_rows(),
    }
    paths = []
    for name, rows in frames.items():
        frame = pl.DataFrame(rows, schema=RESULTS_SCHEMAS[name])
        paths.append(write_results(frame, name, out_dir))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: regenerate the committed fixture (or write to ``--out-dir``)."""
    parser = argparse.ArgumentParser(
        prog="python tests/fixtures/make_results_fixture.py",
        description="Build the fictional results fixture used by figure and report tests.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args(argv)
    build_results_fixture(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
