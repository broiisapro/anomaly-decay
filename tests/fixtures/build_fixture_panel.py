"""Reference builder for tests/fixtures/panel_fixture.parquet (independent implementation).

Deliberately simple: plain Python loops over months and dictionaries, Polars only for
CSV/Parquet I/O. It imports nothing from ``anomaly_decay`` so that Epic 01 can
cross-check the production panel code against a second, independently written
implementation of the window rules (architecture §4.6):

- in_sample: January ``SampleStartYear`` - December ``SampleEndYear`` inclusive;
- post_publication: from January ``Year+1`` (default) or January ``Year`` (alt);
- post_sample: January ``SampleEndYear+1`` up to the month before post-publication;
- pre_sample rows are dropped when the panel is built; if the publication start falls
  inside the in-sample period, in_sample wins.

Returns are percent per month. ``is_mean`` is the predictor x portfolio in-sample mean;
``ret_norm`` is ``ret / is_mean`` and is null when ``is_mean <= 0``. Ken French factors
are joined on ``yyyymm`` (null where absent). Rows with a null ``ret`` are dropped and
counted. Columns and dtypes follow docs/architecture.md §5.2 exactly.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

PORTFOLIOS: tuple[str, ...] = ("op", "deciles_ew", "deciles_vw")
FACTORS: tuple[str, ...] = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd", "rf")

PANEL_FIXTURE_SCHEMA = {
    "signalname": pl.Utf8,
    "portfolio": pl.Utf8,
    "yyyymm": pl.Int32,
    "month": pl.Date,
    "ret": pl.Float64,
    "window": pl.Utf8,
    "post_sample": pl.Int8,
    "post_publication": pl.Int8,
    "window_alt": pl.Utf8,
    "post_sample_alt": pl.Int8,
    "post_publication_alt": pl.Int8,
    "event_time_sample_end": pl.Int32,
    "event_time_pub": pl.Int32,
    "is_mean": pl.Float64,
    "ret_norm": pl.Float64,
    "sample_start_year": pl.Int16,
    "sample_end_year": pl.Int16,
    "pub_year": pl.Int16,
    "clear": pl.Boolean,
    "mkt_rf": pl.Float64,
    "smb": pl.Float64,
    "hml": pl.Float64,
    "rmw": pl.Float64,
    "cma": pl.Float64,
    "umd": pl.Float64,
    "rf": pl.Float64,
}


def _window(yyyymm: int, first_ym: int, last_ym: int, publication_first_ym: int) -> str:
    if yyyymm < first_ym:
        return "pre_sample"
    if yyyymm <= last_ym:
        return "in_sample"
    if yyyymm >= publication_first_ym:
        return "post_publication"
    return "post_sample"


def _publication_first_ym(pub_year: int, convention: str) -> int:
    return (pub_year + 1) * 100 + 1 if convention == "default" else pub_year * 100 + 1


def _event_time(yyyymm: int, anchor_ym: int) -> int:
    return (yyyymm // 100 - anchor_ym // 100) * 12 + (yyyymm % 100 - anchor_ym % 100)


def _read_factors(pinned_dir: Path) -> dict[int, dict[str, float | None]]:
    joined: dict[int, dict[str, float | None]] = {}
    for name in ("ff5_monthly.csv", "umd_monthly.csv"):
        frame = pl.read_csv(pinned_dir / name)
        for row in frame.iter_rows(named=True):
            values = joined.setdefault(row["yyyymm"], {})
            for factor in FACTORS:
                if factor in row:
                    values[factor] = row[factor]
    for values in joined.values():
        for factor in FACTORS:
            values.setdefault(factor, None)
    return joined


def _read_ls_rows(path: Path) -> dict[str, list[dict[str, object]]]:
    frame = pl.read_csv(path, schema_overrides={"date": pl.Utf8})
    by_signal: dict[str, list[dict[str, object]]] = {}
    for row in frame.iter_rows(named=True):
        if row["port"] != "LS":
            continue
        by_signal.setdefault(row["signalname"], []).append(row)
    return by_signal


def build_fixture_panel(fixtures_dir: Path, out_path: Path | None = None) -> pl.DataFrame:
    data_dir = fixtures_dir / "data"
    signal_doc = pl.read_csv(data_dir / "osap" / "202510" / "signal_doc.csv")
    factors = _read_factors(data_dir / "pinned")
    ports = {
        name: _read_ls_rows(data_dir / "osap" / "202510" / f"port_{name}.csv")
        for name in PORTFOLIOS
    }

    records: list[dict[str, object]] = []
    dropped_null_ret = 0
    non_positive_means: list[str] = []
    for predictor in signal_doc.iter_rows(named=True):
        signalname = predictor["Acronym"]
        sample_start_year = int(predictor["SampleStartYear"])
        sample_end_year = int(predictor["SampleEndYear"])
        pub_year = int(predictor["Year"])
        clear = predictor["Predictability in OP"] == "1_clear"
        first_ym = sample_start_year * 100 + 1
        last_ym = sample_end_year * 100 + 12
        pub_first = _publication_first_ym(pub_year, "default")
        pub_first_alt = _publication_first_ym(pub_year, "alt")

        for portfolio in PORTFOLIOS:
            rows = ports[portfolio].get(signalname, [])
            if not rows:
                continue
            months: list[tuple[int, dict[str, object]]] = []
            for row in rows:
                date_text = str(row["date"])
                yyyymm = int(date_text[0:4]) * 100 + int(date_text[5:7])
                months.append((yyyymm, row))
            months.sort()

            in_sample_returns = [
                float(row["ret"])
                for yyyymm, row in months
                if _window(yyyymm, first_ym, last_ym, pub_first) == "in_sample"
                and row["ret"] is not None
            ]
            if not in_sample_returns:
                continue
            is_mean = sum(in_sample_returns) / len(in_sample_returns)
            if is_mean <= 0:
                non_positive_means.append(f"{signalname}/{portfolio}")

            for yyyymm, row in months:
                window = _window(yyyymm, first_ym, last_ym, pub_first)
                if window == "pre_sample":
                    continue
                if row["ret"] is None:
                    dropped_null_ret += 1
                    continue
                window_alt = _window(yyyymm, first_ym, last_ym, pub_first_alt)
                ret = float(row["ret"])
                factor_values = factors.get(yyyymm, {})
                records.append(
                    {
                        "signalname": signalname,
                        "portfolio": portfolio,
                        "yyyymm": yyyymm,
                        "month": datetime.date(yyyymm // 100, yyyymm % 100, 1),
                        "ret": ret,
                        "window": window,
                        "post_sample": int(window == "post_sample"),
                        "post_publication": int(window == "post_publication"),
                        "window_alt": window_alt,
                        "post_sample_alt": int(window_alt == "post_sample"),
                        "post_publication_alt": int(window_alt == "post_publication"),
                        "event_time_sample_end": _event_time(
                            yyyymm, (sample_end_year + 1) * 100 + 1
                        ),
                        "event_time_pub": _event_time(yyyymm, pub_first),
                        "is_mean": is_mean,
                        "ret_norm": ret / is_mean if is_mean > 0 else None,
                        "sample_start_year": sample_start_year,
                        "sample_end_year": sample_end_year,
                        "pub_year": pub_year,
                        "clear": clear,
                        "mkt_rf": factor_values.get("mkt_rf"),
                        "smb": factor_values.get("smb"),
                        "hml": factor_values.get("hml"),
                        "rmw": factor_values.get("rmw"),
                        "cma": factor_values.get("cma"),
                        "umd": factor_values.get("umd"),
                        "rf": factor_values.get("rf"),
                    }
                )

    panel = pl.DataFrame(records, schema=PANEL_FIXTURE_SCHEMA, orient="row").select(
        list(PANEL_FIXTURE_SCHEMA)
    )
    panel = panel.sort(["signalname", "portfolio", "yyyymm"])

    print(f"dropped {dropped_null_ret} rows with null ret")
    print(
        f"in-sample mean <= 0 for {len(non_positive_means)} predictor-portfolios: "
        f"{non_positive_means}"
    )
    print(f"panel rows: {panel.height}, predictors: {panel['signalname'].n_unique()}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(out_path)
        print(f"wrote {out_path}")
    return panel


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, default=None, help="output parquet path")
    args = parser.parse_args(argv)
    fixtures_dir = args.fixtures_dir.resolve()
    out = args.out or fixtures_dir / "panel_fixture.parquet"
    if not (fixtures_dir / "data" / "osap" / "202510" / "signal_doc.csv").is_file():
        print("fixture data missing; run make_fixture_data.py first", file=sys.stderr)
        return 1
    build_fixture_panel(fixtures_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
