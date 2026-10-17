# Architecture — anomaly-decay

## 1. Purpose

anomaly-decay replicates McLean & Pontiff (2016, *Journal of Finance* 71(1), 5–32): long-short anomaly returns
fall after the original sample ends and fall further after publication (reported as 26% lower out-of-sample and
58% lower post-publication, on 97 predictors). We rebuild the result from the open Chen–Zimmermann
*Open Source Asset Pricing* (OSAP) portfolio returns, extend it through the latest data, and publish an executable
Quarto report in which every number is computed from `results/`.

## 2. Data flow

```
config/data.toml ─┐
                  ▼
 data/fetch.py ──► data/raw/<vintage>/*.csv  (gitignored; canonical CSVs)
   │               data/raw/ff/<date>/*       (Ken French zips + canonical CSVs)
   └─────────────► data/manifest.lock         (committed SHA-256 of every cached file)
                  ▼
 data/metadata.py ► data/processed/predictors.parquet   (one row per predictor)
                  ▼
 panel/windows.py (pure) + panel/build.py ► data/processed/panel.parquet
                  ▼
 estimate/main.py, robustness.py, extension.py ► results/*.csv
                  ▼
 figures/*.py ► results/figures/*.png|svg
                  ▼
 report/ (Quarto) ► report/_site/index.html, index.pdf
```

Offline twin: `make reproduce-fixture` runs the same pipeline on `tests/fixtures/` (15 real predictors) and
compares `results/main.csv` with `tests/fixtures/expected_main_fixture.csv` to 1e-10. The report never reads raw
or processed data; it reads only `results/`.

## 3. Package layout

```
anomaly_decay/
  config.py            load + validate config/data.toml and config/specifications.yaml
  schemas.py           panel, predictor-metadata and results schemas (Polars dtypes) + validators
  io.py                deterministic CSV/Parquet writers (sorted keys, LF, fixed column order)
  data/fetch.py        OSAP + Ken French download, retry, validation, canonical cache, manifest
  data/french.py       Ken French CSV-in-zip parser (monthly block only; -99.99/-999 → null)
  data/metadata.py     predictor metadata table (SignalDoc + cited supplement)
  panel/windows.py     window classification (pure functions, no I/O)
  panel/build.py       merge returns + metadata + factors → panel.parquet
  estimate/main.py     main specification; `--print-headline`
  estimate/robustness.py, estimate/extension.py
  figures/event_time.py, figures/distribution.py, figures/coefplot.py
config/  data.toml  specifications.yaml  predictor_dates_supplement.csv
report/  _quarto.yml  index.qmd
tests/   fixtures/  test_*.py
Makefile
```

## 4. Key decisions and reasoning

**4.1 Data access — wrap `openassetpricing` vs. own Google-Drive downloader.** OSAP files sit on a public Google
Drive folder. `openassetpricing` (0.0.2) resolves file IDs and returns DataFrames, but has no retries, no
detection of Drive's "quota exceeded" HTML page, and imports `wrds` at import time. Writing our own Drive
scraper would duplicate fragile logic. *Chosen:* use the package (PRD requirement), wrapped by `data/fetch.py`
with retries/backoff and validation (expected columns, minimum row counts, numeric `ret`). Because the package
returns DataFrames rather than bytes, the cache stores a **canonical CSV** written by our code (fixed column
order, sorted by key, LF line endings) and the manifest hashes those files. If Drive refuses downloads for a long
time, a human can place the canonical files manually and `make verify-hashes` validates them.

**4.2 Vintage.** Pinned to OSAP release `202510` (v2.00, returns through 2024-12), the latest at kickoff.
Phase 2 compares against an earlier vintage.

**4.3 Portfolios.** Main series: `op` (each predictor built as in its original paper), long-short rows
`port == "LS"`. EW/VW robustness: `deciles_ew` and `deciles_vw` long-short rows.

**4.4 Units and month keys.** OSAP and Ken French returns are both **percent per month**; nothing is converted
(asserted by `tests/test_panel.py::test_returns_unit_is_percent_monthly`). OSAP `date` is the last trading day of
the month, so every join uses `yyyymm` (Int32); `month` (first day of month, Date) is the time index for
linearmodels.

**4.5 Predictor universe and metadata.** `Cat.Signal == "Predictor"` (1_clear + 2_likely); "clear" means
`Predictability in OP == "1_clear"`. Sample period: `SampleStartYear`, `SampleEndYear`; publication year: `Year`
(year only — OSAP has no publication month, unlike MP who used the journal cover date; a documented deviation).
Missing fields are filled only from `config/predictor_dates_supplement.csv`, one cited source per row.

**4.6 Windows.** `panel/windows.py` maps (yyyymm, start, end, pub_year, convention) to one of
`pre_sample | in_sample | post_sample | post_publication`:
- in_sample: Jan `SampleStartYear` – Dec `SampleEndYear` inclusive;
- post_publication: from Jan `Year+1` (default) or Jan `Year` (alternative);
- post_sample: Jan `SampleEndYear+1` up to the month before post_publication (may be empty);
- pre_sample: before Jan `SampleStartYear` — dropped when the panel is built.
If the publication start would fall inside the in-sample period, in_sample wins and post_sample is empty
(documented edge rule, tested). Every panel row is therefore in exactly one of three windows.

**4.7 Estimation.** `linearmodels.PanelOLS(ret ~ post_sample + post_publication + EntityEffects)` with
`cov_type="clustered", cluster_time=True` (main) and two-way clustering (robustness). MP's published table uses
FGLS with contemporaneous cross-correlation; their text says month clustering gives similar results. A
Driscoll–Kraay row (`cov_type="kernel"`) is pre-registered as the closest library analogue. PanelOLS needs a
pandas frame with a (signalname, month) MultiIndex; Polars → pandas conversion happens only inside `estimate/`.

**4.8 Main specification = MP-comparable window.** Predictors published by 2013, returns through 2013-12, `op`,
default convention. The extension runs the same specification through the vintage's last month with all
predictors. Decay is reported two ways: (a) pooled ratio — coefficient ÷ pooled in-sample mean of the estimation
sample (this matches MP's 26%/58% headline, 0.150/0.582 and 0.337/0.582); (b) coefficients from regressing
`ret_norm = ret / predictor in-sample mean` (predictors with in-sample mean ≤ 0 excluded, and counted).

**4.9 Factor adjustment.** Per predictor, FF5 + UMD betas are estimated on **in-sample months only**; the adjusted
return is `ret − β'f` for every month. Months before 1963-07 (no FF5) are dropped for that specification.

**4.10 Pre-registration.** `config/specifications.yaml` lists every specification with a stable ID. It is frozen by
the annotated tag `specs-frozen` before the first full-data estimation; a test compares the working file with
the tagged version. Anything added later is labelled `exploratory: true` and shown as such in the report.

**4.11 Determinism.** Results are sorted by their key columns and written by `anomaly_decay.io` with fixed column
order; BLAS/OpenMP threads are pinned to 1 in Make targets; no unseeded randomness anywhere. `make reproduce`
twice must leave `git diff --exit-code results/` clean.

**4.12 Fixture cross-check.** `tests/fixtures/panel_fixture.parquet` is produced in Epic 0 by an independent
reference script (`tests/fixtures/build_fixture_panel.py`). Production code in `anomaly_decay.panel` must
reproduce it exactly — two independent implementations of the window rules guard against silent boundary errors.

**4.13 Report toolchain.** Quarto 1.10.x (bundled Typst) with the Jupyter engine; `jupyter` and `ipykernel` are dev
dependencies; render with `uv run quarto render report`. Numbers in prose are inline expressions
(`` `{python} value` ``) computed from `results/`. If Typst PDF rendering is flaky on CI runners, CI renders HTML
only and the PDF is built locally for releases.

**4.14 Tooling.** Python 3.12, uv (`uv.lock` committed, `uv sync --frozen` in CI), Polars `<2` (the downloader is
untested on Polars 2), pandas where libraries need it, linearmodels + statsmodels, Matplotlib, pytest + Hypothesis,
ruff, mypy (strict on `anomaly_decay`), Make.

## 5. Interface contracts

### 5.1 Predictor metadata — `data/processed/predictors.parquet`
| column | dtype | notes |
|---|---|---|
| signalname | Utf8 | OSAP `Acronym`; primary key |
| sample_start_year | Int16 | `SampleStartYear` |
| sample_end_year | Int16 | `SampleEndYear` |
| pub_year | Int16 | `Year` |
| journal | Utf8 | |
| authors | Utf8 | |
| cat_economic | Utf8 | `Cat.Economic` |
| cat_data | Utf8 | `Cat.Data` |
| clear | Boolean | `Predictability in OP == "1_clear"` |
| date_source | Utf8 | `signaldoc` or `supplement:<citation key>` |

### 5.2 Panel — `data/processed/panel.parquet`
Key: (`signalname`, `portfolio`, `yyyymm`), unique; sorted by key.
| column | dtype | notes |
|---|---|---|
| signalname | Utf8 | |
| portfolio | Utf8 | `op`, `deciles_ew`, `deciles_vw` |
| yyyymm | Int32 | e.g. 199501 |
| month | Date | first day of month |
| ret | Float64 | long-short return, **percent per month** |
| window | Utf8 | `in_sample`/`post_sample`/`post_publication` (default convention) |
| post_sample | Int8 | 0/1, default convention |
| post_publication | Int8 | 0/1, default convention |
| window_alt | Utf8 | alternative convention |
| post_sample_alt | Int8 | |
| post_publication_alt | Int8 | |
| event_time_sample_end | Int32 | months since Jan `SampleEndYear+1` (0 = first post-sample month) |
| event_time_pub | Int32 | months since first post-publication month (default convention) |
| is_mean | Float64 | predictor×portfolio in-sample mean of `ret` |
| ret_norm | Float64 | `ret / is_mean`; null when `is_mean <= 0` |
| sample_start_year, sample_end_year, pub_year | Int16 | copied from metadata |
| clear | Boolean | |
| mkt_rf, smb, hml, rmw, cma, umd, rf | Float64 | Ken French, percent; null before coverage |

### 5.3 Results — `results/` (CSV, comma-separated, header row, LF)
- `main.csv`: `spec, term, coef, se, t, n_predictors, n_obs` — spec `main`, terms `post_sample`, `post_publication`.
- `decay.csv`: `spec, method, term, decay_pct, se_pct, in_sample_mean, n_predictors, n_obs` — method `pooled_ratio` | `normalized`.
- `robustness.csv`: `spec, label, term, coef, se, t, n_predictors, n_obs`.
- `extension.csv`: `spec, label, term, coef, se, t, n_predictors, n_obs`.
- `specs_run.csv`: `spec, label, status, exploratory` — every spec in `specifications.yaml`, including failures.
- `predictor_summary.csv`: `signalname, window, n_months, mean_ret, se_mean` (portfolio `op`, default convention).
- `event_time.csv`: `anchor, event_month, mean_ret, se, n_predictors` — anchor `sample_end` | `publication`, months −60…+120.
- `coverage.csv`: `item, value` — vintage, first/last month, predictor counts.
- `figures/`: `event_time.{png,svg}`, `distribution.{png,svg}`, `coefplot.{png,svg}`.

### 5.4 Specification IDs (`config/specifications.yaml`)
`main`, `main_twoway`, `main_dk`, `pubconv_alt`, `start_1963`, `clear_only`, `ew_deciles`, `vw_deciles`,
`factor_adj`, `normalized`, `full_extension`, `ext_pub_after_2000`. Each entry: `id`, `label`, `group`
(`main|robustness|extension`), `portfolio`, `universe` (filters), `months` (`end_yyyymm`), `dep_var`,
`convention`, `cov` (`clustered_time|clustered_twoway|driscoll_kraay`), `extra_terms`, `exploratory`.

## 6. Makefile targets
`reproduce` (fetch → metadata → panel → estimate → figures → report), `reproduce-fixture` (same, offline, fixture
inputs, then compare to expected), `verify-hashes` (prints `all hashes match`), `report`, `lint`, `types`, `test`.

## 7. CI
GitHub Actions; jobs `lint`, `types`, `tests`, `reproduce-fixture` (job IDs are the required-check names).
Self-hosted macOS ARM64 runners while private (`runs-on: [self-hosted, macOS, ARM64]`, per-branch concurrency
with cancel-in-progress); GitHub-hosted runners after public release.
