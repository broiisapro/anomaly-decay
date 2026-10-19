# Epic 00 — Foundation

## Goal

Stand up everything later epics stand on: a reproducible uv project with CI, a cached and
hash-verified data layer, the interface contracts (schemas, specification list, typed
stubs), a committed 15-predictor fixture with an independent reference panel, and a
Quarto report that renders from `results/` alone.

## How it works

- **Environment.** Python 3.12 pinned in `.python-version`, dependencies resolved by uv
  into a committed `uv.lock`; CI runs `uv sync --frozen` on all four jobs. `polars` is
  constrained to `>=1.30,<2` because the OSAP downloader is untested against Polars 2.
- **Fetch and cache.** `anomaly_decay.data.fetch` wraps `openassetpricing` with retries
  (five attempts, 2**n-second backoff), validates every frame (expected columns, minimum
  rows, numeric `ret`, non-HTML responses) and writes canonical CSVs — fixed column order,
  sorted keys, LF endings — under `data/raw/<vintage>/`. Ken French FF5 and momentum zips
  are downloaded directly, verified against the SHA-256s in `config/data.toml`, parsed by
  `anomaly_decay.data.french` (monthly block only, `-99.99`/`-999` to null, percent per
  month) and committed as canonical CSVs under `data/pinned/`. `data/manifest.lock`
  hashes what is on disk; entries under the gitignored `data/raw/` are reported as "not
  cached" on a fresh clone instead of failing, and `make verify-hashes` prints
  `all hashes match`.
- **Fixture.** `tests/fixtures/make_fixture_data.py` produces the 15-predictor fixture in
  the raw-cache layout (`tests/fixtures/data/...`) plus `tests/fixtures/manifest.lock`.
  `tests/fixtures/build_fixture_panel.py` is an independent reference implementation of
  the window rules (plain Python loops, no imports from `anomaly_decay`) producing
  `tests/fixtures/panel_fixture.parquet` exactly per the panel schema.
- **Contracts.** `anomaly_decay.schemas` holds the predictor, panel and results schemas
  (dtypes and column order) and `validate_frame`; `anomaly_decay.config` loads
  `config/data.toml` and resolves `config/specifications.yaml` (`base` inheritance,
  key-by-key merging of `universe`/`months`/`cov_params`). The pipeline modules are typed
  stubs with fixed names, signatures, CLIs and docstrings.
- **Gates.** `make reproduce-fixture` verifies the fixture manifest, validates the panel
  fixture against the schema, renders the placeholder report from
  `tests/fixtures/dummy_results/main.csv` and asserts both output files exist. CI runs
  `lint` (ruff check + format check), `types` (mypy strict on `anomaly_decay`), `tests`
  (pytest, offline) and `reproduce-fixture` on self-hosted macOS ARM64 runners.
- **Report.** `report/` is a Quarto project (`_site` output, Jupyter engine, HTML +
  Typst PDF) whose inputs are listed explicitly (`project.render`). It reads only from
  the directory given by `ANOMALY_DECAY_RESULTS` (default `results/`, repo-root-relative)
  and computes its numbers inline at render time.

## Key decisions and reasoning

- **Canonical CSV is our format, not the package's.** `openassetpricing` returns frames
  and has no caching contract of its own; writing canonical files makes hashing
  meaningful and lets a human drop in files manually if the downloader breaks. Byte
  determinism of `anomaly_decay.io.write_csv_canonical` is tested.
- **Manifest paths are repo-root-relative and absent entries are preserved.** The OSAP
  cache is gitignored while its manifest is committed, so rewriting a manifest after a
  partial fetch must not drop entries for files that are simply not downloaded yet.
- **Validation is retried with the download.** A Google Drive quota page can degrade into
  a parseable-but-wrong frame; retrying only the HTTP call would not give the outage a
  chance to clear. Every attempt raises a `FetchError` naming the quota/HTML suspicion.
- **Ken French data is pinned in-repo.** The library revises its files monthly, so
  `data/pinned/*.csv` is committed and `make refresh-french` is the only path that
  re-downloads and rewrites the pinned hashes and date. `make reproduce` never touches
  the network for factors.
- **Offline test suite by construction.** An autouse fixture replaces the socket
  primitives and fails any test that opens a connection; fetch tests inject mocked
  `OpenAP` objects and fake HTTP sessions and run with a no-op sleep.
- **The fixture is synthetic and flagged as such.** Google Drive served the quota page
  for the signal-doc file on four attempts spread over 67 minutes (2026-10-16 23:51,
  2026-10-17 00:13, 00:35, 00:58 ET); the `openassetpricing` constructor reads that file,
  so no OSAP file could be downloaded. The fallback is a deterministic synthetic fixture
  (seed `20261016`, normal(0.5, 4.0) percent per month) over the real metadata of the 15
  predictors, with the metadata source committed as `signaldoc_raw.csv` so regeneration
  is offline and exact. Two edge cases are deliberate: `MaxRet` has three null-return
  months and `dVolPut` has a negative in-sample mean, exercising the drop rule and the
  `ret_norm` null rule. `tests/fixtures/data/FIXTURE_SOURCE.txt` records the source;
  `make_fixture_data.py --source osap` rebuilds the fixture from the real cache once
  `make fetch` succeeds.
- **All three portfolios are in the fixture.** `op`, `deciles_ew` and `deciles_vw` make
  the equal- and value-weighted robustness rows runnable offline in Epic 01.
- **Specification resolution is strict.** Unknown keys, unknown or circular `base`
  chains, and enum values outside the documented sets raise `ConfigError`. Later
  additions must carry `exploratory: true`; the freeze is enforced around the
  `specs-frozen` tag by the orchestrator procedure.
- **CI uses `astral-sh/setup-uv` rather than the runner PATH.** The host's `uv` lives in
  `~/.local/bin`, which a launchd-managed runner process may not inherit; the action
  installs the pinned version (`0.11.26`) explicitly. `actions/checkout@v5` (Node 24) is
  verified for the self-hosted macOS ARM64 runners. Action caching is off because uv's
  own wheel cache is shared across the three runners on this machine.
- **The report lists its inputs explicitly** (`project.render: [index.qmd]`). Quarto's
  directory discovery consults git ignore rules; agent worktrees live under a
  gitignored `.worktrees/` path, where discovery silently finds zero inputs. Explicit
  `render` makes the project deterministic in both worktrees and normal clones.
- **The offline gate renders the PDF too.** `reproduce-fixture` asserts both
  `report/_site/index.html` and `report/_site/index.pdf`, so a Typst regression fails CI
  instead of going unnoticed.

## Interfaces for future epics

- Schemas: `anomaly_decay.schemas.PREDICTORS_SCHEMA` (`data/processed/predictors.parquet`),
  `PANEL_SCHEMA` (`data/processed/panel.parquet`, key `signalname, portfolio, yyyymm`),
  `RESULTS_SCHEMAS` for the eight files under `results/`, plus `validate_frame`.
- Specification list: `config/specifications.yaml` (12 entries, IDs `main` through
  `ext_pub_after_2000`); `anomaly_decay.config.load_specifications` returns resolved
  `Spec` objects; `Spec` fields are the filter contract for estimation.
- Stubs with fixed signatures and CLIs: `anomaly_decay.data.metadata.build_metadata`,
  `anomaly_decay.panel.windows.{post_publication_start, classify_month, classify_frame}`,
  `anomaly_decay.panel.build.build_panel`,
  `anomaly_decay.estimate.main.{run_spec, decay_table, main}`,
  `anomaly_decay.estimate.robustness.run_robustness`,
  `anomaly_decay.estimate.extension.run_extension`, and the three figure functions in
  `anomaly_decay.figures.{event_time, distribution, coefplot}`.
- Fixture inputs: `tests/fixtures/data/` (raw-cache layout) hashed by
  `tests/fixtures/manifest.lock`; `tests/fixtures/dummy_results/main.csv` for the
  placeholder render; `tests/fixtures/panel_fixture.parquet` as the cross-check target
  for `build_panel`.
- Commands: `make fetch`, `make verify-hashes`, `make refresh-french`, `make metadata`,
  `make panel`, `make estimate`, `make figures`, `make report`, `make report-placeholder`,
  `make reproduce-fixture`, `make reproduce`.

## Deviations from the plan

- **Synthetic fixture instead of the 202510 subset** (see above). The issue's fallback
  applies, the PR carries the "(synthetic fixture)" marker, and the fixture source file
  records it.
- **The report's `results_dir` default is repo-root-relative** through
  `QUARTO_PROJECT_DIR`, rather than relative to `report/`; this matches the documented
  `results/` location and works for both render targets.
- **`project.render` is explicit** in `_quarto.yml` (see the worktree discovery reason).
- **Typst PDF in CI works on the self-hosted runner.** The first `reproduce-fixture` run
  that includes the render compiled `index.typ` to `index.pdf` and asserted both output
  files (run 39004970881); no HTML-only fallback is needed, so decision S11 is not
  exercised.
- **Ken French pinned files currently run through 2026-08** (the download date's latest
  month), not through the OSAP vintage's 2024-12; this is harmless for joins because the
  panel only uses months present in the returns.

## Known limitations and deferred items

- No OSAP canonical cache exists yet; the committed manifest carries the pinned Ken
  French files and their cached zips only. Once `make fetch` succeeds, the OSAP entries
  and the real fixture remain to be refreshed (`make_fixture_data.py --source osap`).
- `data/processed/` is empty; window classification, metadata building and panel
  construction are stubs. `make reproduce` is wired end to end but fails at `make
  metadata` by design until Epic 01.
- Figures are stubs; the report shows the placeholder `main.csv` table and one computed
  inline number per coefficient, with all sections present but marked as placeholders.
- The signal-doc fixture is a reduced 10-column extract of the OSAP signal documentation
  covering the columns the pipeline consumes; it is not the full documentation file.
- Nothing is tagged yet: `specs-frozen` is created by the orchestrator before the first
  full-data estimation, and `epic-00-complete` after this epic merges.

## Correction (Epic 01)

The fixture described above under "The fixture is synthetic and flagged as such" no longer
exists. Epic 01 (PR #51, "test(fixture): rebuild the 15-predictor fixture from real OSAP
202510 data") replaced it with real OSAP 202510 data: the synthetic generator path was
removed from `tests/fixtures/make_fixture_data.py`, which now builds the fixture as a
subset of the pinned OSAP canonical cache only, and `tests/fixtures/data/FIXTURE_SOURCE.txt`
records `source: osap-202510` with the source-file hashes. Edge cases the synthetic fixture
injected on purpose (a null-return month, a predictor with a non-positive in-sample mean)
are covered by hand-built frames in the unit tests instead. The rest of this document is
kept as the historical record of Epic 00.
