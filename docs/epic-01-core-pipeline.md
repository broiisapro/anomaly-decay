# Epic 01 — Core pipeline

## Goal

Turn the Epic 00 stubs into the working pipeline: cached canonical data → predictor metadata → a
window-classified panel → pre-registered panel regressions → descriptive results → deterministic
figures → an executable Quarto report, plus an offline end-to-end gate that runs every stage on the
committed fixture and checks its outputs against independent expectations. Full-data estimation
remains behind the `specs-frozen` tag; the fixture is the only place estimates run
during development.

## How it works

- **Data layer.** `anomaly_decay.data.fetch` is cache-aware: it reads
  `data/manifest.lock` first, skips files whose size and SHA-256 already match, and fails loudly —
  leaving the manifest byte-identical and keeping the download as `<name>.rejected` — when a new
  download disagrees with an existing entry. Entries change only under `--update-manifest`; a file
  present with no entry is adopted once. All writes are atomic (`.tmp` plus rename). The 202510
  OSAP canonical files and their hashes are committed in `data/manifest.lock`; the Ken French
  factors stay pinned in `data/pinned/`. `anomaly_decay.data.metadata` builds the 212-predictor
  table (`Cat.Signal == "Predictor"`) exactly per `PREDICTORS_SCHEMA`, filling only null fields
  from `config/predictor_dates_supplement.csv` (committed with a header and zero rows; every
  supplement row needs a citation). `tests/test_realdata.py` is an opt-in marker suite
  (`-m realdata`) that checks unit and coverage invariants against the cached files, skipping with
  a reason when `data/raw/202510` is absent.
- **Fixture.** `tests/fixtures/data/` now holds a real subset of OSAP 202510 — 15 predictors,
  long-short rows for `op`, `deciles_ew` and `deciles_vw`, marked `osap-202510` in
  `FIXTURE_SOURCE.txt` with source hashes. `tests/fixtures/panel_fixture.parquet` (38,319 rows) is
  built by the independent reference script `tests/fixtures/build_fixture_panel.py`, which imports
  nothing from `anomaly_decay`; the synthetic generator was removed.
- **Panel.** `anomaly_decay.panel.windows` classifies months with pure integer `yyyymm` arithmetic
  (never calendar-year comparisons) and is Hypothesis-tested against a per-month reference;
  `anomaly_decay.panel.build.build_panel` reads canonical CSVs with
  `schema_overrides={"port": pl.Utf8, "date": pl.Utf8}` (R3), keeps `port == "LS"`, joins metadata
  and the pinned factors on `yyyymm`, classifies both publication conventions, keeps only
  in-sample/post-sample/post-publication months, and adds event-time anchors, `is_mean` and
  `ret_norm`. Drop paths are counted in `BuildStats`. On the real cache the full build is
  369,374 rows for 212/179/179 predictors in 13.9 s; the fixture build matches the reference panel
  bit-for-bit (`assert_frame_equal(check_exact=True)`).
- **Estimation.** `anomaly_decay.estimate.main.run_spec` fits
  `PanelOLS(dep ~ dummies, entity_effects=True)` (no constant) on the pandas `(signalname, month)`
  MultiIndex, mapping `clustered_time` → month-clustered, `clustered_twoway` → entity and month
  clustered, and `driscoll_kraay` → Bartlett-kernel covariance with the spec's bandwidth.
  `decay_table` produces the two decay definitions: pooled ratio (`100 * coef / in_sample_mean`
  over the spec's own sample, `se_pct` treating the mean as fixed) and normalized regression
  (`100 * coef` on `ret_norm`, `in_sample_mean = 1`). `robustness.py`/`extension.py` run the
  robustness and extension groups; `ext_pub_after_2000` reports the
  `post_publication_x_pub_after_2000` interaction as its own term. `factor_adjusted_returns`
  estimates FF5+UMD betas on in-sample months only (≥ 36 usable months) and subtracts `β·f`
  (no `rf`: the long-short series is zero-investment). The CLI runs all twelve specifications,
  isolates failures into `specs_run.csv` (`ok` / `failed: …` / `skipped` under `--only`), and
  writes `main.csv`, `decay.csv`, `robustness.csv`, `extension.csv` and `specs_run.csv` through
  `anomaly_decay.io.write_results`.
- **Results I/O and descriptive results.** `anomaly_decay.io` validates every results frame against
  `RESULTS_SCHEMAS`, orders columns, sorts by `RESULTS_SORT_KEYS` and writes canonical CSV;
  `read_results` parses with the full schema as overrides so all-null float columns keep their
  dtype. `anomaly_decay.estimate.descriptive` computes `predictor_summary`, `event_time` and
  `coverage` for the full-extension universe (`op`, all predictors, all months, default
  convention).
- **Figures.** `anomaly_decay.figures.style` pins Agg, a bundled font and a colour-blind-safe
  palette, fixes the SVG hash salt and suppresses PNG/SVG timestamps; the event-time, distribution
  and coefficient plots are byte-identical across processes (tested by rendering in two separate
  interpreters).
- **Report.** `anomaly_decay.reporting` is the report's only results path
  (`ANOMALY_DECAY_RESULTS`, default `<repo>/results`; relative values resolve against the repo
  root, never the kernel's working directory). It provides schema-typed loads (with an optional
  load that returns `None` for not-yet-produced files), number/percent/t-stat formatting that
  renders `n/a` instead of raising, a Markdown table builder, figure lookup and embedding, and the
  `config/original_paper.yaml` loader. The report has the nine required sections plus a
  specification appendix; every prose number is an inline expression, missing results render
  pending callouts, missing figures render warning callouts, and the paper's coefficients show
  "pending verification" until entered from the paper. The PDF embeds figures
  through the Jupyter image machinery, which avoids output-directory path rewrites in Typst.
- **Build and CI.** `Makefile` variables at the top drive the stage targets `metadata`, `panel`,
  `estimate`, `descriptive`, `figures`, `report`; `reproduce` chains them after `fetch`.
  `reproduce-fixture` is the offline gate described below; `report-fixture` renders the report
  from the committed fictional results fixture for report development; `report-placeholder`
  renders it with only the dummy `main.csv` to exercise the pending paths. CI keeps the four job
  IDs `lint`, `types`, `tests`, `reproduce-fixture`; the `tests` job checks out with
  `fetch-depth: 0` so the pre-registration freeze test can see tags.

### The offline end-to-end gate

`make reproduce-fixture` verifies the fixture manifest, then runs the production CLIs on the
committed fixture inputs, writing only under `data/processed/fixture/`:

1. `metadata` from the fixture signal doc → `predictors.parquet`;
2. `panel` from `tests/fixtures/data/` → `panel.parquet`, asserted exactly equal to the reference
   `tests/fixtures/panel_fixture.parquet`;
3. `estimate` (all twelve specifications), `descriptive` (three tables) and the three figure CLIs
   into `data/processed/fixture/results/`;
4. the three result stages run a second time, with a SHA-256 listing of every file under
   `.../results/` required to be byte-identical (`cmp` of the listings);
5. `tests/test_reproduce_fixture.py` compares `main.csv` with
   `tests/fixtures/expected_main_fixture.csv` — float columns to 1e-10, everything else exactly —
   and re-asserts the panel equality; the comparison helpers themselves are covered by hand-built
   frames in the same module;
6. the report renders from `data/processed/fixture/results/` with Quarto, and both
   `report/_site/index.html` and `index.pdf` must exist.

Nothing in the gate writes to `results/`; the fixture results are throwaway and gitignored. The
fixture headline on the current real fixture is post-sample −57.4% (SE 21.2) and post-publication
−42.7% (SE 13.7) on 12 predictors, printed by the gate for the log.

## Key decisions and reasoning

- **A fixture result is still a result.** The offline gate runs the real estimation code on real
  (subsampled) data; the independent reference panel and the committed expected `main.csv` are two
  separate cross-checks, so a window off-by-one or covariance change fails the gate rather than
  passing silently.
- **Cache adoption without a flag, changes only with one.** A file present with no manifest entry
  is hashed and recorded (that is how a cache gets populated, including the documented manual-copy
  fallback); changing an existing entry requires `--update-manifest`. Declining downloads land as
  `.rejected` and never touch the manifest.
- **Metadata fills only nulls, with citations.** A supplement row targeting a non-null field is
  ignored; duplicates, unknown names/fields and missing citations raise. Same-year publication
  warns rather than fails.
- **The real fixture replaced the synthetic one outright.** The synthetic generator, its raw signal
  doc and its network fallback were removed; the synthetic edge cases (null returns, non-positive
  in-sample mean) live on as hand-built frames in reference-builder tests.
- **`is_mean` uses sequential arithmetic.** Polars SIMD means differed from the reference builder
  in the last ulp for 23 of 45 fixture groups; sequential summation in ascending month order costs
  nothing at this scale and keeps the exact cross-check meaningful.
- **Calendar facts from the real data are encoded, not papered over.** The 202510 cache contains
  Saturday closes before 1952-06 (the NYSE traded on Saturdays historically); the invariants allow
  Saturdays only before that date and never Sundays.
- **Pooled-ratio denominator is the spec's own sample.** The in-sample mean is computed on the
  filtered estimation sample (portfolio, universe, months) restricted to in-sample rows; `se_pct`
  treats it as fixed, since it is a sample characteristic rather than an estimated parameter.
  `decay.csv` counts are the regression's counts because the percentages are transformations of
  its coefficients.
- **Failure isolation lives in the CLI.** One failed specification records its reason and the run
  continues; `run_robustness`/`run_extension` stay strict for direct callers. The CLI estimates
  each spec once and assembles the robustness and extension tables from that loop.
- **Factor betas come from in-sample months only, with no risk-free subtraction.** The estimator
  needs ≥ 36 usable in-sample months; the classic-bug test perturbs post-period factors and
  verifies the adjusted returns still use the in-sample betas.
- **The report never reads anything but the results directory, and it never crashes on a missing
  file.** `reporting.load_optional` turns a not-yet-produced file into a pending callout, which
  keeps the offline gate green between staged merges and lets the same document render from dummy,
  fixture or full results.
- **Figures are embedded as image data, not path references.** Pandoc rewrites absolute Markdown
  image paths relative to the document directory, which broke the Typst build; the Jupyter image
  display machinery works for both HTML and PDF.
- **`table()` selects the requested columns before formatting.** Results files start with a `spec`
  column; zipping row values positionally against a column list that omits it silently formatted
  the wrong values and raised on strings.
- **The end-to-end comparisons live in the test module.** The 1e-10 main comparison is a committed,
  unit-tested helper rather than an inline shell comparison on strings; it runs inside the gate via
  pytest and skips in an ordinary test run when the fixture outputs are absent.
- **Determinism is verified twice.** The gate byte-compares hash listings of two consecutive
  result runs, and the suite separately asserts CLI byte identity and cross-process figure
  byte identity; BLAS/OpenMP threads are pinned in the Makefile.

## Interfaces for future epics

- **Results contract.** Eight files under the results directory with `RESULTS_SCHEMAS`,
  `RESULTS_SORT_KEYS`, `write_results`/`read_results`; the report reads them only through
  `anomaly_decay.reporting` (`ANOMALY_DECAY_RESULTS`).
- **Data layer.** `anomaly_decay.data.metadata.build_metadata(signal_doc, supplement=None)` and
  `anomaly_decay.panel.build.build_panel(data_dir, metadata, portfolios, *, pinned_dir, vintage)`.
- **Estimation.** `run_spec`/`decay_table`/`main`, `robustness.run_robustness`,
  `extension.run_extension`, `factors.factor_adjusted_returns(panel, *, min_months=36)`, and the
  descriptive functions `predictor_summary`, `event_time`, `coverage`.
- **Figures.** `plot_event_time`, `plot_distribution`, `plot_coefficients`, each with the
  `--results-dir` CLI writing both PNG and SVG.
- **Committed expectations.** `tests/fixtures/panel_fixture.parquet`,
  `tests/fixtures/expected_main_fixture.csv`,
  `tests/fixtures/results_fixture/` (fictional, for report tests),
  `tests/fixtures/realdata_expectations.csv` (per-predictor first/last months, opt-in suite).
- **Commands.** `make fetch`, `verify-hashes`, `refresh-french`, `metadata`, `panel`, `estimate`,
  `descriptive`, `figures`, `report`, `report-fixture`, `report-placeholder`,
  `reproduce-fixture`, `reproduce`.
- **Pre-registration.** `specs-frozen` freezes `config/specifications.yaml` before the first
  full-data estimation; the freeze test compares the working file with the tagged version and
  skips until the tag exists (or fails with `ANOMALY_DECAY_REQUIRE_FREEZE=1`). `main_twoway` and
  the other pre-registered IDs, including the `ext_pub_after_2000` interaction, are frozen.

## Deviations from the plan

- **Fixture results go to `data/processed/fixture/` instead of `results/`.** The
  PRD's literal `results/main.csv` would make an offline fixture run indistinguishable from a
  real one; the gate therefore writes under the gitignored processed directory and compares
  against committed expectations.
- **`report-fixture` was added alongside the end-to-end gate.** Rendering from the committed
  fictional results fixture is useful for report development and for CI when only the report
  changes; it generates figures into a throwaway copy so the committed fixture stays
  byte-identical.
- **`report-placeholder` was kept.** It renders with only the dummy `main.csv`, exercising the
  report's missing-results paths; the full gate renders with complete fixture results.
- **The expected fixture briefly lagged the real fixture.** The expected `main.csv` was prepared
  while the fixture was still synthetic and the real fixture merged first; a follow-up regenerated
  the expected file on `osap-202510` and the gate has failed loudly on any fixture change since
  (which is the intended behaviour).
- **A documented historical exception: OSAP dates are last trading days, including Saturdays
  before mid-1952.** The original design note "last trading day of the month" is correct for
  weekdays; the real-data invariant suite records the Saturday exception explicitly rather than
  excluding real sessions.
- **`docs/epic-00-foundation.md` describes the synthetic fixture it was written with.** Its
  closing correction records the real-fixture rebuild; this document is the current description of
  the fixture.
- **`main_twoway` can produce NaN standard errors on degenerate panels.** The two-way covariance
  can be non-positive-semidefinite on small deterministic test panels; this is the estimator's
  output, documented rather than worked around, and it does not occur on the real fixture.

## Known limitations and deferred items

- **No full-data results exist yet.** All estimation in this epic ran on the fixture or hand-built
  panels; the first full run happens after `specs-frozen` is created, and the report
  marks every full-data-dependent statement as pending until then.
- **The report's comparison numbers are unverified.** `config/original_paper.yaml` carries the
  citation and null values with `status: pending-verification`; the report shows "pending
  verification" until a human enters and checks them.
- **`read_results` does not enforce row sortedness or key uniqueness** (the schemas declare no
  key); the writer always sorts, so project-produced files are conformant.
- **`factor_adjusted_returns` checks only the columns it needs** and does not validate the whole
  panel; the panel builder validates upstream.
- **`event_time` drops months with no predictor** (they are absent rather than reported with
  `n_predictors = 0`) and collapses duplicate predictor rows via the group mean (impossible under
  the panel key today).
- **The freeze test skips until the tag exists**; enforcement is active once `specs-frozen` is
  created, and CI now fetches tags in the `tests` job.
- **The coefplot's single-term edge is untested**: a robustness file missing one term would still
  render, with a legend entry for an empty series.
- **The distribution figure drops predictors missing either window**; the committed fixture has
  all 15 in both.
- **Windows on non-Mac platforms** would need `shasum`/`find` equivalents; the project targets
  macOS ARM64 for now.
