# Epic 02 — Integration and Phase 2

## Goal

Turn the verified pipeline into the shipped replication: freeze the pre-registration and record the
original paper's verified values, produce the project's first full-data results, make the Quarto
report tell the McLean and Pontiff (2016) story end to end with every number computed from those
results, add the pre-registered Phase 2 analyses — moving-block bootstrap, multiple-testing lens,
data-vintage sensitivity — as report sections, and let CI enforce the freeze.

## How it works

### Pre-registration and the full-data run

- `config/phase2_preregistration.yaml` (bootstrap, multiple testing, vintages) and the verified
  values in `config/original_paper.yaml` land first (#57); the orchestrator's freeze watcher creates
  the annotated tag `specs-frozen` on the first `origin/main` commit where both pre-registration
  files match the approved SHA-256s and no commit has ever touched `results/` (decision O32). The
  tag was created on `ce0e0d54` about a minute after the merge.
- `tests/test_specs.py` compares both files with their tagged versions through one parametrised
  test; the `tests` CI job sets `ANOMALY_DECAY_REQUIRE_FREEZE=1`, so the freeze is enforced on
  every run (#59).
- After the tag, the first `make reproduce` commits the eight Phase 1 result CSVs and the three
  figure pairs under `results/` (#58); a second run leaves `git diff --exit-code results/` clean.
  Neither the repository nor CI has any fallback for a failed specification: a failure stays
  `failed: …` in `specs_run.csv` and shows as failed in the report's appendix (all twelve
  specifications are `ok` on this vintage).

### The report

`report/index.qmd` carries one narrative: question, headline result (the main table and the
event-time figure on the first screen), data, method, the side-by-side comparison with the paper,
robustness, extension, the three Phase 2 sections, deviations, limitations, reproduction and the
specification appendix. Rules the document follows:

- **No typed result.** Every prose number and table cell is a backticked `{python}` inline
  expression computed at render time through `anomaly_decay.reporting`; a test scans the source for
  typed decimal numbers outside code and inline expressions, and another rejects bare `{python}`
  tags (which render literally).
- **Sign-safe prose.** Wording that depends on the direction of a computed value comes from helpers
  (`direction`, `magnitude`, `comparison_word`, `pp_gap`, `factor`), and the multiple-testing
  reading is computed in a guarded cell, never asserted in static text.
- **Sources of truth.** The paper's values come from `config/original_paper.yaml`; the pinned
  vintage and the Ken French download date from `config/data.toml`; the Phase 2 parameters from
  `config/phase2_preregistration.yaml`; everything else from the files under the results directory.
- **Missing files are visible, not fatal.** The document loads results only through
  `reporting.load`/`load_optional`: an absent file renders a pending callout, so the same source
  renders from the dummy, fixture and full-data results directories (`ANOMALY_DECAY_RESULTS`).
  The committed full-data render has no pending callouts.

### Phase 2 analyses

- **Schemas and loader** (#64): `RESULTS_SCHEMAS` and `RESULTS_SORT_KEYS` gain `bootstrap.csv`,
  `multiple_testing.csv`, `tstat_summary.csv` and `vintage_comparison.csv` with exactly the O37
  columns, order and dtypes; `anomaly_decay.config.load_phase2_preregistration(path) -> Phase2Prereg`
  returns frozen dataclasses for the three sections and rejects unknown keys, missing keys, wrong
  types, invalid enum values, duplicate spec ids and undeclared `tstat_source` references.
- **Bootstrap inference** (#65, #66): a moving-block bootstrap by calendar month of the main
  specification's estimation sample — 12-month blocks from starts drawn uniformly in `0..T-L`
  without wrapping, concatenated and truncated to the sample length, every predictor's observations
  in a month moving together (a month drawn twice appears twice). Each replicate re-estimates the
  `main` coefficients and pooled-ratio decay with a within-predictor demeaned numpy estimator that
  agrees with `run_spec` to ~2e-14; intervals are percentile 95% intervals from seed 2027 and 2000
  replicates. `results/bootstrap.csv` carries `spec = bootstrap_main` (the pre-registered id).
  Runtime: ~1.9 ms per replicate at 101,567 rows; the full run is a few seconds, byte-identical
  across runs.
- **Multiple-testing lens** (#67–#69): two t-statistic sources — the original paper's, parsed from
  the OSAP SignalDoc `T-Stat` column as text (empty, non-numeric and non-finite cells are null and
  counted; the reported sign is kept), and the replicated t from the `op` long-short return over
  the predictor's in-sample months with sample `ddof=1`. Weak means strictly below the
  pre-registered 3.0 hurdle; missing t is excluded and counted, never imputed. The three
  pre-registered specifications (`mt_weak_original`, `mt_weak_replicated`, `mt_tstat_continuous`)
  add predictor-level interactions on the main sample; `results/multiple_testing.csv` and
  `results/tstat_summary.csv` are committed. Full universe: 24 of 212 original t statistics
  missing, 54 below the hurdle (median 4.00); replicated: none missing, 90 below (median 3.22).
- **Data-vintage sensitivity** (#70–#72): `fetch` accepts a vintage and writes any release into its
  own `data/raw/<N>/` under the same validation and manifest rules; `anomaly_decay.estimate.vintages`
  rebuilds each release's metadata and `op` panel, restricts to the predictor intersection and runs
  `main` plus `full_extension` through the frozen public interface. Google Drive refused the 2024.10
  download for 53 attempts over 18 rounds before serving it; the pre-registered preference order
  makes that the comparison release. The intersection is all 212 predictors and no sample-window
  metadata changed between the releases, so `results/vintage_comparison.csv` isolates revisions in
  the return series.

### CI

The four required checks (`lint`, `types`, `tests`, `reproduce-fixture`) run on the single
throttled runner (decision H4); the `tests` job fetches tags and requires the freeze. The report is
part of the `reproduce-fixture` gate, which renders HTML and PDF from fixture results on every PR,
and `make phase2` runs the three Phase 2 CLIs into a chosen results directory.

## Key decisions and reasoning

- **The write-up is part of the product, not a wrapper.** The report is developed against the
  committed fixture and must read correctly on real results later; sign-dependent wording and
  inline computation are what make that possible, and the two source tests keep the discipline
  from decaying once the document stops being edited every day.
- **The paper's numbers live in config, entered only by the maintainer (O25/O33).** The report
  recomputes the 26%/58% headline from the printed coefficients and the 0.582 in-sample mean, shows
  the rounded headline next to the recomputation, and records the printed-significance
  inconsistency for the post-sample coefficient rather than silently repeating the stars.
- **Comparison is side-by-side, not equality.** The predictor universe, portfolio construction and
  covariance treatment differ by design (recorded in the report's deviations list), so the paper's
  estimate and the replication's estimate are presented next to each other with the methods named.
- **Bootstrap months, not rows.** Resampling whole calendar months preserves cross-sectional
  correlation across predictors within a month; i.i.d. resampling would understate uncertainty. A
  replicate with a degenerate term raises rather than being skipped silently, and a classic-bug
  test pins contiguous blocks against an i.i.d. control.
- **The bootstrap's point estimate comes from an independent path.** The fast estimator is the
  within-predictor demeaned OLS, validated against `run_spec` at 1e-10; `estimate` in
  `bootstrap.csv` is not copied from `main.csv`.
- **Missing t statistics are excluded and counted, never imputed.** `n_excluded_missing_t` is part
  of the results contract and the hurdle counts are reported per source.
- **The vintage comparison holds the predictor universe fixed.** Only signals present in both
  releases enter, so the comparison isolates data revision rather than composition change; the
  predictor dates were also compared and happened to be identical.
- **Determinism claim scoped per platform (O34).** Reruns on one machine are byte-identical; across
  operating systems the linear-algebra libraries differ, so the committed results are produced on
  macOS and agree across platforms to about 1e-13 relative, with tests comparing at 1e-10. The
  report states this in the reproduction section.
- **Adding the Phase 2 schemas required owning the fallout.** Three existing tests iterate a
  hard-coded eight-file Phase 1 fixture (`tests/test_io.py`, `tests/test_results_fixture.py`,
  `tests/test_reporting.py`) and the documented-file set in `tests/test_schemas.py`; they were
  scoped to an explicit `FIXTURE_FILES` tuple rather than adding fixture bytes, which the freeze
  forbids.

## Interfaces for future epics

- **Results contract.** Twelve files with schemas, sort keys and `write_results`/`read_results`;
  the Phase 2 files add no new I/O path.
- **Config.** `anomaly_decay.config.load_phase2_preregistration` returns frozen dataclasses for the
  three sections; strict validation raises `ConfigError`.
- **Reporting helpers.** `load`, `load_optional`, `lookup`, `num`, `pct`, `tstat`, `direction`,
  `magnitude`, `ratio`, `ratio_pct`, `comparison_word`, `pp_gap`, `factor`, `month_label`, `table`,
  `figure`, `figure_block`, `original_paper`, `data_config`, `phase2_preregistration`.
- **CLIs.** `python -m anomaly_decay.estimate.bootstrap`, `...multiple_testing`, `...vintages`, and
  `python -m anomaly_decay.data.fetch --osap --vintage N`; the report consumes only their result
  files.
- **Commands.** `make reproduce` runs the full pipeline and renders the report from `results/`;
  `make phase2` runs the three Phase 2 analyses; `make reproduce-phase2` chains them and re-renders.

## Deviations from the plan

- **The first screen moved.** The final pass requires the main table and the event-time figure on
  the first screen, so the headline section sits directly after the question instead of inside the
  paper-comparison section.
- **Fixture results remain fictional.** The committed results fixture still drives the offline gate
  and report development renders; its numbers are marked as a toolchain check and are not
  interpreted. Phase 2 files are absent there and render pending callouts in the fixture document.
- **The window deviations are reported, not corrected.** OSAP's year-granularity sample windows
  differ from the original papers' monthly windows in the documented hand-checked cases; the rule
  stays "OSAP SignalDoc years are the rule" (decision O38).
- **The comparison vintage is whichever release can be downloaded.** The pre-registered preference
  order (`202410, 202408, 2023`) was exercised in full: 2024.10 succeeded on the 18th round after
  53 refusals, so the later fallbacks were never attempted.
- **Phase 2 is labelled `epic:04` on GitHub** but shipped inside this epic's report; this document
  covers it because it is part of the integrated product.
- **The PRD's flagless headline command predates the CLI's required arguments.** A fresh-clone
  check uses the explicit-argument equivalent (`--panel --specs --results-dir`); the frozen CLI was
  not changed.

## Known limitations and deferred items

- **The multiple-testing lens is a lens, not a formal correction.** It reports hurdle counts and
  pre-registered interactions; it does not adjust for the full search that produced the published
  literature, and the report says so. On this vintage the replicated-t post-publication interaction
  is positive (weak predictors decay *less* post-publication), which the report states as computed;
  the original-paper-t interactions are small with large standard errors.
- **The bootstrap covers monthly-block sampling variation only** and keeps the pooled-ratio
  in-sample mean fixed per replicate, as the pre-registered definition does.
- **The vintage comparison has one comparison release** (2024.10) and 212 common predictors; the
  metadata windows are identical between the releases, so only return revisions are measured.
- **No trading costs.** Returns are gross long-short portfolio returns; cost-based explanations of
  decay are out of scope.
- **Reconstruction differences persist.** OSAP rebuilds each predictor following its original
  paper; the hand-checked examples in the report are illustrative, not exhaustive.
- **Option-based predictors end earlier in the extension** because their inputs require listed
  options; the extension's last month is the vintage's, not a guarantee per predictor.
- **Cross-platform float agreement is approximate.** Byte-identical reruns hold per platform;
  committed results were produced on macOS.
- **The runner is a single throttled machine with no automatic restart** (decision H4; the launchd
  job has no `KeepAlive`). It crashed once during this wave and blocked all CI until it was
  restarted manually; a watchdog is the recommended follow-up.
