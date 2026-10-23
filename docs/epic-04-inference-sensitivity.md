# Epic 04 — Inference and sensitivity

## Goal

Put bounds and sensitivity checks around the headline decay estimates before the replication is
released: a pre-registered moving-block bootstrap of the main coefficients and pooled-ratio decay, a
pre-registered multiple-testing lens on the relation between in-sample evidence and post-publication
decay, and a like-for-like comparison against an earlier OSAP release. All three analyses were
registered in `config/phase2_preregistration.yaml` before the first full-data estimation and frozen
with the Phase 1 specifications by the annotated tag `specs-frozen`, so their samples, parameters
and reporting rules could not follow the results. The epic also fixes the results contract for the
four Phase 2 files and records how the analysis is reproduced.

## How it works

### Contract and gates

- `anomaly_decay/schemas.py` gains `bootstrap.csv`, `multiple_testing.csv`, `tstat_summary.csv` and
  `vintage_comparison.csv` in `RESULTS_SCHEMAS`, with the exact columns, order and dtypes fixed
  before the run; `anomaly_decay/io.py` gains their sort keys. The Phase 2 files therefore go
  through the same `write_results`/`read_results` path as Phase 1, with sorted key columns, fixed
  column order and LF line endings.
- `anomaly_decay.config.load_phase2_preregistration(path)` returns frozen dataclasses
  (`BootstrapPrereg`, `MultipleTestingPrereg` with its per-spec entries, `VintagesPrereg`) and
  rejects unknown keys, missing keys, wrong types, invalid enum values, duplicate spec ids and a
  spec that references an undeclared `tstat_source` with `ConfigError`. The `version` field is
  accepted as any positive integer; every other value is pinned to the registered text.
- Every result below was produced after `specs-frozen` (annotated tag `6e2bcaeb` on `ce0e0d5`) and
  after the first full-data Phase 1 run (#58), with byte-identical reruns and the Makefile's
  thread pins.

### Bootstrap inference (#65, #66)

`anomaly_decay/estimate/bootstrap.py` implements the registered scheme only:

- The estimation sample is the `main` specification's sample rebuilt from its specification
  filters: 198 predictors and 101,567 rows. Each replicate draws `ceil(T / 12)` start indices
  uniformly with replacement from `0..T-12` inclusive, concatenates the 12-month blocks and
  truncates to `T` months.
- Blocks never wrap around the end of the sample: a block is a contiguous run of calendar months
  inside the sample, so the seam is never stitched from the two ends. A month drawn twice appears
  twice.
- Every predictor's observations in a drawn month move together, preserving cross-sectional
  correlation; the resample is by month, not by row.
- Each replicate re-estimates the two window coefficients and the pooled-ratio decay with a numpy
  within-predictor demeaned OLS, the same transformation `PanelOLS(entity_effects=True)` applies.
  The estimator is validated against `run_spec`/`decay_table` on the fixture panel at 1e-10 and
  agrees to about 1e-14; the `estimate` column is produced by this path, not copied from
  `main.csv`/`decay.csv`.
- The pooled-ratio decay of a replicate uses that replicate's own in-sample mean; a replicate with
  no in-sample rows or a zero in-sample mean yields a null for that statistic, which is excluded
  from its interval. A replicate whose terms have no variation raises `BootstrapError` with the
  replicate index instead of being skipped silently.
- Intervals are percentile 95% intervals from `numpy.random.default_rng(2027)`, 2,000 replicates,
  block length 12, with one `integers` draw per replicate. `bootstrap.csv.spec` is the registered
  analysis id `bootstrap_main`; the base specification `main` is fixed by the pre-registration.

The full run takes a few seconds at 101,567 rows (two timed runs: 3.70 s and 4.00 s including panel
load) and two consecutive runs are byte-identical. The four intervals, quoted from
`results/bootstrap.csv` rounded to six decimals (the file carries full double precision):

| statistic | estimate | boot_mean | boot_se | ci_low | ci_high |
|---|---|---|---|---|---|
| coef_post_sample | -0.239119 | -0.235647 | 0.074941 | -0.371337 | -0.078753 |
| coef_post_publication | -0.336789 | -0.327667 | 0.081259 | -0.479782 | -0.165277 |
| decay_pct_post_sample | -35.156223 | -34.753131 | 10.984434 | -54.137009 | -11.560657 |
| decay_pct_post_publication | -49.516068 | -48.124316 | 11.062481 | -68.547326 | -25.669822 |

`reps = 2000`, `block_length = 12` and `seed = 2027` in every row. The `estimate` values agree with
`results/main.csv` and `results/decay.csv` to about 2e-14 (coefficients) and 3e-12 (pooled decay),
i.e. at the project's 1e-10 comparison tolerance. The bootstrap intervals quantify monthly-block
sampling variation; they do not describe how the estimate was selected.

### Multiple-testing lens (#67–#69)

`anomaly_decay/estimate/multiple_testing.py` builds two t-statistic sources per predictor:

- **`original_paper`**: the OSAP SignalDoc `T-Stat` column, read as text (`infer_schema_length=0`).
  Whitespace is stripped; empty, whitespace-only, non-numeric and non-finite (`nan`/`inf`) cells
  are null and counted; a missing `T-Stat` column or a predictor absent from the document is also
  null and counted; duplicate predictor rows raise `TstatError`. The reported sign is kept as
  reported — no absolute value — so a negative t is below the hurdle. Only
  `Cat.Signal == "Predictor"` rows are considered when that column exists.
- **`in_sample_replicated`**: `mean / (sd / sqrt(n))` of the `op` long-short return over the
  predictor's own in-sample months, with sample `ddof=1`. It is null for fewer than two returns or
  a zero standard deviation. Missing values in either source follow the registered
  `missing_tstat: exclude_and_count` rule.

Weak means **strictly below** the registered hurdle of 3.0; t = 3.0 is not weak. `weak_original`
and `weak_replicated` are null exactly when their t is null. Missing t statistics are never
imputed. `tstat_summary` describes the predictors present in the panel passed to `tstats` (212 on
the full run); the per-specification estimation universes are separate and visible in
`n_excluded_missing_t`.

The three registered specifications extend the `main` sample with predictor-level interactions
(the interaction's main effect is absorbed by the predictor effects and is not added as a term):
`post_sample × weak` and `post_publication × weak` for each t source, and the continuous
`post_sample × t` / `post_publication × t` analogue of McLean and Pontiff's Table II, column 4.
They use the base specification's covariance, so standard errors are month-clustered throughout.

Counts as written in `results/tstat_summary.csv` (medians to three decimals, hurdle exact):

| tstat_source | n_predictors | n_missing | n_below_hurdle | median_t | hurdle_t |
|---|---|---|---|---|---|
| in_sample_replicated | 212 | 0 | 90 | 3.217 | 3.0 |
| original_paper | 212 | 24 | 54 | 3.995 | 3.0 |

The interactions, quoted from `results/multiple_testing.csv` (coefficients, SEs to four decimals,
t to two; counts exact):

| spec | tstat_source | term | coef | se | t | n_predictors | n_obs | n_weak | n_excluded_missing_t |
|---|---|---|---|---|---|---|---|---|---|
| mt_weak_original | original_paper | post_sample | -0.2099 | 0.0786 | -2.67 | 174 | 90374 | 50 | 24 |
| mt_weak_original | original_paper | post_publication | -0.3651 | 0.0875 | -4.17 | 174 | 90374 | 50 | 24 |
| mt_weak_original | original_paper | post_sample_x_weak | -0.1103 | 0.0996 | -1.11 | 174 | 90374 | 50 | 24 |
| mt_weak_original | original_paper | post_publication_x_weak | 0.0586 | 0.1036 | 0.57 | 174 | 90374 | 50 | 24 |
| mt_weak_replicated | in_sample_replicated | post_sample | -0.3012 | 0.0790 | -3.81 | 198 | 101567 | 85 | 0 |
| mt_weak_replicated | in_sample_replicated | post_publication | -0.4501 | 0.0893 | -5.04 | 198 | 101567 | 85 | 0 |
| mt_weak_replicated | in_sample_replicated | post_sample_x_weak | 0.1460 | 0.0848 | 1.72 | 198 | 101567 | 85 | 0 |
| mt_weak_replicated | in_sample_replicated | post_publication_x_weak | 0.2605 | 0.0714 | 3.65 | 198 | 101567 | 85 | 0 |
| mt_tstat_continuous | in_sample_replicated | post_sample | -0.1432 | 0.1103 | -1.30 | 198 | 101567 | 85 | 0 |
| mt_tstat_continuous | in_sample_replicated | post_publication | -0.1450 | 0.1036 | -1.40 | 198 | 101567 | 85 | 0 |
| mt_tstat_continuous | in_sample_replicated | post_sample_x_tstat | -0.0245 | 0.0178 | -1.38 | 198 | 101567 | 85 | 0 |
| mt_tstat_continuous | in_sample_replicated | post_publication_x_tstat | -0.0505 | 0.0160 | -3.16 | 198 | 101567 | 85 | 0 |

As the report reads them: a negative interaction on a window term means the weak group's window
shift is more negative than the comparison group's — weaker evidence going with deeper decay; a
positive interaction means the opposite. Taking the estimates as computed, the original-paper-t
interactions are -0.1103 (t -1.11) for the post-sample window and 0.0586 (t 0.57) for
post-publication, both small with large standard errors. In the replicated-t specification the
weak group's post-publication shift is higher than the comparison group's by 0.2605 percent per
month (t 3.65), and the continuous post-publication interaction is -0.0505 per unit of t
(t -3.16). The report states that on this vintage the replicated-t interaction is opposite to the
weaker-evidence hypothesis: weaker replicated evidence goes with shallower post-publication decay.
The `n_weak` column counts estimation-sample predictors strictly below the hurdle; for the
continuous specification it is descriptive because the regressor is the continuous t.

### Data-vintage sensitivity (#70–#72)

`anomaly_decay.data.fetch.fetch_osap` accepts a `vintage` and a `portfolios` override and writes
any release into its own `data/raw/<vintage>/` directory under the same validation, canonical
writer and manifest rules as the primary release. A non-primary release defaults to the signal doc
plus `op` only (the decile files are about 100 MB each and the comparison does not need them); an
explicit `--portfolios` wins and is de-duplicated. Legacy SignalDoc headers go through the explicit
`SIGNAL_DOC_RENAMES` table, never fuzzy matching; for the registered vintages the table is empty and
an unmapped rename fails loudly with the observed columns. `config/data.toml` records the
pre-registered comparison preference `[202410, 202408, 2023]`; the primary vintage and the Ken
French entries are unchanged.

Google Drive refused the 2024.10 release for several hours, always by serving its "quota
exceeded" HTML page instead of the CSV; the frame validation caught it as a missing `Acronym`
column and the documented backoff retried. The preference order was retried on a 20-minute loop.
Attempts at 2024.10 (timestamps ET):

| attempt | time | outcome |
|---|---|---|
| initial probe | 03:22:37 | failed |
| round 1 | 03:26:55 | failed |
| round 2 | 03:51:08 | failed |
| round 3 | 04:14:51 | failed |
| round 4 | 04:41:03 | failed |
| round 5 | 05:05:10 | failed |
| round 6 | 05:28:40 | failed |
| round 7 | 05:53:03 | failed |
| round 8 | 06:16:32 | failed |
| round 9 | 06:39:58 | failed |
| round 10 | 07:03:28 | failed |
| round 11 | 07:10:04 | failed (loop relaunched, next round about 3 minutes later) |
| round 12 | 07:33:32 | failed |
| round 13 | 07:56:58 | failed |
| round 14 | 08:20:19 | failed |
| round 15 | 08:43:43 | failed |
| round 16 | 09:07:05 | failed |
| round 17 | 09:30:59 | failed |
| round 18 | 09:55:26 | **success**, file written 09:56:10 |

Across all three preferred releases that is 53 attempts over 18 rounds. The successful 2024.10
cache is the 331-row, 28-column signal doc (the 2025.10 header without the 2025 citation column)
and `port_op.csv` with 1,207,990 rows, 212 predictors with `op` long-short rows, and returns
through 2023-12-29; both files are in `data/manifest.lock` and `make verify-hashes` prints
`all hashes match`. 2024.08 and 2023 were not attempted once 2024.10 succeeded: the
pre-registration takes the first release in preference order that downloads and passes
validation.

`anomaly_decay/estimate/vintages.py` rebuilds each release's metadata (`build_metadata`) and `op`
panel (`build_panel(..., vintage=N)`), keeps the predictors present in every requested release,
runs `main` and `full_extension` through `run_spec`/`decay_table` on each release's own sample,
and writes `vintage_comparison.csv` with `predictor_set = "intersection"`. Metadata date changes
(sample start, sample end, publication year) are reported relative to the primary release and also
restricted to the intersection. The CLI validates `--vintages` against the primary vintage plus
`compare_vintages` and requires at least two releases.

The comparison itself (quoted from `results/vintage_comparison.csv`; coefficients and SEs to four
decimals, decay to one):

| spec | vintage | term | coef | se | decay_pct | n_predictors | n_obs |
|---|---|---|---|---|---|---|---|
| main | 202410 | post_sample | -0.2329 | 0.0685 | -34.3 | 198 | 101488 |
| main | 202410 | post_publication | -0.3233 | 0.0741 | -47.7 | 198 | 101488 |
| main | 202510 | post_sample | -0.2391 | 0.0678 | -35.2 | 198 | 101567 |
| main | 202510 | post_publication | -0.3368 | 0.0738 | -49.5 | 198 | 101567 |
| full_extension | 202410 | post_sample | -0.2374 | 0.0647 | -35.5 | 212 | 133673 |
| full_extension | 202410 | post_publication | -0.3552 | 0.0582 | -53.1 | 212 | 133673 |
| full_extension | 202510 | post_sample | -0.2466 | 0.0641 | -36.8 | 212 | 136338 |
| full_extension | 202510 | post_publication | -0.3695 | 0.0569 | -55.1 | 212 | 136338 |

The intersection is all 212 predictors in both releases; no predictor was dropped. Sample start,
sample end and publication year are identical for all 212 predictors, so the comparison isolates
revisions in the return series plus the additional months of coverage (the 2024.10 release ends in
December 2023, the 2025.10 release in December 2024). The 2025.10 `main` rows reproduce
`results/main.csv` exactly.

## Key decisions and reasoning

- **Resample months, not rows.** An i.i.d. row bootstrap would break the cross-sectional correlation
  of predictors within a month and understate uncertainty. Whole calendar months are drawn, and
  every predictor's observation in a drawn month moves with the month.
- **No wrap, and truncate at the end.** Start indices are restricted to `0..T-12` so every block is
  a contiguous calendar run inside the sample; the concatenation is truncated to the sample length.
  This makes "a month drawn twice appears twice" and "no month is invented at the seam" both true.
  The named classic-bug test `test_resampled_months_come_in_contiguous_blocks` checks that every
  drawn block is a contiguous `+1` run and uses an i.i.d. draw over the same range as the failing
  control.
- **The point estimate is not copied.** The bootstrap's `estimate` comes from its own fast
  estimator, validated against `run_spec` and `decay_table`; agreement with the committed Phase 1
  files is a check, not the source.
- **A degenerate replicate is an error.** Silently skipping a replicate would bias an interval
  without a trace; raising with the replicate index makes the condition visible. With about a
  thousand months and 12-month blocks it is not reachable in the observed sample.
- **Parse the original t as reported.** The SignalDoc cell is text, and its sign is part of the
  reported statistic; the parser therefore keeps the sign, treats unparseable and non-finite cells
  as missing, and never substitutes zero. The missing rule (`exclude_and_count`) drops a predictor
  only from the specification that needs its t, and the count is reported.
- **Weak is a strict inequality at the registered hurdle.** t = 3.0 is not weak, matching the
  pre-registration's "strictly below"; the replication uses the same rule for both sources.
- **Reuse the frozen estimator's covariance.** The multiple-testing runner calls the base
  specification's private `_filter_sample` and `_fit` rather than copying the month-clustered
  covariance; `run_spec`'s extra-term vocabulary is closed, and duplicating the estimator would
  risk divergence. This follows the sibling-module pattern already used by robustness and
  extension, and leaves the frozen `main.py` untouched.
- **Interactions at the predictor level.** The weak flag and the t statistic are predictor
  attributes, so their main effects are collinear with the predictor fixed effects and are not
  separate terms; the window terms are the comparison group's shifts and the interactions are the
  differences.
- **Fix the predictor universe for the vintage comparison.** Restricting to the intersection means
  the comparison measures data revision rather than composition change. Date-change reporting is
  restricted the same way so it describes the sample the coefficients use.
- **Compare within each vintage's own sample.** `decay_pct` is the pooled-ratio decay computed
  within each release, so the comparison shows what a release update alone moves with the
  predictor universe held fixed; it is not a single pooled decay across releases.
- **Report the results as computed.** The replicated-t interactions are positive and the
  continuous post-publication interaction is negative, which is the opposite of the registered
  weaker-evidence direction for the weak split; the report states that reading rather than
  reframing the pre-registered test.

## Interfaces for future epics

- **Results contract.** `RESULTS_SCHEMAS` and `RESULTS_SORT_KEYS` cover the four Phase 2 files:
  `bootstrap.csv` (key `spec, statistic`), `multiple_testing.csv` (`spec, term`),
  `tstat_summary.csv` (`tstat_source`) and `vintage_comparison.csv` (`spec, vintage, term`). They
  read and write through `anomaly_decay.io` like every Phase 1 file.
- **Config.** `load_phase2_preregistration(path) -> Phase2Prereg` exposes frozen dataclasses for
  the three registered analyses; `ConfigError` is the only validation failure.
- **Estimators.**
  - `anomaly_decay.estimate.bootstrap`: `estimation_sample`, `draw_month_positions`,
    `bootstrap_main`/`run_bootstrap` and the CLI.
  - `anomaly_decay.estimate.multiple_testing`: `tstats`, `tstat_summary`,
    `run_multiple_testing_spec`, `run_multiple_testing` and the CLI.
  - `anomaly_decay.estimate.vintages`: `build_vintage_comparison`, `metadata_date_changes` and the
    CLI.
- **Fetch.** `fetch_osap(..., vintage=N, portfolios=...)` writes any release into its own cache
  directory; `load_compare_vintages(config_path)` reads the preference order.
- **CLIs and targets.** `python -m anomaly_decay.estimate.bootstrap`,
  `...multiple_testing` and `...vintages` accept their registered defaults; `make phase2` runs all
  three into `results/` (after the pinned panel target), and `make reproduce-phase2` chains them
  with the report. The report reads the files only through `anomaly_decay.reporting`.
- **Reproduction.** Every Phase 2 file can be rebuilt from the cache by the three CLIs into a
  chosen results directory; `make phase2` is the documented path and the report consumes only
  their output files.

## Deviations from the plan

- **The comparison release is whichever release downloads.** The pre-registered preference order
  was exercised from the start; 2024.10 succeeded on the 18th round, so the later fallbacks were
  not attempted after that. The analysis is therefore on 2024.10, not on 2024.08 or 2023.
- **`tstat_summary` describes the panel-wide universe.** The schema has no universe column, so the
  summary covers the predictors passed to `tstats` (212), while the per-specification samples
  exclude predictors with a missing t (`mt_weak_original`: 174 predictors, 24 excluded) or use the
  full main sample (`mt_weak_replicated` and `mt_tstat_continuous`: 198, 0 excluded). The
  estimation universes are recorded in `n_weak` and `n_excluded_missing_t` rather than folded into
  the summary.
- **The continuous specification's `n_weak` is descriptive.** The regressor is the continuous t
  statistic, so there is no weak/strong split in that regression; the column is still written for
  schema consistency.
- **A non-primary vintage fetches the signal doc and `op` only.** The comparison needs nothing
  else; the decile files are large and were left out of the 2024.10 cache.
- **The replicated t is null for fewer than two observations or zero variation.** This is a
  documented edge rule (excluded and counted); no predictor in the 2025.10 panel hits it.
- **`bootstrap.csv.spec` is the registered analysis id, not the base spec id.** The base
  specification is fixed by the registration and appears in `config/phase2_preregistration.yaml`,
  not in the results row.
- **The vintage comparison adds coverage months.** The releases do not end in the same month
  (2023-12 versus 2024-12), so `n_obs` differs between the two vintages even on the fixed
  intersection; that difference is part of what a release update moves and is visible in the
  results table.

## Known limitations and deferred items

- **The bootstrap covers monthly-block sampling variation only.** It resamples months, not
  specifications or the predictor universe, and the pooled-ratio decay keeps the in-sample mean
  fixed within each replicate, exactly as the registered definition does. It does not quantify
  model-selection uncertainty.
- **The multiple-testing lens is a lens, not a formal correction.** It reports hurdle counts and
  the pre-registered interactions; it does not adjust for the full search that produced the
  published literature, and the report says so. The original-paper t statistics are taken as
  reported in the SignalDoc and may embed the original authors' own sample and method choices; the
  replicated t is computed from this project's `op` long-short return over the predictor's
  in-sample months.
- **The multiple-testing samples are not identical.** `mt_weak_original` loses the 24 predictors
  with a missing original t (174 predictors, 90,374 observations) while the replicated-t
  specifications use 198 predictors and 101,567 observations, so their window coefficients are not
  directly comparable across sources.
- **The vintage comparison has one comparison release and no composition change.** All 212
  predictors are common to both signal docs, and sample start, sample end and publication year are
  identical; the comparison therefore measures return revisions and the extra coverage months, not
  revision of the predictor set or the windows.
- **No trading costs or capacity analysis.** The multiple-testing and vintage analyses operate on
  the same gross long-short returns as Phase 1; the paper's question is about predictability, and
  cost-based explanations remain out of scope.
- **Cross-platform float agreement is approximate.** Byte-identical reruns hold on the machine that
  produced the committed files (macOS); across operating systems the linear-algebra libraries
  differ, so the same computation agrees to about 1e-13 relative and tests compare at 1e-10.
- **Nothing in the registered analyses is deferred.** The three Phase 2 analyses and their results
  files are complete on this vintage; no follow-up analysis is registered.
