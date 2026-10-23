# Epic 06 — Companion page and RA memo

## Goal

Deliver Phase 3 of the product: an interactive static companion page where a reader picks any
predictor and sees its in-sample, post-sample and post-publication mean long-short returns together
with its event-time profile (FR-3.1), and a two-page research-assistant memo in the format a busy
professor reads first (FR-3.2). Both live in the published Quarto site, so the report, the companion
page and the memo are one build from the same committed `results/`. The epic also adds the two
descriptive results files that make a per-predictor view possible without republishing monthly OSAP
series, and it closes with the planned v1.1.0 release.

## How it works

### Website structure (#109)

`report/_quarto.yml` turns the report project into a website:

- `project.type: website` with `output-dir: _site` and a render list of `index.qmd`,
  `companion.qmd` and `memo.qmd`;
- a navbar with Report (`index.html`), Explore a predictor (`companion.html`), RA memo (PDF)
  (`memo.pdf`), the `format-links` control for the report's HTML and PDF, and a GitHub link;
- project-level `format` declaring `html` and `typst`; `companion.qmd` overrides with
  `format: html`, `memo.qmd` with `format: typst`.

The per-document scalar override is the mechanism Quarto 1.10.19 accepts. A document-level format
array is rejected by front-matter validation, so the project declares both formats and each page
selects one. The build produces `index.html`, `index.pdf`, `companion.html` and `memo.pdf`, and no
`companion.pdf` or `memo.html`.

`make report` renders the project. `make site` runs that render and then
`python -m anomaly_decay.site --results-dir results --out report/_site/data`, which writes the JSON
the companion page fetches at run time. The Pages workflow builds with `make site` and uploads
`report/_site`; it still runs only on `release: published` and `workflow_dispatch`. Fixture renders
continue to write to `data/processed/fixture/_site/` and never touch `report/_site/`, so the
published site is always built from the committed full-data results.

#123 added an **Explore and summary** section to the report, after the headline figure, linking
`companion.html` and `memo.pdf` with relative paths that resolve both on Pages and in a local
render, and replaced the README's release placeholder with the live site, companion and memo links.
The report PDF is 14 pages after that polish, down from 15.

### Descriptive data products (#110, #112)

Two files complete the results contract. Both are additions to `RESULTS_SCHEMAS` and
`RESULTS_SORT_KEYS`, so they read and write through `anomaly_decay.io` like every other result:
sorted key columns, fixed column order, LF endings, fixed float formatting.

| File | Key | Columns |
|---|---|---|
| `results/predictor_info.csv` | `signalname` | `signalname, authors, journal, pub_year, sample_start_year, sample_end_year, clear, cat_economic` |
| `results/predictor_event_time.csv` | `signalname, anchor, event_year` | `signalname, anchor, event_year, mean_ret, n_months` |

`anomaly_decay/estimate/predictor_profiles.py` builds them from the committed panel and metadata:

- `predictor_info(metadata)` validates the predictor table, selects the bibliographic fields the site
  shows and casts the three year columns to 64-bit integers.
- `predictor_event_time(panel)` uses only the `op` long-short portfolio, keeps event months −60…119
  (event years −5…9), and groups by predictor, anchor (`sample_end` / `publication`) and event year.
  The event year is `floor(event_month / 12)`, so month −1 and month −12 both belong to year −1 and
  month −13 to year −2. `mean_ret` is the mean monthly long-short return in percent and `n_months`
  counts months with a non-null return.

The `descriptive` Make stage calls the CLI after the Phase 1 descriptive tables. The full-data files
were produced on the Mac from the verified cache with the Makefile's thread environment and committed
alone, with no other file under `results/` changed:

- `predictor_info.csv`: 212 predictors, 165 clear and 47 likely, publication years 1973–2016, 25
  journals and 35 economic categories.
- `predictor_event_time.csv`: 6,306 rows covering 212 predictors at both anchors and every event year
  −5…9 where data exist — 3,167 `sample_end` rows over 37,913 months and 3,139 `publication` rows
  over 37,568 months. The full grid would be 212 × 2 × 15 = 6,360 cells, so 54 predictor-anchor-years
  have no usable month and are omitted rather than reported as zero.
- A second run into a temporary directory is byte-identical; the two SHA-256 values are recorded in
  #112, and `tests/test_full_reproduction.py` now rebuilds both files so the pipeline's output set
  keeps matching `results/*.csv`.

Yearly aggregation is deliberate. The companion page needs a readable event-time profile, not a
second copy of the monthly OSAP panel; event years keep the site on derived statistics and keep the
published JSON small.

### Companion page (#111)

`anomaly_decay/site.py` reads `predictor_info.csv`, `predictor_summary.csv` and
`predictor_event_time.csv` and writes, from `results/` only:

- `data/predictors.json` — the index the selector reads, one entry per predictor with `signalname`,
  `authors`, `journal`, `pub_year`, `sample_start_year`, `sample_end_year` and `clear`;
- `data/predictors/<signalname>.json` — the same header plus `cat_economic`, the three window cells
  and the two event-year series. Each window carries `mean_ret`, `se_mean` and `n_months`; each
  event-year point carries `event_year`, `mean_ret` and `n_months`.

The builder validates that the three files describe one universe: duplicate signal names or result
keys, a predictor in the summary or event file that is absent from the index, unknown windows or
anchors, and signal names that cannot be used as file names all raise `SiteError` instead of
producing a broken site. Floats are rounded once to six decimals, negative zero is normalised, keys
are sorted and `allow_nan=False`, so a rebuild from the same results is byte-identical and every file
is valid JSON. Stale `predictors/*.json` files are removed first and `predictors.json` is written
last, so the index never advertises a missing profile.

`report/companion.qmd` is an Observable JS page built into Quarto, with no extra JavaScript
dependencies. It fetches the index, filters the 212 names with `Inputs.search`, selects one with
`Inputs.select` (a `?predictor=<name>` query pre-fills the search and the selection), fetches that
predictor's profile and renders:

- a header with authors, journal, publication year, sample years, clear/likely status and economic
  category;
- a table of the three window means with standard errors and month counts;
- a dot chart of the three means with ±1 SE intervals (Okabe–Ito colours, zero rule);
- two event-year panels, one per anchor, with a zero line and a dashed year-0 marker. Year −1 is the
  last in-sample year for `sample_end` and the publication year for `publication`; year 0 is the
  first year after the event.

The page states the units (percent per month), that the windows use the default convention and the
`op` portfolio, and that these are descriptive statistics rather than the pre-registered
specification estimates.

`tests/test_site.py` has two variants. The full-data variant is the PRD Phase 3 check: it asserts
that every predictor in `predictor_info.csv` has a profile file, that the index lists exactly the
same set, and that every JSON value equals the source CSV rounded to six decimals. The fixture
variant builds a three-predictor results directory from the committed fictional fixture, so the site
path runs in CI before the full-data files existed. The module has 11 tests; the full-data ones now
run because #112 committed the inputs.

The site build was exercised against the full data: 212 profile files plus the index, 452,311 bytes,
byte-identical on a rebuild, with the JSON step taking well under a second (the Quarto render
dominates `make site`). The page was checked in headless Chrome both under a plain
`python -m http.server` root and under the `/anomaly-decay/` base path: searching "mom" narrows the
selector to the 15 momentum predictors, and `AM`, `Mom12m` and `Size` show window means, standard
errors and month counts equal to `results/predictor_summary.csv` at three decimals. A search with no
match shows a plain "No predictor matches your search" message.

### RA memo (#113, #114)

`report/memo.qmd` replaces the stub with the memo, rendered only to Typst PDF at
`report/_site/memo.pdf`. The memo is US Letter at 10pt with 0.7in horizontal and 0.6in
vertical margins, and it fits two pages. The header carries the title, the subtitle, the author
(`Moksh Siruvani`), the render date (`date: today`) and links to the live report and the repository.

The memo follows the format in the issue:

- **Question** — do long-short anomaly returns fall after the sample ends and further after
  publication, and what does a replication on OSAP data show?
- **Data** — the pinned release, predictor counts, coverage months, units and the estimation design
  in one short paragraph.
- **Result** — one table comparing the replication with McLean and Pontiff's Table II, first column:
  both window coefficients with standard errors, both pooled-ratio declines, both normalized
  declines, both pre-registered bootstrap intervals, and the predictor and observation counts. The
  accompanying paragraph interprets the magnitudes and states that the two exercises share the
  measured object but not their inputs.
- **Robustness and Phase 2** — one paragraph: the coefficient range across the nine pre-registered
  robustness rows, the earlier-vintage re-estimation, and the continuous interaction analogue of the
  paper's Table II fourth column.
- **Caveats** — one point each for publication timing, the reconstructed predictor universe, the
  standard-error method, gross returns, the absence of a causal claim, and the single vintage.
- **Next steps** — proposals for a WRDS rebuild of a few predictors, a trading-cost adjustment and an
  extension to non-US markets.

Every number is computed at render time from `results/` and `config/original_paper.yaml` through
`anomaly_decay.reporting`; the earlier-vintage label and both interval bounds are read from the
results files rather than typed. `tests/test_memo.py` strips the front matter (the date line), code
fences, inline expressions, headings, link targets and parenthetical citation years, then asserts
that no digit remains in the prose. The memo reports, from the committed full-data results:

- post-sample and post-publication returns of −0.239 (SE 0.068) and −0.337 (SE 0.074) percent per
  month;
- pooled-ratio declines of −35.2% and −49.5% against the paper's −25.8% and −57.9% (headline 26% and
  58%), a gap of 9.4 and 8.4 percentage points;
- normalized declines of −28.1% and −48.1%;
- bootstrap intervals of −54.1 to −11.6 and −68.5 to −25.7 percent, neither including zero;
- robustness coefficients from −0.283 to −0.209 (post-sample) and −0.481 to −0.176
  (post-publication); a main-specification post-publication decline of −47.7% on the 2024.10 release
  against −49.5% on 2025.10; and a continuous post-publication interaction of −0.051 (t −3.16)
  against the paper's −0.063 (SE 0.018), with the same sign.

The two-page budget is enforced twice. The document ends with a Typst assertion,
`counter(page).final().first() <= 2`, so any render that produces a longer memo fails instead of
shipping it — this covers `make site`, `make report` and the CI fixture render. `tests/test_memo.py`
also counts page objects in the built PDF bytes without a new dependency: it counts `/Type /Page`
(accepting the spaced and unspaced spelling) and rejects anything followed by a letter, which keeps
out both the `/Pages` tree node and Typst's `/PageLabel` objects. The counter is checked against a
synthetic two-page document, against the `/Count` in each rendered report PDF's own page tree, and
against both built memo PDFs (full-data and fixture), which must each have at most two pages.

## Phase 3 definition of done

The PRD asks for three things, and each is met on the committed results:

| Requirement | Evidence |
|---|---|
| `make site` builds the companion page | `make site` renders `index.html`, `index.pdf`, `companion.html` and `memo.pdf`, then writes 212 profile JSON files plus `predictors.json` (452,311 bytes) to `report/_site/data/`. `companion.html` fetches the predictor it needs from those files at run time. |
| `uv run pytest -q tests/test_site.py` checks that every predictor in the panel has a data file in the site build | 11 tests pass. The full-data variant asserts that every one of the 212 predictors in `predictor_info.csv` has a file, that the index lists the same set, and that every value equals `predictor_summary.csv` or `predictor_event_time.csv` rounded to six decimals; the fixture variant runs the same code path in CI before the full-data files exist. |
| The memo PDF exists at `report/_site/memo.pdf` | `pdfinfo` reports 2 pages and 612 × 792 pts (US Letter); the file is produced by both `make report` and `make site`, and the fixture render produces a two-page `data/processed/fixture/_site/memo.pdf` from fixture results. |

The full checkpoint is green: `uv sync --frozen`, `ruff check`, `ruff format --check`,
`mypy anomaly_decay`, `pytest -q`, `make reproduce-fixture` and `make site` all pass, and the four
required CI checks (`lint`, `types`, `tests`, `reproduce-fixture`) pass on every Phase 3 pull
request.

## Key decisions and reasoning

- **One website, three pages, not three projects.** The report, companion page and memo read the same
  `results/` and must never disagree. A single Quarto website build gives them one navbar, one
  output directory and one Pages deployment; the report keeps both HTML and PDF, while the companion
  is HTML-only and the memo PDF-only.
- **Per-document scalar format overrides.** Quarto 1.10.19 rejects a document-level format array
  ("must instead be 'ansi'"), so the project declares `html` and `typst` and each page overrides with
  a scalar mapping. This is the only combination that produces exactly one format per page.
- **One JSON file per predictor.** The page fetches only the selected profile, so a reader pays for
  one small file rather than a 452KB payload. The index stays small enough to load into the
  selector, and the builder can remove stale profiles and write the index last, which makes a
  rebuild deterministic and self-consistent.
- **Yearly event-time aggregation.** The site should show derived statistics, not republish monthly
  OSAP series. Event years make the profile readable, bound the file size, and match the
  chart's horizontal axis; monthly values remain in the panel and the report's own event-time figure.
- **Descriptive, and labelled so.** The two new files were produced after the specification freeze
  and are not specifications. Both the companion page and the report call them descriptive, and the
  page says the windows use the default convention and the `op` portfolio.
- **The memo is print-first.** It exists to be read on paper or in a PDF viewer, so it renders only
  to Typst and the navbar links the PDF directly. A separate HTML memo would duplicate content and
  invite the two versions to drift.
- **The page budget is part of the document.** A Typst assertion makes an overlong memo fail the
  render itself, including CI's fixture render, which does not run the memo test module after
  rendering. The byte-level page counter in `tests/test_memo.py` then checks the built files
  independently.
- **No new JavaScript dependencies.** Observable JS, its inputs and Plot come with Quarto, so the
  site adds no library of its own; the page works from a static file server under any base path. The
  OJS runtime and its standard library are loaded from Quarto's CDN at page view, as for any Quarto
  OJS page.
- **Two stacked event panels, not facets.** Plot 0.6.11 sorts facet domains, which would put the
  sample-end panel below the publication panel; two labelled panels keep the intended reading order.
- **Search and select are two controls.** The bundled `@observablehq/inputs` 0.10.6 has no filter
  option on `Inputs.select` (it is silently ignored), so the page uses an `Inputs.search` box feeding
  an `Inputs.select` of the matches.

## Interfaces for future epics

- **Results contract.** `RESULTS_SCHEMAS` and `RESULTS_SORT_KEYS` include `predictor_info.csv`
  (key `signalname`) and `predictor_event_time.csv` (key `signalname, anchor, event_year`). Both are
  read and written by `anomaly_decay.io`, so they inherit the canonical CSV rules.
- **Profile estimators.** `anomaly_decay.estimate.predictor_profiles` exposes
  `predictor_info(metadata)`, `predictor_event_time(panel)` and the CLI
  `python -m anomaly_decay.estimate.predictor_profiles --panel P --metadata M --results-dir R`.
  `EVENT_MONTH_LO`/`EVENT_MONTH_HI` and the implied event-year bounds are module constants, and
  `EVENT_YEAR_LO`/`EVENT_YEAR_HI` document the site's range.
- **Site builder.** `anomaly_decay.site` exposes `write_site_data(results_dir, out_dir) -> SiteBuild`
  (`out_dir`, `n_predictors`, `n_bytes`), `round_float`, `SiteError` and the CLI with defaults
  `--results-dir results --out report/_site/data`. The JSON contract is: `predictors.json` is a list
  of index entries; `predictors/<signalname>.json` adds `cat_economic`, `windows` (three keys with
  `mean_ret`, `se_mean`, `n_months`) and `event_time` (two anchors with
  `event_year`, `mean_ret`, `n_months`); floats are rounded to six decimals and non-finite values
  become null.
- **Make targets.** `make report` is the full render; `make site` is the published build (render
  first, JSON second); `make reproduce` ends with `make site`; `make reproduce-fixture` renders the
  site with fixture results into `data/processed/fixture/_site/`. The Pages workflow uploads
  `report/_site`; a new page must be added to the render list in `report/_quarto.yml` to appear in
  the site and the sitemap.
- **Memo contract.** `report/memo.qmd` is the single source; `report/_site/memo.pdf` and
  `data/processed/fixture/_site/memo.pdf` are the two build outputs. The document must keep the
  Typst page assertion and the `papersize: us-letter` front matter; `tests/test_memo.py` pins both,
  the section set, the header fields and the no-typed-number rule.
- **Release.** v1.1.0 is cut by the release script. Its assets
  should be the 14 result CSVs (the 12 existing files plus the two profiles), the report PDF and the
  memo PDF, so the release matches the site built from the same committed results.

## Deviations from the plan

- **`make site` renders first and writes the JSON second.** The companion page fetches its data at
  run time and the Quarto render does not read the JSON, so the order is safe; it also means a
  successful `make site` always leaves the page and its data in the same output directory. The
  Makefile's `site` target therefore chains `report` and the `anomaly_decay.site` CLI rather than
  folding the data step into the render.
- **The full-data profile files were written by the CLI, not by `make descriptive`.** The Makefile's
  `descriptive` stage now includes the profile CLI, but the committed full-data files were produced
  directly into `results/` so no existing results file was rewritten (the commit contains only the
  two new files). A fresh `make reproduce` runs the same CLI through the stage.
- **The companion page's search is two controls.** `Inputs.select` in the bundled inputs version has
  no filter option; the page uses `Inputs.search` plus `Inputs.select` instead. The user-visible
  behaviour is a search box and a filtered list, which is what the acceptance criterion asks for.
- **The event profile is two panels.** Plot 0.6.11 sorts facet domains alphabetically, so a faceted
  chart would order the anchors wrong; the page renders two explicitly labelled panels.
- **The memo has no figure.** The event-time figure was optional and the six required sections plus
  the comparison table already fill the two-page budget; the memo points to the report's figures
  instead.
- **The fixture render checks the memo budget twice.** The Typst assertion in the document fails the
  fixture render if the memo exceeds two pages, and `make reproduce-fixture` then runs
  `tests/test_memo.py`, which counts page objects in the built fixture PDF. CI therefore enforces
  the budget from both directions.
- **The page counter assumes uncompressed page objects.** Typst 0.15.1 writes page objects in plain
  object syntax, which is what the byte counter reads. If a later Typst compresses objects,
  the page-tree cross-check fails first and the counter must be updated rather than silently
  undercounting.
- **v1.0.1 polish shipped alongside this work.** The report's forced page breaks, table headers and
  dead `reporting.py` fallback were fixed under #117, and the report now links the companion and
  memo (#123). That polish is part of the site as it ships, but it is documented in the Epic 03
  write-up and the report's own pull request, not here.

## Known limitations and deferred items

- **The companion page is descriptive.** It shows window means, standard errors and month counts for
  one predictor; it does not run the panel regression, cluster standard errors or show the
  pre-registered specification estimates. The report remains the place for inference.
- **The event-year series can be sparse.** A point exists only for a year with a usable month, and
  the minimum is one month, so an event-year mean can rest on a single observation. The page's
  month count is the guard; no minimum-month filter or confidence band is shown.
- **54 predictor-anchor-years are absent.** The full 212 × 2 × 15 grid has 6,360 cells and the file
  has 6,306 rows; missing corners are omitted rather than zero-filled, and the chart breaks the line
  there instead of inventing a value.
- **The JSON data is not versioned.** It is built into `report/_site/` at release time from the
  committed results and is not in git. A local clone must run `make site` before the companion page
  has data, and a stale `report/_site/` can be out of date until the next build.
- **The OJS runtime is a network dependency.** Quarto loads the Observable runtime and standard
  library from its CDN at page view and the predictor data from local relative URLs; without network
  access the page shows a module error rather than the charts. The data themselves never leave the
  site's own directory.
- **No automated browser test.** The companion page was checked in headless Chrome under both base
  paths, including the deep link and the no-match message, but CI does not run a browser; the
  `tests/test_site.py` checks cover the data contract, not the rendered page.
- **The fixture companion page has no data files.** Fixture renders check that `companion.html`
  renders; `make reproduce-fixture` does not run `anomaly_decay.site` into the fixture site, so the
  fixture page cannot be exercised end to end without adding that step.
- **The memo is a summary, not the report.** It omits the event-time figure, the weak-split
  interactions and most of the robustness table to stay within two pages; it names the report for
  the full evidence.
- **The fixture memo uses fictional results.** Its numbers come from `tests/fixtures/results_fixture`
  and are not interpreted; only the full-data memo is a result of the replication.
- **The memo PDF changes with the render date.** `date: today` makes the PDF differ across days even
  when `results/` is unchanged, so the memo is not byte-identical across builds; that is intended
  for a dated memo, and the result files themselves stay byte-identical.
- **The v1.1.0 release and Pages deployment are not in this document's pull requests.** They are
  release steps taken outside this epic's pull requests; this epic delivers the site, the data, the
  memo and the tests that the release packages.
