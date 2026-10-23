# Epic 03 — Phase 1 public release

## Goal

Make the replication public and citable: a README whose headline numbers are written from
`results/` instead of typed, an MIT licence and complete project metadata, the required checks moved
from the project's own macOS runner to GitHub-hosted Linux runners, the report published to GitHub
Pages by an Actions workflow when a release is published, and an annotated `v1.0.0` release carrying
the result tables and the report PDF. The repository became public only after a secret scan and
after the self-hosted runner had been removed.

## How it works

### README with a generated results block (#100)

`anomaly_decay/readme.py` fills the block between `<!-- results:start -->` and
`<!-- results:end -->` in `README.md`. It reads `main.csv`, `decay.csv`, `coverage.csv` and
`bootstrap.csv` through `anomaly_decay.io.read_results` and the paper's reference values from
`config/original_paper.yaml`, and formats every value with the report's own helpers from
`anomaly_decay.reporting`. `python -m anomaly_decay.readme` rewrites the block;
`--check` fails when the block is stale.

`--check` also fails when a number has been typed anywhere else in the README. Code spans and
fenced blocks are ignored and four-digit years are allowed, so commands and versions live in code
spans and a hand-entered count fails the check. `tests/test_readme.py` asserts that the committed
block equals a fresh render, exercises the stale-block, typed-number and malformed-marker failures,
checks that a mutation in each input file reaches the block, and pins the opening paragraph and the
release sections. The test suite runs `--check`, so CI enforces the block like any other test.

The block carries the headline result and the comparison with the paper: long-short returns are
0.239% per month lower after the original sample ends and 0.337% per month lower after publication
(198 predictors, 101,567 observations), a 35.2% and 49.5% decline against the paper's 26% and 58%,
together with the bootstrap intervals and the vintage coverage. The rest of the README is prose:
what the repository demonstrates, the reproduction commands, the report and documentation links, a
data-and-licence note, and a limitations paragraph that links the report's Deviations and
Limitations sections.

#103 regenerated the block after `reporting.month_label` changed the month wording to
"January 1926" style, and made the README mutation tests whitespace-insensitive. No number changed;
the change kept `main` green for the new wording.

### Licence and project metadata (#103)

- `LICENSE` is the MIT licence, "Copyright (c) 2026 Moksh Siruvani".
- `pyproject.toml` `[project]` carries the release description, `license = "MIT"`, the author, the
  repository URL under `[project.urls]` and `readme = "README.md"`. A built wheel reports
  `License-Expression: MIT` and `License-File: LICENSE` (PEP 639), and `uv build --wheel` is the
  check that the metadata is valid.
- The GitHub repository description was set to the same release description with `gh repo edit`
  and verified with `gh repo view`. At go-public the homepage URL
  (`https://broiisapro.github.io/anomaly-decay/`) and the topic list were added.

### Hosted CI switch (#106)

All four jobs — `lint`, `types`, `tests`, `reproduce-fixture` — moved from the self-hosted macOS
runner to `ubuntu-latest`. The job IDs, commands and test selection are unchanged, so the four
required status checks on `main` kept their names. uv stays at `0.11.26` (`astral-sh/setup-uv`
pinned to `v10.3.0` because the action publishes release tags only), `reproduce-fixture` installs
Quarto `1.10.19` with `quarto-dev/quarto-actions/setup@v2` instead of probing for a Homebrew
install, and `tests` keeps `ANOMALY_DECAY_REQUIRE_FREEZE=1`, `fetch-depth: 0` and
`fetch-tags: true` so the `specs-frozen` tag is present on the hosted checkout.

The switch was prepared while the repository was still private and kept open under a
`do-not-merge` label: merging it earlier would have moved every pull request and `main` push onto
hosted minutes while the self-hosted runner was still installed. It was merged as part of the
go-public step, and its merge commit is the commit the `v1.0.0` tag points at. Before merging, all
four jobs were run on GitHub-hosted Ubuntu: the freeze test ran, and `reproduce-fixture` reproduced
the panel exactly and rendered both report formats on Linux.

### Pages via Actions on release (#105, #107)

`.github/workflows/pages.yml` publishes the site. It runs on `release: published` and
`workflow_dispatch` only — never on push or pull requests — and its build job is:

1. check out the released tag;
2. install uv `0.11.26` and Quarto `1.10.19`;
3. `uv sync --frozen`, then `make report` from the committed `results/`;
4. upload `report/_site` with `actions/upload-pages-artifact`.

The deploy job publishes the artifact with `actions/deploy-pages` under the `github-pages`
environment. Permissions are `contents: read`, `pages: write` and `id-token: write`, and the
concurrency group `pages` never cancels a deployment in progress. There is no `gh-pages` branch and
no `quarto publish gh-pages`: the deployment is an artifact built from the tagged commit's
committed results.

#107 pins `astral-sh/setup-uv` to `v10.3.0` in this workflow too. The first version used `@v10`,
which does not exist (the action publishes release tags only), so the workflow failed during setup
rather than in the build. The same correction was needed in `ci.yml` for the hosted switch.

The first release deployment was rejected by the `github-pages` environment: the environment that
GitHub creates automatically allows deployments from branches only, and a release workflow checks
out a tag. Adding a `v*` tag policy to the environment and re-running the deployment succeeded;
the site serves at <https://broiisapro.github.io/anomaly-decay/> with the HTML report and
`index.pdf`. The policy is repository configuration, not a file in the repository, so it is
recorded here.

### v1.0.0 release

The annotated tag `v1.0.0` ("v1.0.0: Phase 1 public release") points at the merge commit of #106.
The release notes state the main result (0.239% and 0.337% per month lower; 35.2% and 49.5%
declines against the paper's 26% and 58%), what the release contains, and the caveats that the
report documents. The release carries thirteen assets: the twelve result CSVs
(`bootstrap`, `coverage`, `decay`, `event_time`, `extension`, `main`, `multiple_testing`,
`predictor_summary`, `robustness`, `specs_run`, `tstat_summary`, `vintage_comparison`) and the
report PDF.

Release: <https://github.com/broiisapro/anomaly-decay/releases/tag/v1.0.0>

### Secret scan, runner removal and the visibility change

The go-public step ran in this order:

1. **Fresh-clone gates.** A fresh clone passed the full test suite with the real-data tests
   skipping instead of failing, `make reproduce-fixture` reproduced the panel and rendered the
   report, a second `make reproduce` run left `results/` byte-identical to the committed copy,
   `make verify-hashes` matched the manifest, the PRD headline command printed the committed
   decay, the README block check passed, and the full-reproduction test rebuilt every result file
   from the raw cache.
2. **Secret scan.** A gitleaks scan over the repository found no leaked credentials. The
   repository needs no secrets: there is no deployment key, registry token or API key in the
   workflows, and `.env.example` states that no secrets are needed.
3. **Runner removal.** All self-hosted runners were removed (the runner list is empty) before the
   visibility change. A public repository must not offer a self-hosted runner to pull requests
   from forks, and the hosted checks had already been proven on the switch PR.
4. **Visibility and Pages.** The repository was made public with its description and topics; Pages
   was enabled with "GitHub Actions" as the source; #106 was merged; `main` CI ran green on the
   hosted runners; and the `v1.0.0` tag and release were published, which deployed the site.

## Key decisions and reasoning

- **README numbers come from the results, like the report's.** The README is the first thing a
  visitor reads, so it must not be the one place where a number can go stale or be typed by hand.
  `--check` makes the block a testable artefact, and the typed-number rule pushes every result
  value inside it.
- **MIT for the code, no redistribution of the data.** OSAP and Ken French data have their own
  terms; the repository publishes derived result tables and figures only, and the README says so.
- **The CI switch keeps the check names and the toolchain.** The four job IDs are the required
  status checks, so renaming them would block every merge. Keeping uv and Quarto at the same
  versions means the switch changes the operating system of the runner, not what is verified.
- **Pages deploys the released tag's report, not the branch head.** The site and the release assets
  then describe the same commit, and the site cannot drift from the release between publications.
- **The hosted switch waited for go-public.** Running the private repository's checks on hosted
  minutes while a working self-hosted runner existed would have wasted the free allowance and
  hidden any runner-specific problem until the last moment.
- **The release lists individual assets rather than one archive.** Each result table can be
  downloaded on its own, and GitHub records a digest for each file.

## Interfaces for future epics

- **Publishing.** An annotated `vX.Y.Z` tag with a published GitHub release deploys the site; a
  manual dispatch of the `pages` workflow redeploys from the selected ref. The build contract is
  `make report`, which must keep writing `report/_site/index.html` and `report/_site/index.pdf`.
- **Release contents.** Twelve result CSVs plus the report PDF. The website and the release must
  come from the same committed `results/`.
- **README block.** `python -m anomaly_decay.readme` writes it and `--check` verifies it. Changes to
  its content go through `anomaly_decay/readme.py` and its tests, never through a direct edit of
  the block.
- **Required checks.** The names `lint`, `types`, `tests` and `reproduce-fixture` are fixed by the
  `main` ruleset; new checks would be added, not substituted.
- **Repository environment.** The `github-pages` environment accepts deployments from `main` and
  from `v*` tags. A re-created environment needs the tag policy again.
- **Package metadata.** `pyproject.toml` is the single source for the package description, licence,
  author and repository URL; the GitHub description mirrors it.

## Deviations from the plan

- **Pages is deployed by an Actions workflow, not `quarto publish gh-pages`.** The artifact comes
  from the tagged commit and no `gh-pages` branch exists. This was chosen at the start of the epic
  to keep the published site and the release identical and to avoid a long-lived branch that could
  diverge from `results/`.
- **The hosted CI switch was staged as an open pull request.** It was prepared with the rest of the
  CI work but merged only at go-public, so the private repository stayed on its own runner until
  that runner could be removed.
- **The first Pages deployment needed a repository setting.** The environment's default
  branch-only policy rejected a deployment from the release tag; the `v*` tag policy was added and
  the deployment re-run. The fix is configuration, so it cannot be reviewed in a pull request.
- **`make readme` does not exist.** The Makefile was outside #100's file set, so the target was
  requested in the pull request body and the README documents
  `uv run python -m anomaly_decay.readme` instead. The block and its check are unaffected.
- **Secret scanning is a pre-publication check, not a CI job.** It ran once over the repository
  before the visibility change; no workflow runs it on later commits.

## Known limitations and deferred items

- **`make readme` is still missing.** The two-line target is deferred to whoever next owns the
  `Makefile`; the direct module command documented in the README behaves identically.
- **`dist/` is not ignored.** A one-off `uv build` leaves an untracked `dist/` directory; no build
  output is committed and `uv build` is not part of any target, so `.gitignore` was left unchanged.
- **Secret scanning is not automated.** A later commit could introduce a credential without a CI
  failure; a scanning workflow is the natural follow-up.
- **The site is updated only on release or manual dispatch.** `main` can be ahead of the published
  site between releases, by design; the site always matches a tag.
- **Pages settings are not versioned.** The Pages source, the `github-pages` environment's tag
  policy, the repository description and the topics live in GitHub settings. A fork or a
  re-created repository must apply them again.
- **The release assets are a snapshot.** Nothing re-checks the attached files against the tag's
  committed results after upload; GitHub's per-asset digests are the record of what was published.
- **The site and the release each render the report PDF.** Both read the same committed results,
  but the two builds are independent, so the files need not be byte-identical even when their
  content agrees.
