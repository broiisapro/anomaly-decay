# anomaly-decay

Do stock-return anomalies survive being published? McLean and Pontiff (2016) found that long-short returns fall
after a predictor's original sample ends and fall further after publication. This repository replicates that
result with the open Chen–Zimmermann data, extends it through the latest data vintage, and documents every
specification and deviation. `make reproduce` rebuilds every number in the report from scratch.

## Status

Epic 00 (Foundation) has landed: uv project and CI, pinned Ken French factors with hash
manifests, schema and interface contracts, a committed 15-predictor fixture with an independent reference
panel, and a placeholder Quarto report that renders numbers from `results/`. Estimation, figures and the real
report follow in Epic 01.

## Prerequisites

- uv for the Python 3.12 environment (`uv sync --frozen`).
- Quarto 1.10.x for the report: `brew install --cask quarto`.

See docs/architecture.md for the design and docs/epic-00-foundation.md for what is built so far.
