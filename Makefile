# anomaly-decay — one-command reproduction.
# BLAS/OpenMP threads pinned to 1 everywhere for deterministic results.

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

.PHONY: fetch verify-hashes refresh-french metadata panel estimate figures \
        report report-placeholder reproduce reproduce-fixture lint types test

fetch:
	uv run python -m anomaly_decay.data.fetch

verify-hashes:
	uv run python -m anomaly_decay.data.fetch --verify

refresh-french:
	uv run python -m anomaly_decay.data.fetch --refresh-french

# Pipeline stages. These call the interface stubs and therefore fail with
# NotImplementedError until Epic 01 implements them; `reproduce` is expected to
# fail at the first un-implemented stage until then.
metadata:
	uv run python -m anomaly_decay.data.metadata --data-dir data/raw \
		--out data/processed/predictors.parquet

panel:
	uv run python -m anomaly_decay.panel.build --data-dir data/raw \
		--metadata data/processed/predictors.parquet \
		--portfolios op,deciles_ew,deciles_vw \
		--out data/processed/panel.parquet

estimate:
	uv run python -m anomaly_decay.estimate.main --panel data/processed/panel.parquet \
		--specs config/specifications.yaml --results-dir results --print-headline

figures:
	uv run python -m anomaly_decay.figures.event_time --results-dir results
	uv run python -m anomaly_decay.figures.distribution --results-dir results
	uv run python -m anomaly_decay.figures.coefplot --results-dir results

# Report from the real results directory (Quarto 1.10.x, bundled Typst PDF).
report:
	uv run quarto render report

# Report rendered from the committed dummy results; part of the offline gate.
report-placeholder:
	ANOMALY_DECAY_RESULTS=tests/fixtures/dummy_results uv run quarto render report

# Offline gate: fixture manifest hashes, panel schema, placeholder report, output files.
reproduce-fixture:
	uv run python -m anomaly_decay.data.fetch --verify --manifest-scope fixture
	uv run python -c "import polars as pl; from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, validate_frame; validate_frame(pl.read_parquet('tests/fixtures/panel_fixture.parquet'), PANEL_SCHEMA, key=PANEL_KEY)"
	$(MAKE) report-placeholder
	test -f report/_site/index.html
	test -f report/_site/index.pdf

# Full online pipeline: fetch -> metadata -> panel -> estimate -> figures -> report.
# Ends with the rendered report under report/_site/.
reproduce:
	$(MAKE) fetch
	$(MAKE) metadata
	$(MAKE) panel
	$(MAKE) estimate
	$(MAKE) figures
	$(MAKE) report

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy anomaly_decay

test:
	uv run pytest -q
