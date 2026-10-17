# anomaly-decay — one-command reproduction.
# BLAS/OpenMP threads pinned to 1 everywhere for deterministic results.

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

# Pinned inputs and pipeline outputs (paths relative to the repository root).
VINTAGE := 202510
DATA_DIR := data/raw
PINNED_DIR := data/pinned
PROCESSED_DIR := data/processed
RESULTS_DIR := results
SPECS := config/specifications.yaml
PORTFOLIOS := op,deciles_ew,deciles_vw
SIGNAL_DOC := $(DATA_DIR)/$(VINTAGE)/signal_doc.csv
SUPPLEMENT := config/predictor_dates_supplement.csv
METADATA := $(PROCESSED_DIR)/predictors.parquet
PANEL := $(PROCESSED_DIR)/panel.parquet

# Offline fixture inputs and the throwaway results copy used by `report-fixture`.
FIXTURE_RESULTS_DIR := tests/fixtures/results_fixture
FIXTURE_REPORT_DIR := $(PROCESSED_DIR)/report-fixture

.PHONY: fetch verify-hashes refresh-french metadata panel estimate descriptive \
        figures report report-placeholder report-fixture reproduce reproduce-fixture \
        lint types test

fetch:
	uv run python -m anomaly_decay.data.fetch

verify-hashes:
	uv run python -m anomaly_decay.data.fetch --verify

refresh-french:
	uv run python -m anomaly_decay.data.fetch --refresh-french

metadata:
	uv run python -m anomaly_decay.data.metadata --signal-doc $(SIGNAL_DOC) \
		$(if $(wildcard $(SUPPLEMENT)),--supplement $(SUPPLEMENT)) \
		--out $(METADATA)

panel:
	uv run python -m anomaly_decay.panel.build --data-dir $(DATA_DIR) \
		--pinned-dir $(PINNED_DIR) --vintage $(VINTAGE) \
		--metadata $(METADATA) --portfolios $(PORTFOLIOS) --out $(PANEL)

estimate:
	uv run python -m anomaly_decay.estimate.main --panel $(PANEL) \
		--specs $(SPECS) --results-dir $(RESULTS_DIR) --print-headline

descriptive:
	uv run python -m anomaly_decay.estimate.descriptive --panel $(PANEL) \
		--metadata $(METADATA) --vintage $(VINTAGE) --results-dir $(RESULTS_DIR)

figures:
	uv run python -m anomaly_decay.figures.event_time --results-dir $(RESULTS_DIR)
	uv run python -m anomaly_decay.figures.distribution --results-dir $(RESULTS_DIR)
	uv run python -m anomaly_decay.figures.coefplot --results-dir $(RESULTS_DIR)

# Report from the real results directory (Quarto 1.10.x, bundled Typst PDF).
report:
	uv run quarto render report

# Development render from the committed dummy results; every missing results file renders
# a pending note, so this exercises the report's not-yet-produced paths.
report-placeholder:
	ANOMALY_DECAY_RESULTS=tests/fixtures/dummy_results uv run quarto render report

# Render from the committed results fixture. The figure CLIs run on a throwaway copy so
# the committed fixture directory stays byte-identical.
report-fixture:
	rm -rf "$(FIXTURE_REPORT_DIR)"
	mkdir -p "$(FIXTURE_REPORT_DIR)"
	cp -R "$(FIXTURE_RESULTS_DIR)/." "$(FIXTURE_REPORT_DIR)/"
	uv run python -m anomaly_decay.figures.event_time --results-dir $(FIXTURE_REPORT_DIR)
	uv run python -m anomaly_decay.figures.distribution --results-dir $(FIXTURE_REPORT_DIR)
	uv run python -m anomaly_decay.figures.coefplot --results-dir $(FIXTURE_REPORT_DIR)
	ANOMALY_DECAY_RESULTS=$(FIXTURE_REPORT_DIR) uv run quarto render report

# Offline gate: fixture manifest hashes, panel schema, fixture report, output files.
reproduce-fixture:
	uv run python -m anomaly_decay.data.fetch --verify --manifest-scope fixture
	uv run python -c "import polars as pl; from anomaly_decay.schemas import PANEL_KEY, PANEL_SCHEMA, validate_frame; validate_frame(pl.read_parquet('tests/fixtures/panel_fixture.parquet'), PANEL_SCHEMA, key=PANEL_KEY)"
	$(MAKE) report-fixture
	test -f report/_site/index.html
	test -f report/_site/index.pdf

# Full online pipeline:
# fetch -> metadata -> panel -> estimate -> descriptive -> figures -> report.
reproduce:
	$(MAKE) fetch
	$(MAKE) metadata
	$(MAKE) panel
	$(MAKE) estimate
	$(MAKE) descriptive
	$(MAKE) figures
	$(MAKE) report

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy anomaly_decay

test:
	uv run pytest -q
