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
PREREG := config/phase2_preregistration.yaml
PORTFOLIOS := op,deciles_ew,deciles_vw
SIGNAL_DOC := $(DATA_DIR)/$(VINTAGE)/signal_doc.csv
SUPPLEMENT := config/predictor_dates_supplement.csv
METADATA := $(PROCESSED_DIR)/predictors.parquet
PANEL := $(PROCESSED_DIR)/panel.parquet

# Offline fixture inputs, the throwaway results copy used by `report-fixture`, and the
# fixture pipeline outputs. Nothing here is ever written to `results/`.
FIXTURE_RESULTS_DIR := tests/fixtures/results_fixture
FIXTURE_REPORT_DIR := $(PROCESSED_DIR)/report-fixture
FIXTURE_OSAP_DIR := tests/fixtures/data/osap
FIXTURE_PINNED_DIR := tests/fixtures/data/pinned
FIXTURE_SIGNAL_DOC := $(FIXTURE_OSAP_DIR)/$(VINTAGE)/signal_doc.csv
FIXTURE_REFERENCE_PANEL := tests/fixtures/panel_fixture.parquet
FIXTURE_PROCESSED_DIR := $(PROCESSED_DIR)/fixture
FIXTURE_METADATA := $(FIXTURE_PROCESSED_DIR)/predictors.parquet
FIXTURE_PANEL := $(FIXTURE_PROCESSED_DIR)/panel.parquet
FIXTURE_RESULTS := $(FIXTURE_PROCESSED_DIR)/results
FIXTURE_HASHES_BEFORE := $(FIXTURE_PROCESSED_DIR)/results-hashes.before.txt
FIXTURE_HASHES_AFTER := $(FIXTURE_PROCESSED_DIR)/results-hashes.after.txt
# Quarto output for every fixture/development render. `report/_site/` belongs to
# `make report` alone, so the real report can never be overwritten by a fixture run.
FIXTURE_SITE_DIR := $(FIXTURE_PROCESSED_DIR)/_site
FIXTURE_HASH_CMD = cd "$(CURDIR)/$(FIXTURE_RESULTS)" && find . -type f | LC_ALL=C sort | xargs shasum -a 256

# Phase 2 CLIs (pre-registered analyses); the `phase2` target names any that do not exist yet.
PHASE2_CLIS := anomaly_decay.estimate.bootstrap anomaly_decay.estimate.multiple_testing \
               anomaly_decay.estimate.vintages

.PHONY: fetch verify-hashes refresh-french metadata panel estimate descriptive \
        figures report site readme report-placeholder report-fixture reproduce \
        reproduce-fixture phase2 reproduce-phase2 \
        fixture-metadata fixture-panel fixture-panel-check fixture-results \
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
	uv run python -m anomaly_decay.estimate.predictor_profiles --panel $(PANEL) \
		--metadata $(METADATA) --results-dir $(RESULTS_DIR)

figures:
	uv run python -m anomaly_decay.figures.event_time --results-dir $(RESULTS_DIR)
	uv run python -m anomaly_decay.figures.distribution --results-dir $(RESULTS_DIR)
	uv run python -m anomaly_decay.figures.coefplot --results-dir $(RESULTS_DIR)

# Report from the real results directory (Quarto 1.10.x, bundled Typst PDF).
# This is the only target that writes report/_site/.
report:
	uv run quarto render report

# The published site: the rendered report plus the per-predictor data files the
# companion page reads at run time (written into report/_site/data/).
site: report

# Rewrite the generated results block in README.md from the committed results.
readme:
	uv run python -m anomaly_decay.readme

# Development render from the committed dummy results; every missing results file renders
# a pending note, so this exercises the report's not-yet-produced paths. The output goes to
# the fixture site directory, never report/_site/.
report-placeholder:
	rm -rf "$(FIXTURE_SITE_DIR)"
	ANOMALY_DECAY_RESULTS=tests/fixtures/dummy_results uv run quarto render report \
		--output-dir "$(CURDIR)/$(FIXTURE_SITE_DIR)"

# Render from the committed results fixture. The figure CLIs run on a throwaway copy so
# the committed fixture directory stays byte-identical. The output goes to the fixture
# site directory, never report/_site/.
report-fixture:
	rm -rf "$(FIXTURE_REPORT_DIR)" "$(FIXTURE_SITE_DIR)"
	mkdir -p "$(FIXTURE_REPORT_DIR)"
	cp -R "$(FIXTURE_RESULTS_DIR)/." "$(FIXTURE_REPORT_DIR)/"
	uv run python -m anomaly_decay.figures.event_time --results-dir $(FIXTURE_REPORT_DIR)
	uv run python -m anomaly_decay.figures.distribution --results-dir $(FIXTURE_REPORT_DIR)
	uv run python -m anomaly_decay.figures.coefplot --results-dir $(FIXTURE_REPORT_DIR)
	ANOMALY_DECAY_RESULTS=$(FIXTURE_REPORT_DIR) uv run quarto render report \
		--output-dir "$(CURDIR)/$(FIXTURE_SITE_DIR)"

# Offline fixture pipeline: the production CLIs on committed fixture inputs, writing
# under data/processed/fixture/ only.
fixture-metadata:
	mkdir -p "$(FIXTURE_PROCESSED_DIR)"
	rm -f "$(FIXTURE_METADATA)"
	uv run python -m anomaly_decay.data.metadata --signal-doc $(FIXTURE_SIGNAL_DOC) \
		$(if $(wildcard $(SUPPLEMENT)),--supplement $(SUPPLEMENT)) \
		--out $(FIXTURE_METADATA)

fixture-panel: fixture-metadata
	rm -f "$(FIXTURE_PANEL)"
	uv run python -m anomaly_decay.panel.build --data-dir $(FIXTURE_OSAP_DIR) \
		--pinned-dir $(FIXTURE_PINNED_DIR) --vintage $(VINTAGE) \
		--metadata $(FIXTURE_METADATA) --portfolios $(PORTFOLIOS) --out $(FIXTURE_PANEL)

# The production panel must equal the independent reference panel exactly (values, dtypes,
# row order) before any estimate runs on it.
fixture-panel-check: fixture-panel
	uv run python -c "import polars as pl; from polars.testing import assert_frame_equal; assert_frame_equal(pl.read_parquet('$(FIXTURE_PANEL)'), pl.read_parquet('$(FIXTURE_REFERENCE_PANEL)'), check_exact=True); print('panel: exact match with $(FIXTURE_REFERENCE_PANEL)')"

fixture-results: fixture-panel-check
	rm -rf "$(FIXTURE_RESULTS)"
	uv run python -m anomaly_decay.estimate.main --panel $(FIXTURE_PANEL) \
		--specs $(SPECS) --results-dir $(FIXTURE_RESULTS) --print-headline
	uv run python -m anomaly_decay.estimate.descriptive --panel $(FIXTURE_PANEL) \
		--metadata $(FIXTURE_METADATA) --vintage $(VINTAGE) --results-dir $(FIXTURE_RESULTS)
	uv run python -m anomaly_decay.figures.event_time --results-dir $(FIXTURE_RESULTS)
	uv run python -m anomaly_decay.figures.distribution --results-dir $(FIXTURE_RESULTS)
	uv run python -m anomaly_decay.figures.coefplot --results-dir $(FIXTURE_RESULTS)

# Offline end-to-end gate, never writing to `results/`: fixture manifest verify ->
# metadata -> panel (asserted equal to the reference panel) -> estimate -> descriptive ->
# figures -> the same result stages a second time with every file under
# data/processed/fixture/results/ required to hash byte-identically -> the committed
# expectations (panel, expected main.csv to 1e-10) -> report rendered from the fixture
# results with both output files present.
reproduce-fixture:
	uv run python -m anomaly_decay.data.fetch --verify --manifest-scope fixture
	$(MAKE) fixture-results
	$(FIXTURE_HASH_CMD) > "$(CURDIR)/$(FIXTURE_HASHES_BEFORE)"
	$(MAKE) fixture-results
	$(FIXTURE_HASH_CMD) > "$(CURDIR)/$(FIXTURE_HASHES_AFTER)"
	cmp "$(CURDIR)/$(FIXTURE_HASHES_BEFORE)" "$(CURDIR)/$(FIXTURE_HASHES_AFTER)"
	uv run pytest -q tests/test_reproduce_fixture.py
	rm -rf "$(FIXTURE_SITE_DIR)"
	ANOMALY_DECAY_RESULTS=$(FIXTURE_RESULTS) uv run quarto render report \
		--output-dir "$(CURDIR)/$(FIXTURE_SITE_DIR)"
	test -f "$(FIXTURE_SITE_DIR)/index.html"
	test -f "$(FIXTURE_SITE_DIR)/index.pdf"

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

# Phase 2 analyses, pre-registered in config/phase2_preregistration.yaml and frozen with the
# specifications by the `specs-frozen` tag. Each CLI module is imported before use so that a
# module that does not exist yet is named explicitly and the target fails instead of printing
# three tracebacks.
phase2: panel
	@missing=""; for module in $(PHASE2_CLIS); do \
		if ! uv run python -c "import $$module" >/dev/null 2>&1; then \
			missing="$$missing $$module"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "phase2: CLI module(s) not available yet:$$missing"; \
		exit 1; \
	fi
	uv run python -m anomaly_decay.estimate.bootstrap --panel $(PANEL) \
		--prereg $(PREREG) --specs $(SPECS) --results-dir $(RESULTS_DIR) \
		--seed 2027 --reps 2000
	uv run python -m anomaly_decay.estimate.multiple_testing --panel $(PANEL) \
		--signal-doc $(SIGNAL_DOC) --prereg $(PREREG) --specs $(SPECS) \
		--results-dir $(RESULTS_DIR)
	uv run python -m anomaly_decay.estimate.vintages --vintages 202410,$(VINTAGE) \
		--data-dir $(DATA_DIR) --pinned-dir $(PINNED_DIR) --specs $(SPECS) \
		--results-dir $(RESULTS_DIR)

# Phase 2 results plus the report that reads them.
reproduce-phase2: phase2
	$(MAKE) report

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy anomaly_decay

test:
	uv run pytest -q
