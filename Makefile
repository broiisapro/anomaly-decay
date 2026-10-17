# anomaly-decay — one-command reproduction.
# BLAS/OpenMP threads pinned to 1 everywhere for deterministic results.

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

.PHONY: fetch verify-hashes refresh-french lint types test reproduce-fixture

fetch:
	uv run python -m anomaly_decay.data.fetch

verify-hashes:
	uv run python -m anomaly_decay.data.fetch --verify

refresh-french:
	uv run python -m anomaly_decay.data.fetch --refresh-french

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy anomaly_decay

test:
	uv run pytest -q

reproduce-fixture:
	uv run python -c "import anomaly_decay"
	@echo "reproduce-fixture: placeholder (Epic 00 #4 replaces this)"
