.PHONY: install install-vllm-metal install-bench test lint bench dashboard serve-check smoke integration resume

# Resume partial run: make resume RUN_ID=abc123 MODEL=gemma-4-e4b SPEC=on ONLY=E7
RUN_ID ?=
MODEL ?= gemma-4-e4b
SPEC ?= on
ONLY ?=

# Benchmark client (Terminal B)
install-bench:
	pip install -e ".[dev]"

# vllm-metal server into ./.venv-vllm-metal (Terminal A) — large download
install-vllm-metal:
	./scripts/install_vllm_metal.sh

install: install-bench

test:
	.venv/bin/pytest tests/unit -v

integration:
	.venv/bin/pytest tests/integration -v -m integration

lint:
	.venv/bin/ruff check src tests

serve-check:
	.venv/bin/python scripts/check_server.py

bench:
	.venv/bin/cb-bench run --config configs/experiments.yaml

estimate:
	.venv/bin/python scripts/estimate_run_size.py rigorous

estimate-smoke:
	.venv/bin/python scripts/estimate_run_size.py smoke

prompts-long:
	.venv/bin/python scripts/build_prompts.py

bench-qwen-off:
	.venv/bin/cb-bench run --model qwen3.5-4b --speculative off

bench-qwen-on:
	.venv/bin/cb-bench run --model qwen3.5-4b --speculative on

bench-gemma-off:
	.venv/bin/cb-bench run --model gemma-4-e4b --speculative off

bench-gemma-on:
	.venv/bin/cb-bench run --model gemma-4-e4b --speculative on

resume:
	@test -n "$(RUN_ID)" || (echo "Usage: make resume RUN_ID=<run_id> [MODEL=gemma-4-e4b] [SPEC=on|off] [ONLY=E7]" && exit 1)
	.venv/bin/cb-bench run --model $(MODEL) --speculative $(SPEC) \
		--resume-run-id $(RUN_ID) $(if $(ONLY),--only-experiment $(ONLY),)

smoke:
	.venv/bin/cb-bench run --config configs/experiments.yaml --smoke

models:
	.venv/bin/cb-bench models

dashboard:
	.venv/bin/streamlit run src/continuous_batching/reporting/dashboard.py

export:
	.venv/bin/cb-bench export --results-dir results
