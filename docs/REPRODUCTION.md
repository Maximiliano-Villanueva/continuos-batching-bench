# Reproduction guide

Step-by-step instructions for running the **rigorous** benchmark profile on Apple Silicon with [vllm-metal](https://github.com/vllm-project/vllm-metal).

← [Project README](../README.md) · [Documentation index](README.md)

---

## Architecture

Two processes, two virtual environments:

| Terminal | Venv | Command |
|----------|------|---------|
| A (server) | `.venv-vllm-metal` | `./scripts/serve.sh <model> off\|on` |
| B (client) | `.venv` | `make bench-*`, `make dashboard` |

The client sends OpenAI-compatible HTTP requests and records metrics. Continuous batching and speculative decoding are configured on the server only.

---

## Prerequisites

- Apple Silicon Mac (M1–M4)
- Python 3.12+
- 32 GB unified memory recommended for Gemma 4 BF16 + draft speculative model
- Network access for initial model and dependency downloads

---

## One-time setup

```bash
cd continuous-batching

make install-vllm-metal          # → .venv-vllm-metal/
python3 -m venv .venv
source .venv/bin/activate
make install-bench               # → cb-bench in .venv/
make prompts-long                # ~8k-token long prompts for E1/E3
```

All paths and virtual environments live inside the repository—nothing is installed globally or under `$HOME`.

---

## Gemma 4 — speculative OFF

### Terminal A (server)

```bash
./scripts/serve.sh gemma-4-e4b off
```

Wait until `curl -s http://127.0.0.1:8000/v1/models` returns a model list.

### Terminal B (client)

```bash
source .venv/bin/activate
make serve-check
make bench-gemma-off
```

Record the `run_id` printed at completion (or use the newest folder under `results/`).

---

## Gemma 4 — speculative ON

Stop the server (Ctrl+C in Terminal A), then restart with speculative decoding enabled.

### Terminal A

```bash
./scripts/serve.sh gemma-4-e4b on
```

### Terminal B

```bash
make serve-check
make bench-gemma-on
```

---

## View and export results

```bash
make dashboard    # http://localhost:8501
make export       # regenerate CSV and Markdown for the latest run
```

In the dashboard sidebar, select a `run_id` and compare runs that share the same `model_key` but differ in `speculative_enabled`.

---

## Expected runtime

| Profile | Measured requests | Approximate wall time (Gemma 4) |
|---------|-------------------|---------------------------------|
| `smoke` | ~70 | 2–5 minutes |
| `rigorous` | ~8,730 | 5–6 hours per run |

Preview request counts without a live server:

```bash
make estimate
```

Warmup requests (3 per scenario) are sent but not persisted to SQLite.

---

## Resume a partial run

If a run is interrupted (server crash, sleep, etc.):

```bash
make resume RUN_ID=<run_id> MODEL=gemma-4-e4b SPEC=on ONLY=E7
```

Or via CLI:

```bash
.venv/bin/cb-bench run --model gemma-4-e4b --speculative on \
  --resume-run-id <run_id> --only-experiment E7
```

Completed scenarios are detected automatically and skipped.

---

## Qwen 3.5

The same workflow applies with `qwen3.5-4b`:

```bash
# Terminal A
./scripts/serve.sh qwen3.5-4b off
# Terminal B
make bench-qwen-off

# Restart server with speculative on, then:
make bench-qwen-on
```

Qwen uses native **MTP** speculative decoding (`qwen3_next_mtp`), not a draft model.

---

## Gemma-specific configuration

| Setting | Value |
|---------|-------|
| Target checkpoint | `mlx-community/gemma-4-e4b-it-bf16` |
| Speculative draft | `mlx-community/gemma-3-1b-it-qat-4bit` |
| Text-only flags | Applied by `scripts/serve.sh` (`--limit-mm-per-prompt.* 0`, `--max-model-len 8192`) |

The 4-bit Gemma MLX checkpoint (`gemma-4-e4b-it-4bit`) is not supported on current vllm-metal builds. Use BF16.

Full registry: [`configs/models.yaml`](../configs/models.yaml).

---

## Sample results (no GPU required)

Pre-exported auto-summaries from completed rigorous runs:

- [Gemma 4, speculative OFF](sample-results/gemma-4-e4b-off-summary.md)
- [Gemma 4, speculative ON](sample-results/gemma-4-e4b-on-summary.md)

Full `benchmark.db` files remain under `results/` (gitignored). To share interactive dashboard data, attach databases or CSV exports as [GitHub release artifacts](https://docs.github.com/en/repositories/releasing-projects-on-github).

---

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `make serve-check` fails | Start `./scripts/serve.sh …` in Terminal A; confirm port 8000 is listening |
| Dashboard shows no runs | Only directories with stored request data are listed; run `make export` after a benchmark |
| `cb-bench` not found | Activate `.venv` or use Make targets |
| Speculative server JSON error | Use `./scripts/serve.sh` instead of invoking `vllm serve` directly |
| `Too many open files` | `serve.sh` raises `ulimit -n`; restart the terminal session if needed |
| CPU-only vLLM after install | Re-run `make install-vllm-metal` |
