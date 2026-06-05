# Continuous Batching Benchmarks

A reproducible benchmark harness for measuring **latency**, **throughput**, and **memory** on Apple Silicon when serving LLMs with [vllm-metal](https://github.com/vllm-project/vllm-metal). The suite exercises **continuous batching** (client concurrency sweeps), **mixed prompt workloads**, and **speculative decoding** A/B comparisons across seven structured experiments (E1–E7).

**Documentation:** [Reproduction guide](docs/REPRODUCTION.md) · [Architecture walkthrough](docs/CODE_WALKTHROUGH.md) · [Sample results](docs/sample-results/)

---

## Overview

This repository separates concerns deliberately:

| Component | Role |
|-----------|------|
| **vLLM server** (`vllm serve` via vllm-metal) | Model inference, continuous batching, speculative decoding |
| **Benchmark client** (`cb-bench`) | HTTP load generation, metric collection, SQLite storage |
| **Streamlit dashboard** | Interactive exploration and auto-generated conclusions |

The client does not configure batching or speculation—it tags each run and sends realistic mixed workloads so you can compare server configurations under identical load.

```mermaid
flowchart LR
  subgraph server [Terminal A — vllm-metal]
    VLLM[vllm serve]
  end
  subgraph client [Terminal B — cb-bench]
    CLI[cb-bench run]
    DB[(SQLite)]
    Dash[Streamlit dashboard]
  end
  CLI -->|OpenAI HTTP streaming| VLLM
  CLI --> DB
  DB --> Dash
```

---

## Highlights (Gemma 4 E4B, rigorous profile)

Sample auto-summaries from completed runs are in [`docs/sample-results/`](docs/sample-results/). Headline observations on Apple Silicon with `mlx-community/gemma-4-e4b-it-bf16`:

| Area | Finding |
|------|---------|
| **Concurrency sweep (E4)** | Throughput scales from ~18 tok/s at K=1 to ~92–97 tok/s at K=16 (~5×); per-request p99 latency rises with K—the classic batching tradeoff. |
| **Speculative decoding (E4, K=16)** | Draft speculative decoding yields ~6% higher throughput and a smoother p99 tail vs non-speculative at high concurrency. |
| **Prompt placement (E2)** | Long prompt **last** improves short-prompt p99 (~1.3 s vs ~3.4 s) versus long-first ordering; long-prompt tail latency is higher when scheduled last. |
| **API surface (E6)** | Chat completions are dramatically faster than raw Completions for the same model (~1 s vs ~14 s p50 E2E). |
| **Decode-heavy load (E7)** | Long-output scenarios dominate wall time (~22–55 s p50); speculative on/off differs by only ~1–3% overall. |

Re-run the suite locally to reproduce on your hardware, or inspect the committed summaries for the full bullet list.

---

## Quick start

You need **two terminals** with **two virtual environments** inside this repo:

| | Terminal A — server | Terminal B — client |
|---|---------------------|---------------------|
| **Venv** | `.venv-vllm-metal` | `.venv` |
| **Purpose** | `vllm serve` | `cb-bench run`, dashboard |

> **Mac only.** Do not use the CUDA Docker image on Apple Silicon—it will not use the Metal GPU. Use native [vllm-metal](https://github.com/vllm-project/vllm-metal).

### Prerequisites

- Apple Silicon (M1–M4)
- Python 3.12+
- 32 GB unified memory recommended (Gemma 4 BF16 + draft speculative model)

### 1. Install

```bash
git clone https://github.com/Maximiliano-Villanueva/continuos-batching-bench.git
cd continuos-batching-bench

make install-vllm-metal          # server → .venv-vllm-metal/
python3 -m venv .venv
source .venv/bin/activate
make install-bench               # client → .venv/
make prompts-long                # ~8k-token prompts for E1/E3
```

### 2. Start the server (Terminal A)

```bash
./scripts/serve.sh gemma-4-e4b off
```

Wait for `http://127.0.0.1:8000/v1/models`. Keep this terminal open.

### 3. Run benchmarks (Terminal B)

```bash
source .venv/bin/activate
make serve-check
make smoke                       # optional connectivity check (~2–5 min)
make bench-gemma-off             # full suite (~5–6 h, ~8,730 requests)
```

### 4. Explore results

```bash
make dashboard                   # http://localhost:8501
make export                      # refresh CSV / Markdown exports
```

For speculative A/B: restart the server with `on`, then `make bench-gemma-on`. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the complete recipe.

---

## Experiments (E1–E7)

| ID | Question |
|----|----------|
| **E1** | How do long-only, mixed long/short, and short-only waves behave? |
| **E2** | Does long-prompt **position** in a concurrent wave affect tail latency? |
| **E3** | Reasoning (long output) vs plain long-context input |
| **E4** | Short-prompt throughput and latency vs client concurrency **K** (1–16) |
| **E5** | Short prompt with a large `max_tokens` budget |
| **E6** | **Chat** API vs **Completions** API |
| **E7** | Decode-heavy workload for speculative decoding comparison |

Experiments with both **sequential** (no overlap, K=1) and **concurrent** (K>1) modes isolate the effect of overlapping client requests. vLLM batches internally in all cases; the modes control how much load the client presents at once.

**Metrics glossary:** TTFT = time to first token · E2E = end-to-end latency · tok/s = output tokens per wall-clock second · K = max in-flight requests from this client.

---

## Models

Configured in [`configs/models.yaml`](configs/models.yaml):

| Key | Default checkpoint | Speculative strategy |
|-----|-------------------|----------------------|
| `gemma-4-e4b` | `mlx-community/gemma-4-e4b-it-bf16` | Draft model (`gemma-3-1b-it-qat-4bit`) |
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Native MTP (`qwen3_next_mtp`) |

**Gemma on Metal:** use the BF16 checkpoint above. The 4-bit Gemma MLX build currently fails on vllm-metal (KV-shared weights / missing multimodal processor files). `scripts/serve.sh` applies text-only flags automatically (`--limit-mm-per-prompt.* 0`, `--max-model-len 8192`).

**Why mlx-community?** vllm-metal expects MLX-converted weights on Apple Silicon. The `mlx-community` Hugging Face org publishes ready-made checkpoints so you can skip manual conversion.

---

## Command reference

| Command | Description |
|---------|-------------|
| `make install-vllm-metal` | Install vLLM + vllm-metal into `.venv-vllm-metal/` |
| `make install-bench` | Install `cb-bench` into `.venv/` |
| `./scripts/serve.sh <model> off\|on` | Start the inference server |
| `make serve-check` | Health-check `http://127.0.0.1:8000/v1/models` |
| `make smoke` | Fast smoke test (~70 requests) |
| `make bench-gemma-off` / `on` | Full E1–E7, Gemma, speculative off/on |
| `make bench-qwen-off` / `on` | Full E1–E7, Qwen, speculative off/on |
| `make resume RUN_ID=… SPEC=on ONLY=E7` | Resume a partial run |
| `make estimate` | Print expected request counts (rigorous profile) |
| `make dashboard` | Launch Streamlit UI |
| `make export` | Export CSV/Markdown for the latest run |
| `make test` | Unit tests (no server required) |

CLI equivalents:

```bash
.venv/bin/cb-bench run --model gemma-4-e4b --speculative off
.venv/bin/cb-bench run --resume-run-id <run_id> --only-experiment E7
.venv/bin/cb-bench export --run-id <run_id>
```

---

## Rigorous profile

The default full-suite profile is **`rigorous`** in [`configs/experiments.yaml`](configs/experiments.yaml):

| Setting | Value |
|---------|-------|
| Repetitions per scenario | 15 |
| E4 wave size (short prompts) | 48 per K |
| Concurrency sweep | K = 1, 2, 4, 8, 16 |
| E1 long+short mix | 25 prompts per repetition |
| Warmup per scenario | 3 (discarded, not stored) |

**~8,730 measured requests** per full run (~8,800 HTTP calls including warmup). Use `make estimate` for a per-experiment breakdown. Use `make smoke` only to verify connectivity—not for drawing conclusions.

---

## Output layout

```
results/<run_id>/
  benchmark.db      # SQLite (all metrics)
  requests.csv      # per-request export
  summary.csv       # aggregated per scenario
  summary.md        # auto-generated conclusions
  summary.html
```

`results/` is gitignored. Committed sample summaries live under [`docs/sample-results/`](docs/sample-results/) for readers who do not re-run benchmarks.

---

## Monitoring

| Signal | Source |
|--------|--------|
| TTFT / E2E | Client streaming timestamps |
| Throughput | `output_tokens / wall_time` |
| Memory | `psutil` RSS sampled every 500 ms |
| GPU (optional) | `sudo ./scripts/sample_powermetrics.sh 60` |

---

## Project structure

```
configs/              models, experiments, active run defaults
scenarios/prompts/    JSONL prompt fixtures
src/continuous_batching/
  application/        orchestrator, scheduler
  domain/             types, prompt loading, model registry
  infrastructure/     HTTP client, SQLite store, monitor
  evaluation/         statistics, auto-conclusions
  reporting/          dashboard, export
scripts/              serve.sh, install, health checks
docs/                 reproduction guide, walkthrough, sample results
tests/                unit + optional integration tests
```

---

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| `make serve-check` fails | Start `./scripts/serve.sh …` in Terminal A first |
| Dashboard shows no runs | Only directories with stored requests appear; run `make export` after a benchmark |
| `cb-bench` not found | Activate `.venv` or use Make targets (they invoke `.venv/bin/`) |
| Speculative server fails to start | Use `./scripts/serve.sh`—`--speculative-config` JSON quoting is handled there |
| CPU-only vLLM after install | Re-run `make install-vllm-metal` (installs the vllm-metal wheel explicitly) |
| Run interrupted | `make resume RUN_ID=<run_id> MODEL=gemma-4-e4b SPEC=on ONLY=E7` |

More detail: [docs/REPRODUCTION.md](docs/REPRODUCTION.md#troubleshooting).

---

## Tests

```bash
make test
pytest tests/integration -m integration   # requires a live vLLM server
```

---

## Scope and limitations

- **Platform:** Apple Silicon with [vllm-metal](https://github.com/vllm-project/vllm-metal) only. CUDA/Linux paths are out of scope.
- **Hardware:** Reported Gemma 4 numbers reflect a single rigorous profile on one machine class; absolute latencies will vary by chip and memory.
- **Server configuration:** Continuous batching and speculative decoding are controlled by `vllm serve`; this client measures behavior under fixed HTTP workloads.
- **Sample size:** Use the `rigorous` profile for conclusions. The `smoke` profile validates setup only.

---

## Further reading

- [Reproduction guide](docs/REPRODUCTION.md) — step-by-step Gemma and Qwen A/B runs
- [Architecture walkthrough](docs/CODE_WALKTHROUGH.md) — code layout and data flow
- [Sample results](docs/sample-results/) — committed Gemma 4 summaries
- [vllm-metal supported models](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)
