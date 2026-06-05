# Documentation

| Document | Audience | Contents |
|----------|----------|----------|
| [../README.md](../README.md) | Everyone | Project overview, quick start, command reference |
| [REPRODUCTION.md](REPRODUCTION.md) | Practitioners | End-to-end benchmark recipe, runtime estimates, troubleshooting |
| [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) | Contributors | Architecture, module map, data flow |
| [sample-results/](sample-results/) | Reviewers | Committed Gemma 4 auto-summaries (no GPU required) |

## Suggested paths

**I want to run benchmarks on my Mac** → start with [REPRODUCTION.md](REPRODUCTION.md).

**I want to understand the code** → read [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md), then `configs/experiments.yaml` and `src/continuous_batching/application/orchestrator.py`.

**I want to see results without re-running** → open [sample-results/README.md](sample-results/README.md).
