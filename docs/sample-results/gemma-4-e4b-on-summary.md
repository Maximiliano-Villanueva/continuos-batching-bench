# Gemma 4 E4B — speculative ON

Auto-generated conclusions from run `3f61df7ade09`.

| Field | Value |
|-------|-------|
| Model | `gemma-4-e4b` |
| Checkpoint | `mlx-community/gemma-4-e4b-it-bf16` |
| Speculative | `true` (draft: `mlx-community/gemma-3-1b-it-qat-4bit`) |
| Profile | `rigorous` |

Compare with [speculative OFF](gemma-4-e4b-off-summary.md) · [Results index](README.md)

---

## Conclusions

### E4 — concurrency sweep (short prompts)

- At K=2, throughput is −16.0% vs K=1 (15.6 vs 18.6 tok/s); E2E p99 = 804 ms
- At K=4, throughput is +53.1% vs K=1 (28.5 vs 18.6 tok/s); E2E p99 = 895 ms
- At K=8, throughput is +188.9% vs K=1 (53.8 vs 18.6 tok/s); E2E p99 = 978 ms
- At K=16, throughput is +421.5% vs K=1 (97.2 vs 18.6 tok/s); E2E p99 = 1,169 ms

### E2 — long-prompt position in concurrent wave

- `long_first`: long prompt E2E p99 = 13,720 ms; short prompts p99 = 3,847 ms
- `long_last`: long prompt E2E p99 = 16,958 ms; short prompts p99 = 1,379 ms

### E6 — API surface

- Chat API E2E p50 = 964 ms vs Completions p50 = 14,112 ms (−13,148 ms delta)

### E7 — decode-heavy workload

- `speculative_concurrent_k1`: 21.8 tok/s; E2E p50 = 11,464 ms
- `speculative_concurrent_k4`: 36.6 tok/s; E2E p50 = 26,585 ms
- `speculative_concurrent_k8`: 55.3 tok/s; E2E p50 = 26,865 ms
- `speculative_sequential_k1`: 22.2 tok/s; E2E p50 = 11,219 ms

### Caveats

- E1/`all_long` has only 15 successful requests (recommended minimum: 30). Increase repetitions or wave size before drawing firm conclusions on that scenario.
