# Gemma 4 E4B — speculative OFF

Auto-generated conclusions from run `740e0566d01c`.

| Field | Value |
|-------|-------|
| Model | `gemma-4-e4b` |
| Checkpoint | `mlx-community/gemma-4-e4b-it-bf16` |
| Speculative | `false` |
| Profile | `rigorous` |

Compare with [speculative ON](gemma-4-e4b-on-summary.md) · [Results index](README.md)

---

## Conclusions

### E4 — concurrency sweep (short prompts)

- At K=2, throughput is −15.5% vs K=1 (15.5 vs 18.3 tok/s); E2E p99 = 827 ms
- At K=4, throughput is +56.5% vs K=1 (28.7 vs 18.3 tok/s); E2E p99 = 915 ms
- At K=8, throughput is +201.9% vs K=1 (55.4 vs 18.3 tok/s); E2E p99 = 883 ms
- At K=16, throughput is +399.8% vs K=1 (91.7 vs 18.3 tok/s); E2E p99 = 1,398 ms

### E2 — long-prompt position in concurrent wave

- `long_first`: long prompt E2E p99 = 12,035 ms; short prompts p99 = 3,437 ms
- `long_last`: long prompt E2E p99 = 16,202 ms; short prompts p99 = 1,292 ms

### E6 — API surface

- Chat API E2E p50 = 941 ms vs Completions p50 = 13,864 ms (−12,923 ms delta)

### E7 — decode-heavy workload

- `speculative_concurrent_k1`: 21.4 tok/s; E2E p50 = 11,689 ms
- `speculative_concurrent_k4`: 36.0 tok/s; E2E p50 = 27,685 ms
- `speculative_concurrent_k8`: 54.5 tok/s; E2E p50 = 27,773 ms
- `speculative_sequential_k1`: 21.7 tok/s; E2E p50 = 11,303 ms

### Caveats

- E1/`all_long` has only 15 successful requests (recommended minimum: 30). Increase repetitions or wave size before drawing firm conclusions on that scenario.
