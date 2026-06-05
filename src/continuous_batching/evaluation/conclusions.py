from __future__ import annotations

from typing import Any

from continuous_batching.evaluation.stats import aggregate_scenario, compare_modes


def generate_conclusions(
    request_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> list[str]:
    bullets: list[str] = []

    if run_metadata:
        bullets.append(
            f"Run context: model_key={run_metadata.get('model_key')}, "
            f"speculative_enabled={run_metadata.get('speculative_enabled')}, "
            f"checkpoint={run_metadata.get('model')}"
        )

    by_key: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in request_rows:
        key = (
            row["experiment_id"],
            row["scenario_name"],
            row["execution_mode"],
            int(row["concurrency_k"]),
        )
        by_key.setdefault(key, []).append(row)

    def _wall_key(r: dict) -> tuple:
        return (
            r["experiment_id"],
            r["scenario_name"],
            r["execution_mode"],
            int(r["concurrency_k"]),
        )

    wall_by_key = {_wall_key(r): float(r.get("wall_time_s") or 0.0) for r in scenario_rows}

    # E4: throughput vs K
    e4_rows = [r for r in request_rows if r["experiment_id"] == "E4"]
    if e4_rows:
        by_k: dict[int, dict[str, Any]] = {}
        for k in sorted({int(r["concurrency_k"]) for r in e4_rows}):
            subset = [r for r in e4_rows if int(r["concurrency_k"]) == k]
            wall = wall_by_key.get(
                ("E4", f"short_prompt_k_sweep_k{k}", "concurrent", k),
                0.0,
            )
            by_k[k] = aggregate_scenario(subset, wall_time_s=wall or None)
        k1 = by_k.get(1, {}).get("throughput_tok_s", 0.0)
        for k, stats in by_k.items():
            if k == 1:
                continue
            tok_s = stats.get("throughput_tok_s", 0.0)
            if k1 > 0:
                pct = ((tok_s - k1) / k1) * 100
                p99 = stats.get("e2e_p99")
                bullets.append(
                    f"E4: At K={k}, throughput is {pct:+.1f}% vs K=1 "
                    f"({tok_s:.1f} vs {k1:.1f} tok/s); E2E p99={p99:.0f}ms"
                    if p99
                    else f"E4: At K={k}, throughput is {pct:+.1f}% vs K=1."
                )

    # CB impact: sequential vs concurrent per experiment
    experiments = sorted({r["experiment_id"] for r in request_rows})
    for exp_id in experiments:
        exp_rows = [r for r in request_rows if r["experiment_id"] == exp_id]
        scenarios = sorted({r["scenario_name"] for r in exp_rows})
        for scenario in scenarios:
            seq = [
                r
                for r in exp_rows
                if r["scenario_name"] == scenario and r["execution_mode"] == "sequential"
            ]
            conc = [
                r
                for r in exp_rows
                if r["scenario_name"] == scenario and r["execution_mode"] == "concurrent"
            ]
            if not seq or not conc:
                continue
            k = int(conc[0]["concurrency_k"])
            seq_wall = wall_by_key.get((exp_id, scenario, "sequential", 1), 0.0)
            conc_wall = wall_by_key.get((exp_id, scenario, "concurrent", k), 0.0)
            seq_stats = aggregate_scenario(seq, wall_time_s=seq_wall or None)
            conc_stats = aggregate_scenario(conc, wall_time_s=conc_wall or None)
            cmp = compare_modes(seq_stats, conc_stats)
            tp = cmp.get("throughput_tok_s_delta_pct")
            p99 = cmp.get("e2e_p99_delta_ms")
            if tp is not None:
                bullets.append(
                    f"{exp_id}/{scenario}: concurrent vs sequential throughput {tp:+.1f}%"
                    + (f", E2E p99 {p99:+.0f}ms" if p99 is not None else "")
                )

    # E2: long first vs long last
    e2_rows = [r for r in request_rows if r["experiment_id"] == "E2"]
    for order in ("long_first", "long_last"):
        subset = [r for r in e2_rows if r["scenario_name"] == order]
        if not subset:
            continue
        long_pos = [r for r in subset if r["prompt_class"] == "long"]
        short_pos = [r for r in subset if r["prompt_class"] == "short"]
        if long_pos and short_pos:
            from continuous_batching.evaluation.stats import percentile

            long_p99 = percentile([float(r["e2e_ms"]) for r in long_pos], 99)
            short_p99 = percentile([float(r["e2e_ms"]) for r in short_pos], 99)
            if long_p99 is not None and short_p99 is not None:
                bullets.append(
                    f"E2/{order}: long prompt E2E p99={long_p99:.0f}ms, "
                    f"short prompts p99={short_p99:.0f}ms"
                )

    # E6: chat vs completions
    e6 = [r for r in request_rows if r["experiment_id"] == "E6"]
    if e6:
        chat = [r for r in e6 if r["api_mode"] == "chat"]
        comp = [r for r in e6 if r["api_mode"] == "completions"]
        if chat and comp:
            chat_stats = aggregate_scenario(chat)
            comp_stats = aggregate_scenario(comp)
            e2e_chat = chat_stats.get("e2e_p50") or 0
            e2e_comp = comp_stats.get("e2e_p50") or 0
            bullets.append(
                f"E6: Chat API E2E p50={e2e_chat:.0f}ms vs Completions p50={e2e_comp:.0f}ms "
                f"({e2e_chat - e2e_comp:+.0f}ms delta)"
            )

    e7 = [r for r in request_rows if r["experiment_id"] == "E7"]
    if e7:
        for scenario in sorted({r["scenario_name"] for r in e7}):
            subset = [r for r in e7 if r["scenario_name"] == scenario]
            stats = aggregate_scenario(subset)
            bullets.append(
                f"E7/{scenario}: throughput {stats.get('throughput_tok_s', 0):.1f} tok/s, "
                f"E2E p50={stats.get('e2e_p50', 0):.0f}ms"
            )

    if not bullets:
        bullets.append(
            "No conclusions available. Run the benchmark suite first (see README)."
        )

    return bullets
