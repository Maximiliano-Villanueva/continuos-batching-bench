from __future__ import annotations

from pathlib import Path

import pandas as pd

from continuous_batching.evaluation.conclusions import generate_conclusions
from continuous_batching.evaluation.stats import aggregate_scenario
from continuous_batching.infrastructure.store import ResultStore


def export_run(store: ResultStore, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = store.load_request_results()
    scenarios = store.load_scenario_runs()
    samples = store.load_system_samples()

    if requests:
        pd.DataFrame(requests).to_csv(out_dir / "requests.csv", index=False)
    if samples:
        pd.DataFrame(samples).to_csv(out_dir / "system_samples.csv", index=False)

    summary_rows = []
    by_key: dict[tuple, list] = {}
    for row in requests:
        key = (
            row["experiment_id"],
            row["scenario_name"],
            row["execution_mode"],
            int(row["concurrency_k"]),
        )
        by_key.setdefault(key, []).append(row)

    wall_map = {
        (
            r["experiment_id"],
            r["scenario_name"],
            r["execution_mode"],
            int(r["concurrency_k"]),
        ): float(r.get("wall_time_s") or 0.0)
        for r in scenarios
    }

    for key, rows in sorted(by_key.items()):
        exp_id, scenario, mode, k = key
        wall = wall_map.get(key, 0.0)
        stats = aggregate_scenario(rows, wall_time_s=wall or None)
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "scenario_name": scenario,
                "execution_mode": mode,
                "concurrency_k": k,
                **stats,
            }
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    meta = store.load_run_metadata(requests[0]["run_id"]) if requests else {}
    bullets = generate_conclusions(requests, scenarios, run_metadata=meta)
    min_n = int(meta.get("min_samples_per_scenario", 0)) if meta else 0
    if min_n and summary_rows:
        for row in summary_rows:
            if row.get("success_count", 0) < min_n:
                bullets.append(
                    f"Caveat: {row['experiment_id']}/{row['scenario_name']} has only "
                    f"{row.get('success_count', 0)} successful requests "
                    f"(recommended minimum: {min_n}). "
                    "Increase repetitions or wave size before drawing conclusions."
                )
    md = ["# Benchmark conclusions\n", "\n"]
    for b in bullets:
        md.append(f"- {b}\n")
    (out_dir / "summary.md").write_text("".join(md), encoding="utf-8")

    if summary_rows:
        html_df = pd.DataFrame(summary_rows)
        (out_dir / "summary.html").write_text(
            _summary_html(html_df, bullets),
            encoding="utf-8",
        )


def _summary_html(summary_df: pd.DataFrame, bullets: list[str]) -> str:
    table = summary_df.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Benchmark Summary</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h1>Continuous Batching Benchmark Summary</h1>
<h2>Metrics</h2>{table}
<h2>Conclusions</h2><ul>{items}</ul>
</body></html>"""
