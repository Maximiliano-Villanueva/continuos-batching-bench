#!/usr/bin/env python3
"""Generate static charts for the README from local benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

import plotly.graph_objects as go
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OFF = ROOT / "results" / "740e0566d01c" / "summary.csv"
ON = ROOT / "results" / "3f61df7ade09" / "summary.csv"

CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, sans-serif", size=13),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


def _load(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _e4_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {
        int(r["concurrency_k"]): r
        for r in rows
        if r["experiment_id"] == "E4"
    }


def chart_e4_throughput() -> go.Figure:
    off, on = _e4_rows(_load(OFF)), _e4_rows(_load(ON))
    ks = sorted(off)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=[float(off[k]["throughput_tok_s"]) for k in ks],
            mode="lines+markers",
            name="Speculative off",
            line=dict(color="#636EFA", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=[float(on[k]["throughput_tok_s"]) for k in ks],
            mode="lines+markers",
            name="Speculative on",
            line=dict(color="#EF553B", width=2.5),
        )
    )
    fig.update_layout(
        title="E4 — Throughput vs client concurrency (K)",
        xaxis_title="Concurrent requests (K)",
        yaxis_title="Output tokens / second",
        **CHART_LAYOUT,
    )
    return fig


def chart_e4_p99() -> go.Figure:
    off, on = _e4_rows(_load(OFF)), _e4_rows(_load(ON))
    ks = sorted(off)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=[float(off[k]["e2e_p99"]) for k in ks],
            mode="lines+markers",
            name="Speculative off",
            line=dict(color="#636EFA", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=[float(on[k]["e2e_p99"]) for k in ks],
            mode="lines+markers",
            name="Speculative on",
            line=dict(color="#EF553B", width=2.5),
        )
    )
    fig.update_layout(
        title="E4 — Tail latency (E2E p99) vs K",
        xaxis_title="Concurrent requests (K)",
        yaxis_title="End-to-end latency p99 (ms)",
        **CHART_LAYOUT,
    )
    return fig


def chart_e6_api() -> go.Figure:
    off = _load(OFF)
    chat = next(r for r in off if r["experiment_id"] == "E6" and "chat" in r["scenario_name"])
    comp = next(r for r in off if r["experiment_id"] == "E6" and "completions" in r["scenario_name"])
    fig = go.Figure(
        go.Bar(
            x=["Chat API", "Completions API"],
            y=[float(chat["e2e_p50"]), float(comp["e2e_p50"])],
            marker_color=["#00CC96", "#AB63FA"],
            text=[f"{float(chat['e2e_p50']):,.0f} ms", f"{float(comp['e2e_p50']):,.0f} ms"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="E6 — Median latency by API (speculative off)",
        yaxis_title="E2E p50 (ms)",
        **CHART_LAYOUT,
    )
    return fig


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] if f == c else values[f] + (values[c] - values[f]) * (k - f)


def _e2_p99_by_class(db_path: Path) -> dict[str, dict[str, float]]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    out: dict[str, dict[str, float]] = {}
    for order in ("long_first", "long_last"):
        out[order] = {}
        for pc in ("long", "short"):
            rows = conn.execute(
                """
                SELECT e2e_ms FROM request_results
                WHERE experiment_id='E2' AND scenario_name=? AND prompt_class=? AND success=1
                """,
                (order, pc),
            ).fetchall()
            out[order][pc] = _percentile([r[0] for r in rows], 99)
    conn.close()
    return out


def chart_e2_position() -> go.Figure:
    db = ROOT / "results" / "740e0566d01c" / "benchmark.db"
    e2 = _e2_p99_by_class(db)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Short prompts p99",
            x=["Long first", "Long last"],
            y=[e2["long_first"]["short"], e2["long_last"]["short"]],
            marker_color="#636EFA",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Long prompt p99",
            x=["Long first", "Long last"],
            y=[e2["long_first"]["long"], e2["long_last"]["long"]],
            marker_color="#FFA15A",
        )
    )
    fig.update_layout(
        barmode="group",
        title="E2 — Who gets the latency? (p99 E2E, speculative off)",
        yaxis_title="Milliseconds",
        **CHART_LAYOUT,
    )
    return fig


def main() -> None:
    if not OFF.is_file() or not ON.is_file():
        raise SystemExit(
            "Missing results CSVs. Run benchmarks first or copy summary.csv into results/."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    charts = {
        "e4-throughput.png": chart_e4_throughput(),
        "e4-p99.png": chart_e4_p99(),
        "e6-api.png": chart_e6_api(),
        "e2-position.png": chart_e2_position(),
    }
    for name, fig in charts.items():
        path = OUT / name
        fig.write_image(str(path), width=900, height=480, scale=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
