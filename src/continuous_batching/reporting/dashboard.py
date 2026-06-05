from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from continuous_batching.constants import RIGOROUS_MEASURED_REQUESTS
from continuous_batching.evaluation.conclusions import generate_conclusions
from continuous_batching.evaluation.stats import aggregate_scenario, compare_modes
from continuous_batching.infrastructure.store import ResultStore
from continuous_batching.paths import find_repo_root

_BOX_PLOT_HELP = (
    "**How to read box plots:** the line in the box is the median (typical request); "
    "the box covers the middle 50%; dots are outliers. **Lower is better** for latency."
)


def _chart_help(text: str) -> None:
    st.markdown(text)


def _render_glossary() -> None:
    with st.sidebar.expander("Metric glossary", expanded=False):
        st.markdown(
            """
**TTFT** — time until the first token appears (ms).  
**E2E** — total time until the full answer is done (ms).  
**p99** — 99th percentile; slowest 1% of requests (tail latency).  
**K** — concurrent requests in flight at once.  
**tok/s** — tokens generated per second (throughput).  
**sequential** — one request at a time (K=1).  
**concurrent** — many requests together (continuous batching).
            """
        )


def _list_runs(results_dir: Path) -> list[tuple[str, str]]:
    """Return (run_id, sidebar_label) for runs that contain request data, newest first."""
    if not results_dir.exists():
        return []
    runs: list[tuple[str, str, float]] = []
    for p in results_dir.iterdir():
        db_path = p / "benchmark.db"
        if not p.is_dir() or not db_path.is_file():
            continue
        store = ResultStore(db_path)
        count = store.count_request_results(p.name)
        meta = store.load_run_metadata(p.name)
        store.close()
        if count == 0:
            continue
        model_key = meta.get("model_key", "unknown")
        spec = meta.get("speculative_enabled")
        spec_label = "on" if spec else "off" if spec is not None else "?"
        profile = meta.get("profile", meta.get("smoke") and "smoke" or "rigorous")
        partial = (
            " [incomplete]"
            if profile == "rigorous" and count < RIGOROUS_MEASURED_REQUESTS
            else ""
        )
        label = f"{p.name} — {model_key} spec {spec_label} ({count:,} reqs){partial}"
        runs.append((p.name, label, p.stat().st_mtime))
    runs.sort(key=lambda item: item[2], reverse=True)
    return [(run_id, label) for run_id, label, _ in runs]


def main() -> None:
    st.set_page_config(page_title="Continuous Batching Benchmarks", layout="wide")
    st.title("Continuous Batching Benchmarks")
    st.caption("vLLM / vllm-metal on Apple Silicon")

    results_dir = find_repo_root() / "results"
    run_options = _list_runs(results_dir)
    if not run_options:
        st.warning(
            "No benchmark runs with stored request data were found. "
            "Run the benchmark client after starting the vLLM server (see README)."
        )
        return

    run_labels = {label: run_id for run_id, label in run_options}
    selected_label = st.sidebar.selectbox(
        "Run",
        options=[label for _, label in run_options],
        help="Only runs with stored requests are listed. Pick spec off vs on to compare.",
    )
    run_id = run_labels[selected_label]
    store = ResultStore(results_dir / run_id / "benchmark.db")
    requests = store.load_request_results(run_id)
    scenarios = store.load_scenario_runs(run_id)
    samples = store.load_system_samples(run_id)
    meta = store.load_run_metadata(run_id)
    store.close()

    _render_glossary()

    st.sidebar.markdown("**Run metadata**")
    st.sidebar.json(meta)

    if not requests:
        st.warning("No request data in this run.")
        return

    df = pd.DataFrame(requests)
    df["success"] = df["success"].astype(bool)

    tab_names = [
        "Overview",
        "E1 Long prompts",
        "E2 Position",
        "E3 Reasoning",
        "E4 K sweep",
        "E6 API",
        "E7 Speculative",
        "Memory",
        "Conclusions",
    ]
    tabs = st.tabs(tab_names)
    tab_overview = tabs[0]
    tab_e1 = tabs[1]
    tab_e2 = tabs[2]
    tab_e3 = tabs[3]
    tab_e4 = tabs[4]
    tab_e6 = tabs[5]
    tab_e7 = tabs[6]
    tab_mem = tabs[7]
    tab_conc = tabs[8]

    with tab_overview:
        st.subheader("Summary by scenario")
        _chart_help(
            "One row per test scenario. Use this table for exact numbers; charts below are "
            "the same data visualized. Compare **sequential** vs **concurrent** rows to see "
            "continuous batching impact."
        )
        summary_rows = []
        wall_map = {
            (
                r["experiment_id"],
                r["scenario_name"],
                r["execution_mode"],
                int(r["concurrency_k"]),
            ): float(r.get("wall_time_s") or 0)
            for r in scenarios
        }
        for key, grp in df.groupby(
            ["experiment_id", "scenario_name", "execution_mode", "concurrency_k"]
        ):
            wall = wall_map.get(key, 0.0)
            stats = aggregate_scenario(grp.to_dict("records"), wall_time_s=wall or None)
            row = {
                "experiment_id": key[0],
                "scenario_name": key[1],
                "execution_mode": key[2],
                "concurrency_k": key[3],
                **stats,
            }
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

        if "throughput_tok_s" in summary_df.columns:
            _chart_help(
                "**Throughput (tokens per second)** across all experiments. "
                "Taller bars mean the server produced more text per second in that scenario. "
                "Color = execution mode (sequential vs concurrent)."
            )
            fig = px.bar(
                summary_df,
                x="scenario_name",
                y="throughput_tok_s",
                color="execution_mode",
                facet_col="experiment_id",
                barmode="group",
                title="Throughput (tok/s) by scenario",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_e1:
        e1 = df[df["experiment_id"] == "E1"]
        if e1.empty:
            st.info("No E1 data.")
        else:
            _chart_help(
                "**Experiment 1 — long vs short prompts.** "
                "`all_long` = only long prompts; `all_short` = only short; "
                "`long_short_mix` = both in the same concurrent batch. "
                "Tests whether prompt length and mixing affect latency."
            )
            for metric in ("e2e_ms", "ttft_ms"):
                if metric == "e2e_ms":
                    _chart_help(
                        "**End-to-end latency (ms):** time from request sent until the "
                        "full response is finished."
                    )
                else:
                    _chart_help(
                        "**Time to first token (ms):** how long until the model starts "
                        "streaming the answer (ignores generation time after that)."
                    )
                st.caption(_BOX_PLOT_HELP)
                fig = px.box(
                    e1[e1[metric].notna()],
                    x="scenario_name",
                    y=metric,
                    color="execution_mode",
                    title=f"E1 — {metric}",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_e2:
        e2 = df[df["experiment_id"] == "E2"]
        if e2.empty:
            st.info("No E2 data.")
        else:
            _chart_help(
                "**Experiment 2 — prompt order in a batch.** "
                "`long_first` = long prompt first, then shorts; `long_last` = shorts first, "
                "long at the end. Color = prompt type. Question: does a long prompt at the "
                "front make short requests wait longer?"
            )
            st.caption(_BOX_PLOT_HELP)
            fig = px.box(
                e2,
                x="scenario_name",
                y="e2e_ms",
                color="prompt_class",
                title="E2 — E2E latency by order and prompt class",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_e3:
        e3 = df[df["experiment_id"] == "E3"]
        if e3.empty:
            st.info("No E3 data.")
        else:
            _chart_help(
                "**Experiment 3 — reasoning-style vs plain long prompts.** "
                "Compares chain-of-thought style prompts against equally long non-reasoning text."
            )
            st.caption(_BOX_PLOT_HELP)
            _chart_help("**End-to-end latency** by prompt type and execution mode.")
            fig = px.box(
                e3,
                x="prompt_class",
                y="e2e_ms",
                color="execution_mode",
                title="E3 — Reasoning vs long prompt (E2E)",
            )
            st.plotly_chart(fig, use_container_width=True)
            _chart_help(
                "**Output length (tokens):** how much text the model generated. "
                "Reasoning prompts often produce longer answers."
            )
            fig2 = px.box(
                e3,
                x="prompt_class",
                y="output_tokens",
                color="execution_mode",
                title="E3 — Output tokens by prompt class",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_e4:
        e4 = df[df["experiment_id"] == "E4"]
        if e4.empty:
            st.info("No E4 data.")
        else:
            _chart_help(
                "**Experiment 4 — concurrency sweep (core continuous-batching test).** "
                "Many short prompts with **K** = 1, 2, 4, 8, 16 simultaneous requests. "
                "Expect throughput to rise and per-request tail latency to worsen as K grows."
            )
            k_stats = []
            for k, grp in e4.groupby("concurrency_k"):
                stats = aggregate_scenario(grp.to_dict("records"))
                k_stats.append({"concurrency_k": k, **stats})
            kdf = pd.DataFrame(k_stats)
            _chart_help(
                "**Throughput vs K:** total tokens per second as concurrency increases. "
                "**Higher is better** — shows batching gains under load."
            )
            fig = px.line(
                kdf,
                x="concurrency_k",
                y="throughput_tok_s",
                markers=True,
                title="E4 — Throughput vs K",
            )
            st.plotly_chart(fig, use_container_width=True)
            _chart_help(
                "**Tail latency (E2E p99) vs K:** time for the slowest 1% of requests to finish. "
                "**Lower is better.** Rising line = busier server, slower worst-case latency."
            )
            fig2 = px.line(
                kdf,
                x="concurrency_k",
                y="e2e_p99",
                markers=True,
                title="E4 — E2E p99 vs K",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_e6:
        e6 = df[df["experiment_id"] == "E6"]
        if e6.empty:
            st.info("No E6 data.")
        else:
            _chart_help(
                "**Experiment 6 — Chat API vs Completions API.** "
                "Same short prompts sent via `/v1/chat/completions` (roles, assistant) "
                "vs `/v1/completions` (raw text). Large gaps are normal — different server paths."
            )
            st.caption(_BOX_PLOT_HELP)
            fig = px.box(e6, x="api_mode", y="e2e_ms", title="E6 — Chat vs Completions E2E")
            st.plotly_chart(fig, use_container_width=True)

    with tab_e7:
        e7 = df[df["experiment_id"] == "E7"]
        if e7.empty:
            st.info("No E7 data. Run with speculative on/off server profiles.")
        else:
            _chart_help(
                "**Experiment 7 — speculative decoding.** "
                "Decode-heavy workload at different K values. "
                "Pick **spec off** and **spec on** runs in the sidebar to compare; "
                "this tab shows one run at a time."
            )
            st.caption(_BOX_PLOT_HELP)
            _chart_help("**End-to-end latency** per scenario (sequential K=1 vs concurrent K>1).")
            fig = px.box(
                e7,
                x="scenario_name",
                y="e2e_ms",
                color="execution_mode",
                facet_col="prompt_class",
                title="E7 — Speculative decode benchmark (E2E)",
            )
            st.plotly_chart(fig, use_container_width=True)
            _chart_help("**Throughput (tok/s)** per E7 scenario. Higher = faster decoding under load.")
            tp_rows = []
            for (scenario, mode), grp in e7.groupby(["scenario_name", "execution_mode"]):
                stats = aggregate_scenario(grp.to_dict("records"))
                tp_rows.append(
                    {
                        "scenario_name": scenario,
                        "execution_mode": mode,
                        "throughput_tok_s": stats.get("throughput_tok_s", 0),
                    }
                )
            fig2 = px.bar(
                pd.DataFrame(tp_rows),
                x="scenario_name",
                y="throughput_tok_s",
                color="execution_mode",
                title="E7 — Throughput by scenario",
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_mem:
        if samples:
            _chart_help(
                "**Host memory (RSS)** sampled during each scenario. "
                "Shows RAM pressure on your Mac while the server runs — not GPU-only metrics."
            )
            sdf = pd.DataFrame(samples)
            fig = px.line(
                sdf,
                x="sampled_at",
                y="memory_rss_mb",
                color="scenario_name",
                title="Host RSS memory during scenarios",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No memory samples.")

    with tab_conc:
        st.subheader("Continuous batching impact")
        _chart_help(
            "Auto-generated takeaways from this run. For speculative decoding, "
            "switch runs in the sidebar and compare bullets side by side."
        )
        bullets = generate_conclusions(requests, scenarios, run_metadata=meta)
        for b in bullets:
            st.markdown(f"- {b}")

        st.subheader("Sequential vs concurrent (where available)")
        for exp_id in sorted(df["experiment_id"].unique()):
            exp_df = df[df["experiment_id"] == exp_id]
            for scenario in exp_df["scenario_name"].unique():
                mask_seq = (exp_df["scenario_name"] == scenario) & (
                    exp_df["execution_mode"] == "sequential"
                )
                mask_conc = (exp_df["scenario_name"] == scenario) & (
                    exp_df["execution_mode"] == "concurrent"
                )
                seq = exp_df[mask_seq]
                conc = exp_df[mask_conc]
                if seq.empty or conc.empty:
                    continue
                cmp = compare_modes(
                    aggregate_scenario(seq.to_dict("records")),
                    aggregate_scenario(conc.to_dict("records")),
                )
                st.write(f"**{exp_id}/{scenario}**", cmp)


if __name__ == "__main__":
    main()
