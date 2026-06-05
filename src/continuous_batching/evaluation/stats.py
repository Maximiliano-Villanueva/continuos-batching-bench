from __future__ import annotations

from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _successful(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("success", r.get("success", 0)) in (1, True)]


def aggregate_scenario(
    rows: list[dict[str, Any]],
    wall_time_s: float | None = None,
) -> dict[str, Any]:
    ok = _successful(rows)
    e2e = [float(r["e2e_ms"]) for r in ok if r.get("e2e_ms") is not None]
    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    out_tokens = sum(int(r.get("output_tokens", 0)) for r in ok)
    in_tokens = sum(int(r.get("input_tokens", 0)) for r in ok)

    if wall_time_s is None and ok:
        from datetime import datetime

        starts = [datetime.fromisoformat(r["started_at"]) for r in ok]
        ends = [datetime.fromisoformat(r["finished_at"]) for r in ok]
        wall_time_s = max(ends).timestamp() - min(starts).timestamp()
    wall_time_s = wall_time_s or 0.0

    return {
        "request_count": len(rows),
        "success_count": len(ok),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "throughput_tok_s": out_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        "throughput_req_s": len(ok) / wall_time_s if wall_time_s > 0 else 0.0,
        "ttft_p50": percentile(ttft, 50),
        "ttft_p95": percentile(ttft, 95),
        "ttft_p99": percentile(ttft, 99),
        "e2e_p50": percentile(e2e, 50),
        "e2e_p95": percentile(e2e, 95),
        "e2e_p99": percentile(e2e, 99),
        "wall_time_s": wall_time_s,
    }


def compare_modes(
    sequential: dict[str, Any],
    concurrent: dict[str, Any],
) -> dict[str, Any]:
    def delta_pct(concurrent_val: float, sequential_val: float) -> float | None:
        if sequential_val == 0:
            return None
        return ((concurrent_val - sequential_val) / sequential_val) * 100.0

    return {
        "throughput_tok_s_delta_pct": delta_pct(
            concurrent.get("throughput_tok_s", 0.0),
            sequential.get("throughput_tok_s", 0.0),
        ),
        "e2e_p99_delta_ms": (
            (concurrent.get("e2e_p99") or 0.0) - (sequential.get("e2e_p99") or 0.0)
            if concurrent.get("e2e_p99") is not None and sequential.get("e2e_p99") is not None
            else None
        ),
        "ttft_p99_delta_ms": (
            (concurrent.get("ttft_p99") or 0.0) - (sequential.get("ttft_p99") or 0.0)
            if concurrent.get("ttft_p99") is not None and sequential.get("ttft_p99") is not None
            else None
        ),
    }
