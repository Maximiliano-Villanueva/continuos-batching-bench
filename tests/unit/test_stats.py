from continuous_batching.evaluation.stats import aggregate_scenario, compare_modes, percentile


def test_percentile_empty():
    assert percentile([], 50) is None


def test_percentile_single():
    assert percentile([42.0], 99) == 42.0


def test_percentile_interpolation():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0
    p50 = percentile(vals, 50)
    assert p50 is not None
    assert 2.0 <= p50 <= 4.0


def test_aggregate_scenario_throughput():
    rows = [
        {
            "success": 1,
            "e2e_ms": 100.0,
            "ttft_ms": 20.0,
            "output_tokens": 50,
            "input_tokens": 10,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:10+00:00",
        },
        {
            "success": 1,
            "e2e_ms": 200.0,
            "ttft_ms": 40.0,
            "output_tokens": 50,
            "input_tokens": 10,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:10+00:00",
        },
    ]
    stats = aggregate_scenario(rows, wall_time_s=10.0)
    assert stats["success_count"] == 2
    assert stats["output_tokens"] == 100
    assert stats["throughput_tok_s"] == 10.0
    assert stats["throughput_req_s"] == 0.2
    assert stats["e2e_p50"] == 150.0


def test_compare_modes():
    seq = {"throughput_tok_s": 100.0, "e2e_p99": 500.0, "ttft_p99": 100.0}
    conc = {"throughput_tok_s": 150.0, "e2e_p99": 800.0, "ttft_p99": 120.0}
    cmp = compare_modes(seq, conc)
    assert cmp["throughput_tok_s_delta_pct"] == 50.0
    assert cmp["e2e_p99_delta_ms"] == 300.0
