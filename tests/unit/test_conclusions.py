from continuous_batching.evaluation.conclusions import generate_conclusions


def test_generate_conclusions_e4_k_sweep():
    requests = [
        {
            "experiment_id": "E4",
            "scenario_name": "short_prompt_k_sweep_k1",
            "execution_mode": "concurrent",
            "concurrency_k": 1,
            "success": 1,
            "e2e_ms": 100.0,
            "ttft_ms": 10.0,
            "output_tokens": 50,
            "input_tokens": 10,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:10+00:00",
        },
        {
            "experiment_id": "E4",
            "scenario_name": "short_prompt_k_sweep_k1",
            "execution_mode": "concurrent",
            "concurrency_k": 4,
            "success": 1,
            "e2e_ms": 150.0,
            "ttft_ms": 15.0,
            "output_tokens": 200,
            "input_tokens": 40,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:05+00:00",
        },
    ]
    scenarios = [
        {
            "experiment_id": "E4",
            "scenario_name": "short_prompt_k_sweep_k1",
            "execution_mode": "concurrent",
            "concurrency_k": 1,
            "wall_time_s": 10.0,
        },
        {
            "experiment_id": "E4",
            "scenario_name": "short_prompt_k_sweep_k1",
            "execution_mode": "concurrent",
            "concurrency_k": 4,
            "wall_time_s": 5.0,
        },
    ]
    bullets = generate_conclusions(requests, scenarios)
    assert any("E4" in b for b in bullets)
