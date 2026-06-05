from pathlib import Path

from continuous_batching.domain.models import (
    ApiMode,
    ExecutionMode,
    PromptClass,
    RequestResult,
    ScenarioRun,
    utc_now,
)
from continuous_batching.infrastructure.store import ResultStore


def test_store_roundtrip(tmp_path: Path):
    store = ResultStore(tmp_path / "bench.db")
    run_id = "testrun"
    now = utc_now()
    result = RequestResult(
        request_id="r1",
        run_id=run_id,
        experiment_id="E4",
        scenario_name="k_sweep",
        execution_mode=ExecutionMode.CONCURRENT,
        concurrency_k=4,
        api_mode=ApiMode.CHAT,
        prompt_class=PromptClass.SHORT,
        prompt_id="short-1",
        position_in_wave=0,
        success=True,
        error=None,
        input_tokens=10,
        output_tokens=20,
        ttft_ms=5.0,
        e2e_ms=100.0,
        inter_token_ms_avg=4.0,
        started_at=now,
        finished_at=now,
    )
    scenario = ScenarioRun(
        run_id=run_id,
        experiment_id="E4",
        scenario_name="k_sweep",
        execution_mode=ExecutionMode.CONCURRENT,
        concurrency_k=4,
        started_at=now,
        finished_at=now,
        request_results=[result],
    )
    store.save_run_metadata(run_id, {"model": "test"})
    store.save_scenario_run(scenario)
    rows = store.load_request_results(run_id)
    store.close()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "r1"
