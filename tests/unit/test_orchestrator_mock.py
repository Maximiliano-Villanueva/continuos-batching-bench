import pytest

from continuous_batching.application.orchestrator import ExperimentOrchestrator
from continuous_batching.config_loader import load_run_config
from continuous_batching.domain.models import ApiMode
from continuous_batching.infrastructure.vllm_client import MockInferenceClient


@pytest.mark.asyncio
async def test_smoke_run_with_mock(repo_root):
    config, _, _ = load_run_config(
        repo_root / "configs" / "model.yaml",
        repo_root / "configs" / "experiments.yaml",
        repo_root / "configs" / "models.yaml",
        smoke=True,
        results_dir=str(repo_root / "results_test"),
    )
    client = MockInferenceClient(latency_ms=1.0)
    orch = ExperimentOrchestrator(
        config=config,
        client=client,
        prompts_dir=repo_root / "scenarios" / "prompts",
        experiments_path=repo_root / "configs" / "experiments.yaml",
    )
    run_id = await orch.run_all()
    assert run_id
    assert (repo_root / "results_test" / run_id / "benchmark.db").exists()


@pytest.mark.asyncio
async def test_warmup_calls_client(repo_root):
    config, _, _ = load_run_config(
        repo_root / "configs" / "model.yaml",
        repo_root / "configs" / "experiments.yaml",
        repo_root / "configs" / "models.yaml",
        smoke=True,
    )
    config.warmup_requests = 2
    client = MockInferenceClient(latency_ms=0.1)
    orch = ExperimentOrchestrator(
        config=config,
        client=client,
        prompts_dir=repo_root / "scenarios" / "prompts",
        experiments_path=repo_root / "configs" / "experiments.yaml",
    )
    await orch._warmup(ApiMode.CHAT, None)
    assert len(client.call_log) == 2
