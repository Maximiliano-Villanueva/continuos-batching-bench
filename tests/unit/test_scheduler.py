import asyncio

import pytest

from continuous_batching.application.scheduler import RequestScheduler
from continuous_batching.domain.models import (
    ApiMode,
    ExecutionMode,
    PromptClass,
    PromptSpec,
    RequestSpec,
)
from continuous_batching.infrastructure.vllm_client import MockInferenceClient


def _spec(rid: str, pos: int = 0) -> RequestSpec:
    return RequestSpec(
        request_id=rid,
        prompt=PromptSpec(id=rid, prompt_class=PromptClass.SHORT, text="hi"),
        api_mode=ApiMode.CHAT,
        position_in_wave=pos,
        experiment_id="E4",
        scenario_name="test",
        execution_mode=ExecutionMode.CONCURRENT,
        concurrency_k=2,
    )


@pytest.mark.asyncio
async def test_sequential_preserves_order():
    client = MockInferenceClient(latency_ms=1.0)
    scheduler = RequestScheduler(1, ExecutionMode.SEQUENTIAL)
    specs = [_spec("a", 0), _spec("b", 1), _spec("c", 2)]

    async def handler(spec: RequestSpec):
        return await client.complete(spec, 8)

    results = await scheduler.run(specs, handler)
    assert [s.request_id for s in client.call_log] == ["a", "b", "c"]
    assert len(results) == 3


@pytest.mark.asyncio
async def test_concurrent_respects_semaphore():
    client = MockInferenceClient(latency_ms=50.0)
    scheduler = RequestScheduler(2, ExecutionMode.CONCURRENT)
    specs = [_spec(f"r{i}", i) for i in range(6)]

    in_flight = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def handler(spec: RequestSpec):
        nonlocal in_flight, max_seen
        async with lock:
            in_flight += 1
            max_seen = max(max_seen, in_flight)
        await client.complete(spec, 8)
        async with lock:
            in_flight -= 1
        return spec.request_id

    results = await scheduler.run(specs, handler)
    assert len(results) == 6
    assert max_seen <= 2


def test_build_request_order():
    order = RequestScheduler.build_request_order(
        ["short-1", "long-1", "short-2"],
        submit_order=["long-1", "short-1", "short-2"],
    )
    assert order[0] == "long-1"


def test_invalid_concurrency_raises():
    with pytest.raises(ValueError):
        RequestScheduler(0, ExecutionMode.CONCURRENT)
