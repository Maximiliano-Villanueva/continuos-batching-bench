import os

import httpx
import pytest

from continuous_batching.domain.models import (
    ApiMode,
    ExecutionMode,
    PromptClass,
    PromptSpec,
    RequestSpec,
)
from continuous_batching.infrastructure.vllm_client import VllmOpenAIClient


def _server_available() -> bool:
    base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")] + "/v1/models"
    else:
        url = url + "/models"
    try:
        r = httpx.get(url, timeout=5.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_chat_completion():
    if not _server_available():
        pytest.skip("vLLM server not running")

    base = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.environ.get("VLLM_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")
    client = VllmOpenAIClient(base_url=base, model=model, timeout_seconds=120.0)
    spec = RequestSpec(
        request_id="integration-1",
        prompt=PromptSpec(id="s1", prompt_class=PromptClass.SHORT, text="Say hi in three words."),
        api_mode=ApiMode.CHAT,
        position_in_wave=0,
        experiment_id="INT",
        scenario_name="smoke",
        execution_mode=ExecutionMode.SEQUENTIAL,
        concurrency_k=1,
    )
    resp = await client.complete(spec, max_tokens=16)
    assert resp.success, resp.error
    assert resp.output_tokens > 0
