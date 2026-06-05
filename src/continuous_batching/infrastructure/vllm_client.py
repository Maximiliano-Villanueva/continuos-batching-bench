from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import AsyncOpenAI

from continuous_batching.domain.models import ApiMode, RequestSpec


@dataclass
class InferenceResponse:
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    e2e_ms: float
    inter_token_ms_avg: float | None
    success: bool
    error: str | None = None


class InferenceClient(ABC):
    @abstractmethod
    async def complete(self, spec: RequestSpec, max_tokens: int) -> InferenceResponse:
        pass


class VllmOpenAIClient(InferenceClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=timeout_seconds,
        )

    async def complete(self, spec: RequestSpec, max_tokens: int) -> InferenceResponse:
        start = time.perf_counter()
        try:
            if spec.api_mode == ApiMode.CHAT:
                return await self._chat(spec, max_tokens, start)
            return await self._completions(spec, max_tokens, start)
        except Exception as exc:  # noqa: BLE001
            e2e_ms = (time.perf_counter() - start) * 1000.0
            return InferenceResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                e2e_ms=e2e_ms,
                inter_token_ms_avg=None,
                success=False,
                error=str(exc),
            )

    async def _chat(
        self,
        spec: RequestSpec,
        max_tokens: int,
        start: float,
    ) -> InferenceResponse:
        ttft_ms: float | None = None
        chunks: list[str] = []
        chunk_times: list[float] = []
        last_chunk_time = start

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": spec.prompt.text}],
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if ttft_ms is None:
                    ttft_ms = (now - start) * 1000.0
                chunks.append(delta)
                chunk_times.append((now - last_chunk_time) * 1000.0)
                last_chunk_time = now

        text = "".join(chunks)
        e2e_ms = (time.perf_counter() - start) * 1000.0
        usage = _estimate_tokens(spec.prompt.text, text)
        inter_avg = sum(chunk_times) / len(chunk_times) if chunk_times else None
        return InferenceResponse(
            text=text,
            input_tokens=usage[0],
            output_tokens=usage[1],
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            inter_token_ms_avg=inter_avg,
            success=True,
        )

    async def _completions(
        self,
        spec: RequestSpec,
        max_tokens: int,
        start: float,
    ) -> InferenceResponse:
        ttft_ms: float | None = None
        chunks: list[str] = []
        chunk_times: list[float] = []
        last_chunk_time = start

        stream = await self._client.completions.create(
            model=self.model,
            prompt=spec.prompt.text,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            now = time.perf_counter()
            delta = chunk.choices[0].text if chunk.choices else None
            if delta:
                if ttft_ms is None:
                    ttft_ms = (now - start) * 1000.0
                chunks.append(delta)
                chunk_times.append((now - last_chunk_time) * 1000.0)
                last_chunk_time = now

        text = "".join(chunks)
        e2e_ms = (time.perf_counter() - start) * 1000.0
        usage = _estimate_tokens(spec.prompt.text, text)
        inter_avg = sum(chunk_times) / len(chunk_times) if chunk_times else None
        return InferenceResponse(
            text=text,
            input_tokens=usage[0],
            output_tokens=usage[1],
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            inter_token_ms_avg=inter_avg,
            success=True,
        )


def _estimate_tokens(prompt: str, completion: str) -> tuple[int, int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt)), len(enc.encode(completion))
    except Exception:  # noqa: BLE001
        return max(1, len(prompt) // 4), max(1, len(completion) // 4)


class MockInferenceClient(InferenceClient):
    """Deterministic client for unit tests."""

    def __init__(self, latency_ms: float = 10.0, fail_ids: set[str] | None = None) -> None:
        self.latency_ms = latency_ms
        self.fail_ids = fail_ids or set()
        self.call_log: list[RequestSpec] = []

    async def complete(self, spec: RequestSpec, max_tokens: int) -> InferenceResponse:
        import asyncio

        self.call_log.append(spec)
        await asyncio.sleep(self.latency_ms / 1000.0)
        if spec.request_id in self.fail_ids:
            return InferenceResponse(
                text="",
                input_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                e2e_ms=self.latency_ms,
                inter_token_ms_avg=None,
                success=False,
                error="mock failure",
            )
        out_tokens = min(max_tokens, 16)
        return InferenceResponse(
            text="mock response",
            input_tokens=32,
            output_tokens=out_tokens,
            ttft_ms=self.latency_ms * 0.3,
            e2e_ms=self.latency_ms,
            inter_token_ms_avg=self.latency_ms / max(out_tokens, 1),
            success=True,
        )
