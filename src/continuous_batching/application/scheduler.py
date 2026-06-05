from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from continuous_batching.domain.models import ExecutionMode, RequestSpec

T = TypeVar("T")


class RequestScheduler:
    """Schedules requests sequentially or with a concurrency cap (semaphore K)."""

    def __init__(self, concurrency_k: int, execution_mode: ExecutionMode) -> None:
        if concurrency_k < 1:
            raise ValueError("concurrency_k must be >= 1")
        self.concurrency_k = concurrency_k
        self.execution_mode = execution_mode

    async def run(
        self,
        requests: list[RequestSpec],
        handler: Callable[[RequestSpec], Awaitable[T]],
    ) -> list[T]:
        if self.execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._run_sequential(requests, handler)
        return await self._run_concurrent(requests, handler)

    async def _run_sequential(
        self,
        requests: list[RequestSpec],
        handler: Callable[[RequestSpec], Awaitable[T]],
    ) -> list[T]:
        results: list[T] = []
        for spec in requests:
            results.append(await handler(spec))
        return results

    async def _run_concurrent(
        self,
        requests: list[RequestSpec],
        handler: Callable[[RequestSpec], Awaitable[T]],
    ) -> list[T]:
        semaphore = asyncio.Semaphore(self.concurrency_k)
        max_in_flight = 0
        in_flight = 0
        lock = asyncio.Lock()

        async def wrapped(spec: RequestSpec) -> T:
            nonlocal max_in_flight, in_flight
            async with semaphore:
                async with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                try:
                    return await handler(spec)
                finally:
                    async with lock:
                        in_flight -= 1

        tasks = [asyncio.create_task(wrapped(spec)) for spec in requests]
        results = await asyncio.gather(*tasks)
        self.last_max_in_flight = max_in_flight
        return list(results)

    @staticmethod
    def build_request_order(
        prompt_ids: list[str],
        submit_order: list[str] | None = None,
    ) -> list[str]:
        """Return prompt ids in submission order (for E2 long-first vs long-last)."""
        if submit_order is None:
            return list(prompt_ids)
        order_index = {pid: i for i, pid in enumerate(submit_order)}
        return sorted(prompt_ids, key=lambda pid: order_index.get(pid, len(submit_order)))
