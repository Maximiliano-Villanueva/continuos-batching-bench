from __future__ import annotations

import asyncio
from collections.abc import Callable

import psutil

from continuous_batching.domain.models import SystemSample, utc_now


class SystemMonitor:
    """Samples host memory while a scenario runs."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._samples: list[SystemSample] = []
        self._meta: dict[str, str] = {}

    def start(self, run_id: str, experiment_id: str, scenario_name: str) -> None:
        self._samples = []
        self._meta = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "scenario_name": scenario_name,
        }
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[SystemSample]:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return list(self._samples)

    async def _loop(self) -> None:
        proc = psutil.Process()
        while True:
            mem = proc.memory_info()
            vm = psutil.virtual_memory()
            self._samples.append(
                SystemSample(
                    run_id=self._meta["run_id"],
                    experiment_id=self._meta["experiment_id"],
                    scenario_name=self._meta["scenario_name"],
                    sampled_at=utc_now(),
                    memory_rss_mb=mem.rss / (1024 * 1024),
                    memory_percent=vm.percent,
                )
            )
            await asyncio.sleep(self.interval_s)

    @staticmethod
    def peak_rss_mb(samples: list[SystemSample]) -> float | None:
        if not samples:
            return None
        return max(s.memory_rss_mb for s in samples)


async def run_with_monitor(
    monitor: SystemMonitor,
    run_id: str,
    experiment_id: str,
    scenario_name: str,
    coro_factory: Callable[[], asyncio.Future],
) -> tuple[object, list[SystemSample]]:
    monitor.start(run_id, experiment_id, scenario_name)
    try:
        result = await coro_factory()
        return result, await monitor.stop()
    except Exception:
        await monitor.stop()
        raise
