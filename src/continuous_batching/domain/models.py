from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ApiMode(StrEnum):
    CHAT = "chat"
    COMPLETIONS = "completions"


class ExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"


class PromptClass(StrEnum):
    SHORT = "short"
    LONG = "long"
    REASONING = "reasoning"
    SHORT_LONG_OUTPUT = "short_long_output"


@dataclass(frozen=True)
class PromptSpec:
    id: str
    prompt_class: PromptClass
    text: str
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt: PromptSpec
    api_mode: ApiMode
    position_in_wave: int
    experiment_id: str
    scenario_name: str
    execution_mode: ExecutionMode
    concurrency_k: int


@dataclass
class RequestResult:
    request_id: str
    run_id: str
    experiment_id: str
    scenario_name: str
    execution_mode: ExecutionMode
    concurrency_k: int
    api_mode: ApiMode
    prompt_class: PromptClass
    prompt_id: str
    position_in_wave: int
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    e2e_ms: float
    inter_token_ms_avg: float | None
    started_at: datetime
    finished_at: datetime

    def to_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "scenario_name": self.scenario_name,
            "execution_mode": self.execution_mode.value,
            "concurrency_k": self.concurrency_k,
            "api_mode": self.api_mode.value,
            "prompt_class": self.prompt_class.value,
            "prompt_id": self.prompt_id,
            "position_in_wave": self.position_in_wave,
            "success": int(self.success),
            "error": self.error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "inter_token_ms_avg": self.inter_token_ms_avg,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }


@dataclass
class SystemSample:
    run_id: str
    experiment_id: str
    scenario_name: str
    sampled_at: datetime
    memory_rss_mb: float
    memory_percent: float

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "scenario_name": self.scenario_name,
            "sampled_at": self.sampled_at.isoformat(),
            "memory_rss_mb": self.memory_rss_mb,
            "memory_percent": self.memory_percent,
        }


@dataclass
class ScenarioRun:
    run_id: str
    experiment_id: str
    scenario_name: str
    execution_mode: ExecutionMode
    concurrency_k: int
    started_at: datetime
    finished_at: datetime | None = None
    request_results: list[RequestResult] = field(default_factory=list)
    system_samples: list[SystemSample] = field(default_factory=list)

    @property
    def wall_time_s(self) -> float:
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()


@dataclass
class RunConfig:
    base_url: str
    model: str
    model_key: str
    speculative_enabled: bool
    default_max_tokens: int
    long_output_max_tokens: int
    timeout_seconds: float
    results_dir: str = "results"
    smoke: bool = False
    warmup_requests: int = 1
    repetitions: int = 3
    use_mlx_weights: bool = True


def utc_now() -> datetime:
    return datetime.now(UTC)
