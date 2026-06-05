from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from continuous_batching.application.scheduler import RequestScheduler
from continuous_batching.domain.models import (
    ApiMode,
    ExecutionMode,
    PromptClass,
    PromptSpec,
    RequestResult,
    RequestSpec,
    RunConfig,
    ScenarioRun,
    utc_now,
)
from continuous_batching.domain.scenarios import build_wave, load_prompts, pick_prompts
from continuous_batching.infrastructure.metrics import fetch_prometheus_metrics
from continuous_batching.infrastructure.monitor import SystemMonitor
from continuous_batching.infrastructure.store import ResultStore
from continuous_batching.infrastructure.vllm_client import InferenceClient, InferenceResponse


class ExperimentOrchestrator:
    def __init__(
        self,
        config: RunConfig,
        client: InferenceClient,
        prompts_dir: Path,
        experiments_path: Path,
        *,
        resume_run_id: str | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.prompts = load_prompts(prompts_dir)
        self.experiments_path = experiments_path
        self.monitor = SystemMonitor()
        results_root = Path(config.results_dir)
        if resume_run_id:
            self.run_id = resume_run_id
            self.results_dir = results_root / self.run_id
            if not (self.results_dir / "benchmark.db").is_file():
                raise FileNotFoundError(
                    f"Cannot resume: no benchmark.db in {self.results_dir}"
                )
        else:
            self.run_id = uuid.uuid4().hex[:12]
            self.results_dir = results_root / self.run_id
        self.store = ResultStore.from_results_dir(results_root, self.run_id)
        self._run_meta: dict[str, Any] = {}
        self._root: dict[str, Any] = {}

    def load_experiments(self) -> dict[str, Any]:
        with self.experiments_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        profile_name = "smoke" if self.config.smoke else str(data.get("profile", "rigorous"))
        profiles = data.get("profiles") or {}
        if profile_name in profiles:
            overlay = profiles[profile_name]
            for key, value in overlay.items():
                if key != "experiments":
                    data[key] = value
            if overlay.get("smoke"):
                self.config = replace(self.config, smoke=True)

        if self.config.smoke:
            data["repetitions"] = int(data.get("repetitions", 1))
            data["warmup_requests"] = int(data.get("warmup_requests", 0))

        self._root = data
        self.config.repetitions = int(data.get("repetitions", self.config.repetitions))
        self.config.warmup_requests = int(data.get("warmup_requests", self.config.warmup_requests))
        return data

    def _cfg(self, key: str, default: Any) -> Any:
        return self._root.get(key, default)

    async def run_all(
        self,
        *,
        only_experiment_ids: list[str] | None = None,
        resume: bool = False,
    ) -> str:
        experiments = self.load_experiments()
        self._run_meta = {
            "model": self.config.model,
            "model_key": self.config.model_key,
            "speculative_enabled": self.config.speculative_enabled,
            "use_mlx_weights": self.config.use_mlx_weights,
            "base_url": self.config.base_url,
            "smoke": self.config.smoke,
            "profile": self._root.get("profile", "rigorous"),
            "repetitions": self.config.repetitions,
            "warmup_requests": self.config.warmup_requests,
            "min_samples_per_scenario": int(self._cfg("min_samples_per_scenario", 30)),
        }
        if resume:
            existing = self.store.load_run_metadata(self.run_id)
            if existing:
                self._run_meta = {**existing, **self._run_meta}
        self.store.save_run_metadata(self.run_id, self._run_meta)
        for exp in experiments.get("experiments", []):
            if not exp.get("enabled", True):
                continue
            if only_experiment_ids and exp["id"] not in only_experiment_ids:
                continue
            await self._run_experiment(exp, experiments)

        metrics_text = fetch_prometheus_metrics(self.config.base_url)
        if metrics_text:
            self._run_meta["prometheus_metrics"] = metrics_text[:50_000]
            self.store.save_run_metadata(self.run_id, self._run_meta)
        self.store.close()
        return self.run_id

    async def _run_experiment(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        exp_id = exp["id"]
        if exp_id == "E1":
            await self._run_e1(exp, root)
        elif exp_id == "E2":
            await self._run_e2(exp, root)
        elif exp_id == "E3":
            await self._run_e3(exp, root)
        elif exp_id == "E4":
            await self._run_e4(exp, root)
        elif exp_id == "E5":
            await self._run_e5(exp, root)
        elif exp_id == "E6":
            await self._run_e6(exp, root)
        elif exp_id == "E7":
            await self._run_e7(exp, root)

    def _wave_from_spec(
        self,
        wave_cfg: dict[str, Any],
        root: dict[str, Any],
        default_classes: list[str] | None = None,
    ) -> list[PromptSpec]:
        classes = wave_cfg.get("prompt_classes", default_classes or [])
        cycles = int(
            wave_cfg.get(
                "wave_cycles",
                self._cfg("e1_mix_cycles", 1) if "mix" in wave_cfg.get("name", "") else 1,
            )
        )
        target = wave_cfg.get("wave_size")
        if target is not None:
            return build_wave(self.prompts, classes, target_size=int(target))
        return build_wave(self.prompts, classes, cycles=cycles)

    async def _run_e1(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        for wave in exp.get("waves", []):
            if wave["name"] == "long_short_mix":
                wave = {
                    **wave,
                    "wave_cycles": wave.get(
                        "wave_cycles", self._cfg("e1_mix_cycles", 5)
                    ),
                }
            for mode_name in exp.get("modes", ["concurrent"]):
                mode = ExecutionMode(mode_name)
                k = 1 if mode == ExecutionMode.SEQUENTIAL else root.get(
                    "default_concurrency", 8
                )
                prompts = self._wave_from_spec(wave, root)
                await self._execute_scenario(
                    experiment_id=exp["id"],
                    scenario_name=wave["name"],
                    prompts=prompts,
                    mode=mode,
                    k=k,
                )

    async def _run_e2(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        k = exp.get("concurrency", root.get("default_concurrency", 8))
        cycles = int(self._cfg("e2_wave_cycles", 5))
        for order in exp.get("orders", []):
            prompts = build_wave(
                self.prompts,
                order["prompt_classes"],
                cycles=order.get("wave_cycles", cycles),
            )
            await self._execute_scenario(
                experiment_id=exp["id"],
                scenario_name=order["name"],
                prompts=prompts,
                mode=ExecutionMode.CONCURRENT,
                k=k,
            )

    async def _run_e3(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        size = int(self._cfg("e3_wave_size", 8))
        for mode_name in exp.get("modes", ["concurrent"]):
            mode = ExecutionMode(mode_name)
            k = 1 if mode == ExecutionMode.SEQUENTIAL else root.get(
                "default_concurrency", 8
            )
            for pc in exp.get("prompt_classes", []):
                prompts = build_wave(self.prompts, [pc], target_size=size)
                await self._execute_scenario(
                    experiment_id=exp["id"],
                    scenario_name=str(pc),
                    prompts=prompts,
                    mode=mode,
                    k=k,
                )

    async def _run_e4(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        k_values = exp.get("k_values", self._cfg("concurrency_sweep", [1, 2, 4, 8, 16]))
        wave_size = int(self._cfg("e4_wave_size", 48))
        wave = build_wave(self.prompts, exp.get("prompt_classes", ["short"]), target_size=wave_size)
        for k in k_values:
            await self._execute_scenario(
                experiment_id=exp["id"],
                scenario_name=f"short_prompt_k_sweep_k{k}",
                prompts=wave,
                mode=ExecutionMode.CONCURRENT,
                k=int(k),
            )

    async def _run_e5(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        mix_count = int(self._cfg("e5_short_mix_count", 12))
        for mode_name in exp.get("modes", ["concurrent"]):
            mode = ExecutionMode(mode_name)
            k = 1 if mode == ExecutionMode.SEQUENTIAL else root.get(
                "default_concurrency", 8
            )
            main = pick_prompts(self.prompts, ["short_long_output"])
            mixed = build_wave(self.prompts, exp.get("mixed_with", ["short"]), target_size=mix_count)
            prompts = main + mixed
            await self._execute_scenario(
                experiment_id=exp["id"],
                scenario_name="short_long_output",
                prompts=prompts,
                mode=mode,
                k=k,
                max_tokens_override=self.config.long_output_max_tokens,
            )

    async def _run_e7(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        k_values = exp.get("k_values", self._cfg("e7_k_values", [1, 4, 8]))
        wave_size = int(self._cfg("e7_wave_size", 24))
        short_n = wave_size // 2 + wave_size % 2
        short_w = build_wave(self.prompts, ["short"], target_size=short_n)
        long_w = build_wave(
            self.prompts, ["short_long_output"], target_size=wave_size - short_n
        )
        wave: list[PromptSpec] = []
        si = li = 0
        while len(wave) < wave_size:
            if si < len(short_w):
                wave.append(short_w[si])
                si += 1
            if len(wave) >= wave_size:
                break
            if li < len(long_w):
                wave.append(long_w[li])
                li += 1
        for mode_name in exp.get("modes", ["concurrent"]):
            mode = ExecutionMode(mode_name)
            for k in k_values:
                if mode == ExecutionMode.SEQUENTIAL and int(k) != 1:
                    continue
                effective_k = 1 if mode == ExecutionMode.SEQUENTIAL else int(k)
                await self._execute_scenario(
                    experiment_id=exp["id"],
                    scenario_name=f"speculative_{mode.value}_k{effective_k}",
                    prompts=wave,
                    mode=mode,
                    k=effective_k,
                    max_tokens_override=self.config.long_output_max_tokens,
                )

    async def _run_e6(self, exp: dict[str, Any], root: dict[str, Any]) -> None:
        k = root.get("default_concurrency", 8)
        wave_size = int(self._cfg("e6_wave_size", 32))
        wave = build_wave(
            self.prompts, exp.get("prompt_classes", ["short"]), target_size=wave_size
        )
        for api_name in exp.get("api_modes", ["chat", "completions"]):
            await self._execute_scenario(
                experiment_id=exp["id"],
                scenario_name=f"api_{api_name}",
                prompts=wave,
                mode=ExecutionMode.CONCURRENT,
                k=k,
                api_mode=ApiMode(api_name),
            )

    def _scenario_completed(
        self,
        experiment_id: str,
        scenario_name: str,
        mode: ExecutionMode,
        k: int,
    ) -> bool:
        cur = self.store._conn.execute(
            """
            SELECT 1 FROM scenario_runs
            WHERE run_id = ? AND experiment_id = ? AND scenario_name = ?
              AND execution_mode = ? AND concurrency_k = ?
            """,
            (self.run_id, experiment_id, scenario_name, mode.value, k),
        )
        return cur.fetchone() is not None

    async def _execute_scenario(
        self,
        experiment_id: str,
        scenario_name: str,
        prompts: list[PromptSpec],
        mode: ExecutionMode,
        k: int,
        api_mode: ApiMode = ApiMode.CHAT,
        max_tokens_override: int | None = None,
    ) -> None:
        if self._scenario_completed(experiment_id, scenario_name, mode, k):
            print(
                f"Skipping completed scenario {experiment_id}/{scenario_name} "
                f"({mode.value}, K={k})"
            )
            return
        reps = self.config.repetitions
        all_results: list[RequestResult] = []
        scenario = ScenarioRun(
            run_id=self.run_id,
            experiment_id=experiment_id,
            scenario_name=scenario_name,
            execution_mode=mode,
            concurrency_k=k,
            started_at=utc_now(),
        )
        await self._warmup(api_mode, max_tokens_override)
        self.monitor.start(self.run_id, experiment_id, scenario_name)

        for rep in range(reps):
            requests = self._build_requests(
                experiment_id,
                scenario_name,
                prompts,
                mode,
                k,
                api_mode,
                rep,
            )
            scheduler = RequestScheduler(concurrency_k=k, execution_mode=mode)

            async def handler(spec: RequestSpec) -> RequestResult:
                max_tok = (
                    max_tokens_override
                    or spec.prompt.max_tokens
                    or self.config.default_max_tokens
                )
                started = utc_now()
                response: InferenceResponse = await self.client.complete(spec, max_tok)
                finished = utc_now()
                return RequestResult(
                    request_id=spec.request_id,
                    run_id=self.run_id,
                    experiment_id=experiment_id,
                    scenario_name=scenario_name,
                    execution_mode=mode,
                    concurrency_k=k,
                    api_mode=spec.api_mode,
                    prompt_class=spec.prompt.prompt_class,
                    prompt_id=spec.prompt.id,
                    position_in_wave=spec.position_in_wave,
                    success=response.success,
                    error=response.error,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    ttft_ms=response.ttft_ms,
                    e2e_ms=response.e2e_ms,
                    inter_token_ms_avg=response.inter_token_ms_avg,
                    started_at=started,
                    finished_at=finished,
                )

            batch_results = await scheduler.run(requests, handler)
            all_results.extend(batch_results)

        samples = await self.monitor.stop()
        scenario.finished_at = utc_now()
        scenario.request_results = all_results
        scenario.system_samples = samples
        self.store.save_scenario_run(scenario)

    async def _warmup(
        self,
        api_mode: ApiMode,
        max_tokens_override: int | None,
    ) -> None:
        n = self.config.warmup_requests
        if n <= 0:
            return
        prompt = pick_prompts(self.prompts, ["short"])[0]
        max_tok = max_tokens_override or prompt.max_tokens or self.config.default_max_tokens
        for i in range(n):
            spec = RequestSpec(
                request_id=f"warmup-{i}",
                prompt=prompt,
                api_mode=api_mode,
                position_in_wave=-1,
                experiment_id="warmup",
                scenario_name="warmup",
                execution_mode=ExecutionMode.SEQUENTIAL,
                concurrency_k=1,
            )
            await self.client.complete(spec, min(max_tok, 16))

    def _build_requests(
        self,
        experiment_id: str,
        scenario_name: str,
        prompts: list[PromptSpec],
        mode: ExecutionMode,
        k: int,
        api_mode: ApiMode,
        rep: int,
    ) -> list[RequestSpec]:
        specs: list[RequestSpec] = []
        for pos, prompt in enumerate(prompts):
            specs.append(
                RequestSpec(
                    request_id=(
                        f"{experiment_id}-{scenario_name}-{mode.value}-r{rep}-p{pos}"
                    ),
                    prompt=prompt,
                    api_mode=api_mode,
                    position_in_wave=pos,
                    experiment_id=experiment_id,
                    scenario_name=scenario_name,
                    execution_mode=mode,
                    concurrency_k=k,
                )
            )
        return specs
