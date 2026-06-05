from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from continuous_batching.domain.model_registry import ModelRegistry, load_model_registry
from continuous_batching.domain.models import RunConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_config(
    model_path: Path,
    experiments_path: Path | None = None,
    models_registry_path: Path | None = None,
    smoke: bool = False,
    results_dir: str = "results",
    model_key_override: str | None = None,
    speculative_override: bool | None = None,
    use_mlx_override: bool | None = None,
) -> tuple[RunConfig, dict[str, Any], ModelRegistry]:
    model_cfg = load_yaml(model_path)
    exp_cfg: dict[str, Any] = {}
    if experiments_path and experiments_path.exists():
        exp_cfg = load_yaml(experiments_path)

    registry_path = models_registry_path or model_path.parent / "models.yaml"
    registry = load_model_registry(registry_path)

    model_key = model_key_override or model_cfg.get("model_key") or registry.default_model_key
    entry = registry.get(model_key)

    use_mlx = (
        use_mlx_override
        if use_mlx_override is not None
        else bool(model_cfg.get("use_mlx_weights", True))
    )
    resolved_model = entry.resolve_hf_id(use_mlx)

    speculative = (
        speculative_override
        if speculative_override is not None
        else bool(model_cfg.get("speculative_enabled", False))
    )

    config = RunConfig(
        base_url=model_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
        model=resolved_model,
        model_key=model_key,
        speculative_enabled=speculative,
        default_max_tokens=int(model_cfg.get("default_max_tokens", 128)),
        long_output_max_tokens=int(model_cfg.get("long_output_max_tokens", 512)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 300)),
        results_dir=results_dir,
        smoke=smoke or bool(exp_cfg.get("smoke", False)),
        warmup_requests=int(exp_cfg.get("warmup_requests", 1)),
        repetitions=int(exp_cfg.get("repetitions", 3)),
        use_mlx_weights=use_mlx,
    )
    return config, exp_cfg, registry
